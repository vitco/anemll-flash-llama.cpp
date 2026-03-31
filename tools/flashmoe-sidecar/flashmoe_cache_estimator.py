#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Flash-MoE persistent bank cost and trace coverage from a sidecar manifest.",
    )
    parser.add_argument("--sidecar", required=True, type=Path, help="sidecar directory or manifest path")
    parser.add_argument("--trace", type=Path, help="optional JSONL trace captured with --moe-trace")
    parser.add_argument(
        "--banks",
        type=int,
        action="append",
        default=[],
        help="per-layer persistent bank sizes to evaluate; pass multiple times",
    )
    parser.add_argument(
        "--byte-budget-gib",
        type=float,
        action="append",
        default=[],
        help="global static persistent-cache budgets in GiB; pass multiple times",
    )
    parser.add_argument(
        "--token-count",
        type=int,
        help="override inferred token count for trace-based metrics",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=5,
        help="how many highest-miss layers to print for each simulated policy",
    )
    parser.add_argument(
        "--watch",
        type=float,
        help="refresh the terminal dashboard every N seconds until interrupted",
    )
    parser.add_argument("--svg-out", type=Path, help="optional output path for a self-contained SVG dashboard")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args()


def load_manifest(sidecar_path: Path) -> tuple[Path, dict[str, Any]]:
    sidecar_path = sidecar_path.expanduser().resolve()
    manifest_path = sidecar_path / "manifest.json" if sidecar_path.is_dir() else sidecar_path
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("entries"), list):
        raise SystemExit(f"manifest '{manifest_path}' is missing an entries array")
    return manifest_path, manifest


def dedupe_preserve(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


@dataclass
class TraceSummary:
    token_count: int
    total_layer_calls: int
    total_expert_requests: int
    total_misses: int
    total_full_hit_calls: int
    total_bytes_read: int
    layer_misses: dict[int, int]
    layer_calls: dict[int, int]

    @property
    def expert_hit_rate(self) -> float:
        if self.total_expert_requests == 0:
            return 0.0
        return 1.0 - (self.total_misses / float(self.total_expert_requests))

    @property
    def full_hit_rate(self) -> float:
        if self.total_layer_calls == 0:
            return 0.0
        return self.total_full_hit_calls / float(self.total_layer_calls)

    @property
    def misses_per_token(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.total_misses / float(self.token_count)

    @property
    def mib_per_token(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.total_bytes_read / float(self.token_count) / float(1024 * 1024)


def build_layer_geometry(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    layers: dict[int, dict[str, Any]] = collections.defaultdict(
        lambda: {
            "families": {},
            "slot_bytes": 0,
            "total_bytes": 0,
            "expert_count": None,
        }
    )
    for entry in entries:
        layer = int(entry["layer"])
        family = str(entry["tensor_family"])
        bpe = entry.get("bytes_per_expert")
        if bpe is None:
            raise SystemExit(f"manifest entry '{entry['tensor_name']}' is missing bytes_per_expert")
        slot_bytes = int(bpe)
        exact_bytes = int(entry["exact_byte_length"])
        expert_count = entry.get("shape", [None, None, None])[-1]
        layer_info = layers[layer]
        layer_info["families"][family] = slot_bytes
        layer_info["slot_bytes"] += slot_bytes
        layer_info["total_bytes"] += exact_bytes
        if expert_count is not None:
            layer_info["expert_count"] = int(expert_count)
    return dict(sorted(layers.items()))


def parse_trace(
    trace_path: Path,
) -> tuple[dict[int, list[list[int]]], dict[int, collections.Counter[int]], dict[int, int], list[str]]:
    trace_path = trace_path.expanduser().resolve()
    calls_by_layer: dict[int, list[list[int]]] = collections.defaultdict(list)
    freq_by_layer: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    token_sums_by_layer: dict[int, int] = collections.Counter()
    warnings: list[str] = []

    with trace_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            layer = int(record["layer"])
            n_tokens = max(0, int(record.get("n_tokens", 0)))
            experts_raw = [int(value) for value in record.get("experts", [])]
            requested = dedupe_preserve(experts_raw)
            if not requested:
                continue
            calls_by_layer[layer].append(requested)
            freq_by_layer[layer].update(requested)
            token_sums_by_layer[layer] += n_tokens
            if n_tokens == 0:
                warnings.append(f"trace line {line_no} has n_tokens=0 for layer {layer}")

    return dict(sorted(calls_by_layer.items())), freq_by_layer, dict(token_sums_by_layer), warnings


def infer_token_count(token_sums_by_layer: dict[int, int], override: int | None) -> tuple[int, list[str]]:
    warnings: list[str] = []
    if override is not None:
        return override, warnings
    if not token_sums_by_layer:
        return 0, warnings
    values = [value for value in token_sums_by_layer.values() if value > 0]
    if not values:
        return 0, warnings
    inferred = max(values)
    if len(set(values)) > 1:
        warnings.append(
            f"trace token totals differ by layer; using max cumulative tokens {inferred} as the workload token count"
        )
    return inferred, warnings


def resident_bytes_for_bank(layer_geometry: dict[int, dict[str, Any]], bank_size: int) -> int:
    return sum(int(info["slot_bytes"]) * bank_size for info in layer_geometry.values())


def _simulate_static_sets(
    calls_by_layer: dict[int, list[list[int]]],
    layer_geometry: dict[int, dict[str, Any]],
    resident_sets: dict[int, set[int]],
    token_count: int,
) -> TraceSummary:
    total_layer_calls = 0
    total_expert_requests = 0
    total_misses = 0
    total_full_hit_calls = 0
    total_bytes_read = 0
    layer_misses: dict[int, int] = collections.Counter()
    layer_calls: dict[int, int] = collections.Counter()

    for layer, calls in calls_by_layer.items():
        resident = resident_sets.get(layer, set())
        slot_bytes = int(layer_geometry[layer]["slot_bytes"])
        for requested in calls:
            total_layer_calls += 1
            layer_calls[layer] += 1
            total_expert_requests += len(requested)
            missing = [expert_id for expert_id in requested if expert_id not in resident]
            if not missing:
                total_full_hit_calls += 1
            total_misses += len(missing)
            layer_misses[layer] += len(missing)
            total_bytes_read += len(missing) * slot_bytes

    return TraceSummary(
        token_count=token_count,
        total_layer_calls=total_layer_calls,
        total_expert_requests=total_expert_requests,
        total_misses=total_misses,
        total_full_hit_calls=total_full_hit_calls,
        total_bytes_read=total_bytes_read,
        layer_misses=dict(layer_misses),
        layer_calls=dict(layer_calls),
    )


def simulate_uniform_static(
    calls_by_layer: dict[int, list[list[int]]],
    freq_by_layer: dict[int, collections.Counter[int]],
    layer_geometry: dict[int, dict[str, Any]],
    token_count: int,
    bank_size: int,
) -> tuple[TraceSummary, dict[int, set[int]]]:
    resident_sets: dict[int, set[int]] = {}
    for layer, counter in freq_by_layer.items():
        resident_sets[layer] = {expert_id for expert_id, _ in counter.most_common(bank_size)}
    return _simulate_static_sets(calls_by_layer, layer_geometry, resident_sets, token_count), resident_sets


def simulate_lru(
    calls_by_layer: dict[int, list[list[int]]],
    layer_geometry: dict[int, dict[str, Any]],
    token_count: int,
    bank_size: int,
) -> TraceSummary:
    total_layer_calls = 0
    total_expert_requests = 0
    total_misses = 0
    total_full_hit_calls = 0
    total_bytes_read = 0
    layer_misses: dict[int, int] = collections.Counter()
    layer_calls: dict[int, int] = collections.Counter()

    for layer, calls in calls_by_layer.items():
        resident: collections.OrderedDict[int, None] = collections.OrderedDict()
        slot_bytes = int(layer_geometry[layer]["slot_bytes"])
        for requested in calls:
            total_layer_calls += 1
            layer_calls[layer] += 1
            total_expert_requests += len(requested)
            requested_set = set(requested)
            missing = [expert_id for expert_id in requested if expert_id not in resident]
            if not missing:
                total_full_hit_calls += 1
            for expert_id in requested:
                if expert_id in resident:
                    resident.move_to_end(expert_id)
            while len(resident) + len(missing) > bank_size and resident:
                oldest = next(iter(resident))
                if oldest in requested_set:
                    resident.move_to_end(oldest)
                    if all(expert_id in requested_set for expert_id in resident.keys()):
                        break
                    continue
                resident.popitem(last=False)
            for expert_id in missing:
                if len(resident) >= bank_size:
                    evicted = False
                    for candidate in list(resident.keys()):
                        if candidate not in requested_set:
                            resident.pop(candidate)
                            evicted = True
                            break
                    if not evicted:
                        break
                resident[expert_id] = None
            total_misses += len(missing)
            layer_misses[layer] += len(missing)
            total_bytes_read += len(missing) * slot_bytes

    return TraceSummary(
        token_count=token_count,
        total_layer_calls=total_layer_calls,
        total_expert_requests=total_expert_requests,
        total_misses=total_misses,
        total_full_hit_calls=total_full_hit_calls,
        total_bytes_read=total_bytes_read,
        layer_misses=dict(layer_misses),
        layer_calls=dict(layer_calls),
    )


def simulate_global_static(
    calls_by_layer: dict[int, list[list[int]]],
    freq_by_layer: dict[int, collections.Counter[int]],
    layer_geometry: dict[int, dict[str, Any]],
    token_count: int,
    budget_bytes: int,
) -> tuple[TraceSummary, int, dict[int, set[int]]]:
    candidates: list[tuple[float, int, int, int, int]] = []
    for layer, counter in freq_by_layer.items():
        slot_bytes = int(layer_geometry[layer]["slot_bytes"])
        for expert_id, count in counter.items():
            ratio = float(count) / float(slot_bytes)
            candidates.append((ratio, count, -slot_bytes, layer, expert_id))

    candidates.sort(reverse=True)
    used = 0
    resident_sets: dict[int, set[int]] = collections.defaultdict(set)
    for _, _, _, layer, expert_id in candidates:
        slot_bytes = int(layer_geometry[layer]["slot_bytes"])
        if used + slot_bytes > budget_bytes:
            continue
        used += slot_bytes
        resident_sets[layer].add(expert_id)

    summary = _simulate_static_sets(calls_by_layer, layer_geometry, dict(resident_sets), token_count)
    return summary, used, dict(resident_sets)


def top_layer_rows(summary: TraceSummary, top_layers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, misses in sorted(summary.layer_misses.items(), key=lambda item: (-item[1], item[0]))[:top_layers]:
        calls = max(1, int(summary.layer_calls.get(layer, 0)))
        rows.append(
            {
                "layer": layer,
                "misses": int(misses),
                "calls": int(summary.layer_calls.get(layer, 0)),
                "misses_per_call": float(misses) / float(calls),
            }
        )
    return rows


def coverage_rows(
    freq_by_layer: dict[int, collections.Counter[int]],
    layer_geometry: dict[int, dict[str, Any]],
    banks: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bank_size in banks:
        layer_coverages: list[float] = []
        weighted_hits = 0
        weighted_total = 0
        for layer, counter in freq_by_layer.items():
            total = sum(counter.values())
            if total <= 0:
                continue
            hits = sum(count for _, count in counter.most_common(bank_size))
            layer_coverages.append(float(hits) / float(total))
            weighted_hits += hits
            weighted_total += total
        rows.append(
            {
                "bank_size": bank_size,
                "resident_gib": resident_bytes_for_bank(layer_geometry, bank_size) / float(1024 ** 3),
                "avg_layer_coverage": 0.0 if not layer_coverages else sum(layer_coverages) / float(len(layer_coverages)),
                "weighted_coverage": 0.0 if weighted_total == 0 else float(weighted_hits) / float(weighted_total),
            }
        )
    return rows


def ascii_bar(value: float, max_value: float, width: int = 28, full: str = "#", empty: str = ".") -> str:
    if width <= 0:
        return ""
    if max_value <= 0.0:
        return empty * width
    ratio = max(0.0, min(1.0, value / max_value))
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return (full * filled) + (empty * (width - filled))


def pct_bar(value: float, width: int = 28) -> str:
    return ascii_bar(value, 1.0, width=width)


def metric_bar_lower_better(value: float, worst_value: float, width: int = 28) -> str:
    if worst_value <= 0.0:
        return "#" * width
    score = 1.0 - max(0.0, min(1.0, value / worst_value))
    return pct_bar(score, width=width)


def best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row[key])


def format_pct(value: float) -> str:
    return f"{value * 100.0:5.1f}%"


def print_visual_series(
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    label_fmt: str,
    value_key: str,
    suffix: str,
    max_value: float | None = None,
) -> None:
    print(title)
    if not rows:
        print("  (no data)")
        return
    if max_value is None:
        max_value = max(float(row[value_key]) for row in rows)
    for row in rows:
        label = label_fmt.format(row[label_key])
        value = float(row[value_key])
        print(f"  {label} |{ascii_bar(value, max_value)}| {value:.2f}{suffix}")


def print_visual_pct_series(
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    label_fmt: str,
    value_key: str,
) -> None:
    print(title)
    if not rows:
        print("  (no data)")
        return
    for row in rows:
        value = float(row[value_key])
        label = label_fmt.format(row[label_key])
        print(f"  {label} |{pct_bar(value)}| {format_pct(value)}")


def print_visual_policy_rows(
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    label_fmt: str,
) -> None:
    print(title)
    if not rows:
        print("  (no data)")
        return
    worst_mib = max(float(row["mib_per_token"]) for row in rows)
    for row in rows:
        label = label_fmt.format(row[label_key])
        hit_rate = float(row["expert_hit_rate"])
        full_hit = float(row["full_hit_rate"])
        miss_mib = float(row["mib_per_token"])
        miss_token = float(row["misses_per_token"])
        print(
            f"  {label} |{pct_bar(hit_rate)}| hit {format_pct(hit_rate)} "
            f"full {format_pct(full_hit)} miss {miss_token:6.1f}/tok "
            f"io {miss_mib:7.1f} MiB/tok |{metric_bar_lower_better(miss_mib, worst_mib, width=14)}|"
        )


def print_takeaways(trace: dict[str, Any]) -> None:
    print("takeaways:")
    best_uniform = best_row(trace["uniform_static"], "expert_hit_rate")
    best_lru = best_row(trace["lru"], "expert_hit_rate")
    best_global = best_row(trace["global_static"], "expert_hit_rate")

    if best_uniform is not None:
        print(
            f"  best uniform per-layer bank in this sweep: "
            f"{best_uniform['bank_size']} slots -> {best_uniform['resident_gib']:.2f} GiB, "
            f"{format_pct(best_uniform['expert_hit_rate'])} expert-hit"
        )
    if best_lru is not None:
        print(
            f"  best refillable LRU in this sweep: "
            f"{best_lru['bank_size']} slots -> {best_lru['resident_gib']:.2f} GiB, "
            f"{format_pct(best_lru['expert_hit_rate'])} expert-hit"
        )
    if best_global is not None:
        print(
            f"  best global static budget in this sweep: "
            f"{best_global['budget_gib']:.1f} GiB -> {best_global['used_gib']:.2f} GiB used, "
            f"{format_pct(best_global['expert_hit_rate'])} expert-hit"
        )


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(x: float, y: float, text: str, size: int = 14, weight: str = "normal", fill: str = "#1f2937") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Menlo, SFMono-Regular, ui-monospace, monospace" '
        f'font-weight="{weight}" fill="{fill}">{svg_escape(text)}</text>'
    )


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: int = 6) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" rx="{rx}" ry="{rx}"/>'
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#9ca3af", width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"/>'
    )


def draw_bar_chart(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    rows: list[dict[str, Any]],
    label_fn,
    value_key: str,
    value_fmt,
    color: str,
    max_value: float | None = None,
) -> str:
    parts: list[str] = [
        svg_rect(x, y, w, h, fill="#ffffff", stroke="#d1d5db", rx=10),
        svg_text(x + 16, y + 28, title, size=16, weight="600"),
    ]
    if not rows:
        parts.append(svg_text(x + 16, y + 56, "(no data)", size=13, fill="#6b7280"))
        return "\n".join(parts)

    chart_x = x + 54
    chart_y = y + 44
    chart_w = w - 72
    chart_h = h - 86
    if max_value is None:
        max_value = max(float(row[value_key]) for row in rows) or 1.0

    parts.append(svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, stroke="#cbd5e1"))
    parts.append(svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, stroke="#cbd5e1"))

    bar_gap = 10.0
    bar_w = max(10.0, (chart_w - bar_gap * (len(rows) + 1)) / float(len(rows)))
    for idx, row in enumerate(rows):
        value = float(row[value_key])
        bar_h = 0.0 if max_value <= 0.0 else (value / max_value) * (chart_h - 20.0)
        bx = chart_x + bar_gap + idx * (bar_w + bar_gap)
        by = chart_y + chart_h - bar_h
        parts.append(svg_rect(bx, by, bar_w, bar_h, fill=color, rx=4))
        parts.append(svg_text(bx, chart_y + chart_h + 18, str(label_fn(row)), size=11, fill="#374151"))
        parts.append(svg_text(bx, max(chart_y + 14, by - 6), value_fmt(value), size=11, fill="#111827"))

    return "\n".join(parts)


def draw_dual_line_chart(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    label_key: str,
    label_fmt: str,
    value_key: str,
    color_a: str,
    color_b: str,
    legend_a: str,
    legend_b: str,
) -> str:
    parts: list[str] = [
        svg_rect(x, y, w, h, fill="#ffffff", stroke="#d1d5db", rx=10),
        svg_text(x + 16, y + 28, title, size=16, weight="600"),
    ]
    if not rows_a or not rows_b:
        parts.append(svg_text(x + 16, y + 56, "(no data)", size=13, fill="#6b7280"))
        return "\n".join(parts)

    chart_x = x + 44
    chart_y = y + 48
    chart_w = w - 68
    chart_h = h - 92
    parts.append(svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, stroke="#cbd5e1"))
    parts.append(svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, stroke="#cbd5e1"))

    def points(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
        pts = []
        count = max(1, len(rows) - 1)
        for idx, row in enumerate(rows):
            px = chart_x + (chart_w * idx / float(count))
            value = max(0.0, min(1.0, float(row[value_key])))
            py = chart_y + chart_h - (value * chart_h)
            pts.append((px, py, label_fmt.format(row[label_key])))
        return pts

    pts_a = points(rows_a)
    pts_b = points(rows_b)
    parts.append(
        f'<polyline fill="none" stroke="{color_a}" stroke-width="3" points="'
        + " ".join(f"{px:.1f},{py:.1f}" for px, py, _ in pts_a)
        + '"/>'
    )
    parts.append(
        f'<polyline fill="none" stroke="{color_b}" stroke-width="3" points="'
        + " ".join(f"{px:.1f},{py:.1f}" for px, py, _ in pts_b)
        + '"/>'
    )
    for px, py, label in pts_a:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color_a}"/>')
        parts.append(svg_text(px - 8, chart_y + chart_h + 18, label, size=11, fill="#374151"))
    for px, py, _ in pts_b:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color_b}"/>')

    parts.append(svg_rect(x + w - 180, y + 16, 12, 12, fill=color_a, rx=2))
    parts.append(svg_text(x + w - 162, y + 27, legend_a, size=11))
    parts.append(svg_rect(x + w - 88, y + 16, 12, 12, fill=color_b, rx=2))
    parts.append(svg_text(x + w - 70, y + 27, legend_b, size=11))
    return "\n".join(parts)


def write_svg_dashboard(payload: dict[str, Any], out_path: Path) -> None:
    model = payload["model"]
    geometry = payload["resident_geometry"]
    trace = payload.get("trace")

    width = 1360
    height = 980 if trace is not None else 380
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, fill="#f8fafc", rx=0),
        svg_text(32, 42, "Flash-MoE Cache Estimator Dashboard", size=28, weight="700", fill="#0f172a"),
        svg_text(
            32,
            68,
            f"model={model.get('arch', 'unknown')} experts={model.get('expert_count', 'unknown')} "
            f"native_topk={model.get('expert_used_count', 'unknown')} routed_layers={payload['layers']}",
            size=13,
            fill="#475569",
        ),
    ]

    parts.append(
        draw_bar_chart(
            24,
            96,
            640,
            280,
            "Uniform persistent bank cost",
            geometry["uniform_bank_costs"],
            label_fn=lambda row: row["bank_size"],
            value_key="resident_gib",
            value_fmt=lambda value: f"{value:.1f} GiB",
            color="#2563eb",
        )
    )

    if trace is not None:
        parts.append(
            draw_bar_chart(
                696,
                96,
                640,
                280,
                "Weighted hot-expert coverage",
                trace["coverage"],
                label_fn=lambda row: row["bank_size"],
                value_key="weighted_coverage",
                value_fmt=lambda value: f"{value * 100.0:.0f}%",
                color="#0f766e",
                max_value=1.0,
            )
        )
        parts.append(
            draw_dual_line_chart(
                24,
                408,
                640,
                280,
                "Expert-hit rate by bank",
                trace["uniform_static"],
                trace["lru"],
                label_key="bank_size",
                label_fmt="{:>3}",
                value_key="expert_hit_rate",
                color_a="#7c3aed",
                color_b="#ea580c",
                legend_a="uniform static",
                legend_b="LRU",
            )
        )
        parts.append(
            draw_bar_chart(
                696,
                408,
                640,
                280,
                "Global static budget expert-hit",
                trace["global_static"],
                label_fn=lambda row: f"{row['budget_gib']:.0f}",
                value_key="expert_hit_rate",
                value_fmt=lambda value: f"{value * 100.0:.0f}%",
                color="#16a34a",
                max_value=1.0,
            )
        )

        y = 724
        parts.append(svg_rect(24, y, 1312, 216, fill="#ffffff", stroke="#d1d5db", rx=10))
        parts.append(svg_text(40, y + 28, "Current snapshot", size=16, weight="600"))
        parts.append(svg_text(40, y + 58, f"trace={trace['path']}", size=12, fill="#475569"))
        parts.append(svg_text(40, y + 82, f"inferred token_count={trace['token_count']}", size=12, fill="#475569"))
        if trace["warnings"]:
            parts.append(svg_text(40, y + 106, trace["warnings"][0], size=12, fill="#b45309"))
        best_uniform = best_row(trace["uniform_static"], "expert_hit_rate")
        best_lru = best_row(trace["lru"], "expert_hit_rate")
        best_global = best_row(trace["global_static"], "expert_hit_rate")
        lines = []
        if best_uniform is not None:
            lines.append(
                f"best uniform static: {best_uniform['bank_size']} slots, "
                f"{best_uniform['resident_gib']:.2f} GiB, {best_uniform['expert_hit_rate'] * 100.0:.1f}% hit"
            )
        if best_lru is not None:
            lines.append(
                f"best LRU: {best_lru['bank_size']} slots, "
                f"{best_lru['resident_gib']:.2f} GiB, {best_lru['expert_hit_rate'] * 100.0:.1f}% hit"
            )
        if best_global is not None:
            lines.append(
                f"best global static in sweep: {best_global['budget_gib']:.0f} GiB budget, "
                f"{best_global['used_gib']:.2f} GiB used, {best_global['expert_hit_rate'] * 100.0:.1f}% hit"
            )
        for idx, line in enumerate(lines):
            parts.append(svg_text(40, y + 140 + idx * 24, line, size=14, fill="#0f172a"))

    parts.append("</svg>")
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path, manifest = load_manifest(args.sidecar)
    entries = manifest["entries"]
    model = manifest.get("model", {})
    layer_geometry = build_layer_geometry(entries)

    banks = sorted(set(args.banks or [4, 8, 16, 32, 64]))
    budgets_gib = sorted(set(args.byte_budget_gib or [8.0, 16.0, 24.0, 32.0]))

    payload: dict[str, Any] = {
        "sidecar": str(manifest_path),
        "model": model,
        "layers": len(layer_geometry),
        "resident_geometry": {
            "per_layer_slot_bytes_min": min(int(info["slot_bytes"]) for info in layer_geometry.values()),
            "per_layer_slot_bytes_max": max(int(info["slot_bytes"]) for info in layer_geometry.values()),
            "per_layer_slot_bytes_avg": sum(int(info["slot_bytes"]) for info in layer_geometry.values()) / float(len(layer_geometry)),
            "uniform_bank_costs": [
                {
                    "bank_size": bank_size,
                    "resident_bytes": resident_bytes_for_bank(layer_geometry, bank_size),
                    "resident_gib": resident_bytes_for_bank(layer_geometry, bank_size) / float(1024 ** 3),
                }
                for bank_size in banks
            ],
        },
        "trace": None,
    }

    if args.trace is None:
        return payload

    calls_by_layer, freq_by_layer, token_sums_by_layer, trace_warnings = parse_trace(args.trace)
    token_count, token_warnings = infer_token_count(token_sums_by_layer, args.token_count)

    trace_payload: dict[str, Any] = {
        "path": str(args.trace.expanduser().resolve()),
        "layers": len(calls_by_layer),
        "token_count": token_count,
        "token_sums_by_layer": token_sums_by_layer,
        "warnings": trace_warnings + token_warnings,
        "unique_experts": {
            str(layer): len(counter)
            for layer, counter in sorted(freq_by_layer.items())
        },
        "coverage": coverage_rows(freq_by_layer, layer_geometry, banks),
        "uniform_static": [],
        "lru": [],
        "global_static": [],
    }

    for bank_size in banks:
        static_summary, _ = simulate_uniform_static(calls_by_layer, freq_by_layer, layer_geometry, token_count, bank_size)
        trace_payload["uniform_static"].append(
            {
                "bank_size": bank_size,
                "resident_bytes": resident_bytes_for_bank(layer_geometry, bank_size),
                "resident_gib": resident_bytes_for_bank(layer_geometry, bank_size) / float(1024 ** 3),
                "expert_hit_rate": static_summary.expert_hit_rate,
                "full_hit_rate": static_summary.full_hit_rate,
                "misses_per_token": static_summary.misses_per_token,
                "mib_per_token": static_summary.mib_per_token,
                "top_miss_layers": top_layer_rows(static_summary, args.top_layers),
            }
        )

        lru_summary = simulate_lru(calls_by_layer, layer_geometry, token_count, bank_size)
        trace_payload["lru"].append(
            {
                "bank_size": bank_size,
                "resident_bytes": resident_bytes_for_bank(layer_geometry, bank_size),
                "resident_gib": resident_bytes_for_bank(layer_geometry, bank_size) / float(1024 ** 3),
                "expert_hit_rate": lru_summary.expert_hit_rate,
                "full_hit_rate": lru_summary.full_hit_rate,
                "misses_per_token": lru_summary.misses_per_token,
                "mib_per_token": lru_summary.mib_per_token,
                "top_miss_layers": top_layer_rows(lru_summary, args.top_layers),
            }
        )

    for budget_gib in budgets_gib:
        budget_bytes = int(budget_gib * (1024 ** 3))
        summary, used_bytes, resident_sets = simulate_global_static(
            calls_by_layer,
            freq_by_layer,
            layer_geometry,
            token_count,
            budget_bytes,
        )
        trace_payload["global_static"].append(
            {
                "budget_gib": budget_gib,
                "budget_bytes": budget_bytes,
                "used_bytes": used_bytes,
                "used_gib": used_bytes / float(1024 ** 3),
                "resident_pairs": sum(len(layer_set) for layer_set in resident_sets.values()),
                "expert_hit_rate": summary.expert_hit_rate,
                "full_hit_rate": summary.full_hit_rate,
                "misses_per_token": summary.misses_per_token,
                "mib_per_token": summary.mib_per_token,
                "top_miss_layers": top_layer_rows(summary, args.top_layers),
            }
        )

    payload["trace"] = trace_payload
    return payload


def print_payload(payload: dict[str, Any]) -> None:
    model = payload["model"]
    print(f"model arch: {model.get('arch', 'unknown')}")
    print(
        f"experts: {model.get('expert_count', 'unknown')} total, "
        f"native top-k: {model.get('expert_used_count', 'unknown')}, "
        f"routed layers: {payload['layers']}"
    )

    geometry = payload["resident_geometry"]
    print("resident geometry:")
    print(
        "  per-slot bytes: "
        f"min={geometry['per_layer_slot_bytes_min']} "
        f"max={geometry['per_layer_slot_bytes_max']} "
        f"avg={geometry['per_layer_slot_bytes_avg']:.2f}"
    )
    print_visual_series(
        "  uniform per-layer persistent bank cost:",
        geometry["uniform_bank_costs"],
        label_key="bank_size",
        label_fmt="bank={:>3}",
        value_key="resident_gib",
        suffix=" GiB",
    )

    trace = payload.get("trace")
    if trace is None:
        return

    print("trace summary:")
    print(f"  path: {trace['path']}")
    print(f"  token_count: {trace['token_count']}")
    if trace["warnings"]:
        for warning in trace["warnings"]:
            print(f"  warning: {warning}")

    print_visual_pct_series(
        "  weighted hot-expert coverage by uniform static bank:",
        trace["coverage"],
        label_key="bank_size",
        label_fmt="bank={:>3}",
        value_key="weighted_coverage",
    )

    print_visual_policy_rows(
        "  uniform per-layer static simulation:",
        trace["uniform_static"],
        label_key="bank_size",
        label_fmt="bank={:>3}",
    )

    print_visual_policy_rows(
        "  refillable LRU simulation:",
        trace["lru"],
        label_key="bank_size",
        label_fmt="bank={:>3}",
    )

    print_visual_policy_rows(
        "  global static simulation under byte budget:",
        trace["global_static"],
        label_key="budget_gib",
        label_fmt="{:>4.0f} GiB",
    )

    print_takeaways(trace)


def main() -> int:
    args = parse_args()
    if args.watch is not None and args.watch <= 0:
        raise SystemExit("--watch must be greater than 0")
    if args.watch is not None and args.json:
        raise SystemExit("--watch is only supported with the terminal dashboard output, not --json")

    def run_once() -> None:
        payload = build_payload(args)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=False))
        else:
            print_payload(payload)
        if args.svg_out is not None:
            write_svg_dashboard(payload, args.svg_out)
            if not args.json:
                print(f"svg dashboard: {args.svg_out.expanduser().resolve()}")

    if args.watch is None:
        run_once()
        return 0

    try:
        while True:
            print("\033[2J\033[H", end="")
            run_once()
            print()
            print(f"watching every {args.watch:.1f}s - Ctrl+C to stop")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
