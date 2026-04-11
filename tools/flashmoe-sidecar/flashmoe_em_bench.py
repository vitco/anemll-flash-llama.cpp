#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from flashmoe_sidecar import load_manifest  # type: ignore

FAMILIES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps")


@dataclass
class BenchResult:
    mode: str
    tokens: int
    layer_calls: int
    expert_requests: int
    reads: int
    bytes_read: int
    wall_s: float
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def gib_read(self) -> float:
        return self.bytes_read / float(1024 ** 3)

    @property
    def mib_per_read(self) -> float:
        if self.reads == 0:
            return 0.0
        return self.bytes_read / float(self.reads) / float(1024 ** 2)

    @property
    def gib_per_s(self) -> float:
        if self.wall_s <= 0:
            return 0.0
        return self.gib_read / self.wall_s

    @property
    def ms_per_token(self) -> float:
        if self.tokens == 0:
            return 0.0
        return self.wall_s * 1000.0 / float(self.tokens)

    @property
    def tok_per_s(self) -> float:
        if self.wall_s <= 0:
            return 0.0
        return float(self.tokens) / self.wall_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype trace-driven read benchmarks for layer-major and expert-major Flash-MoE sidecars."
    )
    parser.add_argument("--layer-sidecar", required=True, type=Path, help="layer-major sidecar directory or manifest")
    parser.add_argument("--em-sidecar", required=True, type=Path, help="expert-major sidecar directory or manifest")
    parser.add_argument("--trace", required=True, type=Path, help="JSONL trace captured with --moe-trace")
    parser.add_argument(
        "--mode",
        action="append",
        choices=(
            "layer-exact",
            "em-exact",
            "em-bundle",
            "em-oracle-exact",
            "em-oracle-span",
            "em-oracle-next-layer",
            "em-oracle-next-layer-bundle",
            "em-oracle-hole1-bundle",
            "em-oracle-hole2-bundle",
        ),
        help="benchmark mode; may be passed multiple times (default: all)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        action="append",
        default=[],
        help="bundle horizon in routed layers for em-bundle; may be passed multiple times",
    )
    parser.add_argument("--token-limit", type=int, help="limit to the first N tokens in the trace")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="open files with macOS F_NOCACHE so reads bypass the page cache as much as possible",
    )
    parser.add_argument(
        "--drop-cache-between-runs",
        action="store_true",
        help="best-effort page-cache drop before each mode using purge(8) on macOS",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args()


def split_trace_tokens(trace_path: Path, token_limit: int | None) -> list[list[dict[str, Any]]]:
    tokens: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev_layer: int | None = None
    with trace_path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            layer = int(record["layer"])
            if prev_layer is not None and layer < prev_layer:
                if current:
                    tokens.append(current)
                    if token_limit is not None and len(tokens) >= token_limit:
                        return tokens
                current = []
            current.append(
                {
                    "layer": layer,
                    "experts": dedupe_preserve([int(value) for value in record.get("experts", [])]),
                }
            )
            prev_layer = layer
    if current and (token_limit is None or len(tokens) < token_limit):
        tokens.append(current)
    return tokens[:token_limit] if token_limit is not None else tokens


def dedupe_preserve(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_layer_index(manifest: dict[str, Any], manifest_dir: Path) -> dict[int, dict[str, dict[str, int | str]]]:
    index: dict[int, dict[str, dict[str, int | str]]] = {}
    for entry in manifest["entries"]:
        family = str(entry["tensor_family"])
        if family not in FAMILIES:
            continue
        layer = int(entry["layer"])
        layer_map = index.setdefault(layer, {})
        layer_map[family] = {
            "path": str((manifest_dir / entry["repacked_file"]).resolve()),
            "base_offset": int(entry["repacked_offset"]),
            "bytes_per_expert": int(entry["bytes_per_expert"]),
        }
    return index


def build_em_index(
    manifest: dict[str, Any], manifest_dir: Path
) -> tuple[
    dict[int, dict[int, dict[str, dict[str, int | str]]]],
    dict[int, dict[int, dict[int, tuple[int, int]]]],
]:
    index: dict[int, dict[int, dict[str, dict[str, int | str]]]] = {}
    bundle_ranges: dict[int, dict[int, dict[int, tuple[int, int]]]] = {}

    for entry in manifest["entries"]:
        family = str(entry["tensor_family"])
        if family not in FAMILIES:
            continue
        expert_id = int(entry["expert_id"])
        layer = int(entry["layer"])
        family_map = index.setdefault(expert_id, {}).setdefault(layer, {})
        family_map[family] = {
            "path": str((manifest_dir / entry["repacked_file"]).resolve()),
            "offset": int(entry["repacked_offset"]),
            "nbytes": int(entry["exact_byte_length"]),
        }

    for expert_id, layers in index.items():
        sorted_layers = sorted(layers)
        starts: dict[int, dict[int, tuple[int, int]]] = {}
        for start_layer in sorted_layers:
            per_horizon: dict[int, tuple[int, int]] = {}
            available = [layer for layer in sorted_layers if layer >= start_layer]
            for horizon in range(1, 65):
                included = [layer for layer in available if layer <= start_layer + horizon - 1]
                if not included:
                    continue
                first_layer = included[0]
                last_layer = included[-1]
                first_slices = list(index[expert_id][first_layer].values())
                last_slices = list(index[expert_id][last_layer].values())
                start = min(int(item["offset"]) for item in first_slices)
                end = max(int(item["offset"]) + int(item["nbytes"]) for item in last_slices)
                per_horizon[horizon] = (start, end - start)
            starts[start_layer] = per_horizon
        bundle_ranges[expert_id] = starts

    return index, bundle_ranges


def open_fds_for_paths(paths: set[str], stack: ExitStack, no_cache: bool = False) -> dict[str, int]:
    fds: dict[str, int] = {}
    for path in sorted(paths):
        fd = os.open(path, os.O_RDONLY)
        if no_cache and hasattr(fcntl, "F_NOCACHE"):
            try:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            except OSError:
                pass
        stack.callback(os.close, fd)
        fds[path] = fd
    return fds


def read_exact(fd: int, offset: int, nbytes: int) -> int:
    chunk = os.pread(fd, nbytes, offset)
    if len(chunk) != nbytes:
        raise RuntimeError(f"short read: wanted {nbytes}, got {len(chunk)} at offset {offset}")
    return len(chunk)


def maybe_drop_cache() -> None:
    # Best-effort cache drop on macOS. If unavailable or not permitted, continue.
    try:
        os.system("purge >/dev/null 2>&1")
    except Exception:
        pass


def bench_layer_exact(
    tokens: list[list[dict[str, Any]]],
    layer_index: dict[int, dict[str, dict[str, int | str]]],
    no_cache: bool = False,
) -> BenchResult:
    paths = {str(desc["path"]) for family_map in layer_index.values() for desc in family_map.values()}
    with ExitStack() as stack:
        fds = open_fds_for_paths(paths, stack, no_cache=no_cache)
        start = time.perf_counter()
        reads = 0
        bytes_read = 0
        layer_calls = 0
        expert_requests = 0

        for token in tokens:
            for call in token:
                layer = int(call["layer"])
                experts = call["experts"]
                family_map = layer_index[layer]
                layer_calls += 1
                expert_requests += len(experts)
                for expert_id in experts:
                    for family in FAMILIES:
                        desc = family_map.get(family)
                        if desc is None:
                            continue
                        nbytes = int(desc["bytes_per_expert"])
                        offset = int(desc["base_offset"]) + expert_id * nbytes
                        bytes_read += read_exact(fds[str(desc["path"])], offset, nbytes)
                        reads += 1
        wall = time.perf_counter() - start

    return BenchResult(
        mode="layer-exact",
        tokens=len(tokens),
        layer_calls=layer_calls,
        expert_requests=expert_requests,
        reads=reads,
        bytes_read=bytes_read,
        wall_s=wall,
    )


def bench_em_exact(
    tokens: list[list[dict[str, Any]]],
    em_index: dict[int, dict[int, dict[str, dict[str, int | str]]]],
    no_cache: bool = False,
) -> BenchResult:
    paths = {
        str(desc["path"])
        for layer_map in em_index.values()
        for family_map in layer_map.values()
        for desc in family_map.values()
    }
    with ExitStack() as stack:
        fds = open_fds_for_paths(paths, stack, no_cache=no_cache)
        start = time.perf_counter()
        reads = 0
        bytes_read = 0
        layer_calls = 0
        expert_requests = 0

        for token in tokens:
            for call in token:
                layer = int(call["layer"])
                experts = call["experts"]
                layer_calls += 1
                expert_requests += len(experts)
                for expert_id in experts:
                    family_map = em_index[expert_id][layer]
                    for family in FAMILIES:
                        desc = family_map.get(family)
                        if desc is None:
                            continue
                        nbytes = int(desc["nbytes"])
                        offset = int(desc["offset"])
                        bytes_read += read_exact(fds[str(desc["path"])], offset, nbytes)
                        reads += 1
        wall = time.perf_counter() - start

    return BenchResult(
        mode="em-exact",
        tokens=len(tokens),
        layer_calls=layer_calls,
        expert_requests=expert_requests,
        reads=reads,
        bytes_read=bytes_read,
        wall_s=wall,
    )


def bench_em_bundle(
    tokens: list[list[dict[str, Any]]],
    em_index: dict[int, dict[int, dict[str, dict[str, int | str]]]],
    bundle_ranges: dict[int, dict[int, dict[int, tuple[int, int]]]],
    horizon: int,
    no_cache: bool = False,
) -> BenchResult:
    paths = {
        str(next(iter(next(iter(layer_map.values())).values()))["path"])
        for layer_map in em_index.values()
        if layer_map and next(iter(layer_map.values()))
    }
    with ExitStack() as stack:
        fds = open_fds_for_paths(paths, stack, no_cache=no_cache)
        start = time.perf_counter()
        reads = 0
        bytes_read = 0
        layer_calls = 0
        expert_requests = 0
        cache_hits = 0
        cache_misses = 0

        for token in tokens:
            cached: set[tuple[int, int]] = set()
            for call in token:
                layer = int(call["layer"])
                experts = call["experts"]
                layer_calls += 1
                expert_requests += len(experts)
                for expert_id in experts:
                    key = (expert_id, layer)
                    if key in cached:
                        cache_hits += 1
                        continue
                    cache_misses += 1
                    per_layer = bundle_ranges.get(expert_id, {})
                    horizons = per_layer.get(layer, {})
                    if horizon not in horizons:
                        # Fallback to exact current layer family reads if the requested horizon has no range.
                        family_map = em_index[expert_id][layer]
                        for family in FAMILIES:
                            desc = family_map.get(family)
                            if desc is None:
                                continue
                            nbytes = int(desc["nbytes"])
                            offset = int(desc["offset"])
                            bytes_read += read_exact(fds[str(desc["path"])], offset, nbytes)
                            reads += 1
                        cached.add(key)
                        continue

                    offset, nbytes = horizons[horizon]
                    path = str(next(iter(em_index[expert_id][layer].values()))["path"])
                    bytes_read += read_exact(fds[path], offset, nbytes)
                    reads += 1

                    max_layer = layer + horizon - 1
                    for covered_layer in em_index[expert_id]:
                        if covered_layer < layer or covered_layer > max_layer:
                            continue
                        cached.add((expert_id, covered_layer))
        wall = time.perf_counter() - start

    return BenchResult(
        mode=f"em-bundle-h{horizon}",
        tokens=len(tokens),
        layer_calls=layer_calls,
        expert_requests=expert_requests,
        reads=reads,
        bytes_read=bytes_read,
        wall_s=wall,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def bench_em_oracle_exact(
    tokens: list[list[dict[str, Any]]],
    em_index: dict[int, dict[int, dict[str, dict[str, int | str]]]],
    no_cache: bool = False,
) -> BenchResult:
    paths = {
        str(next(iter(next(iter(layer_map.values())).values()))["path"])
        for layer_map in em_index.values()
        if layer_map and next(iter(layer_map.values()))
    }
    with ExitStack() as stack:
        fds = open_fds_for_paths(paths, stack, no_cache=no_cache)
        start = time.perf_counter()
        reads = 0
        bytes_read = 0
        layer_calls = 0
        expert_requests = 0
        cache_hits = 0
        cache_misses = 0

        for token in tokens:
            future_layers_by_expert: dict[int, list[int]] = {}
            for call in token:
                layer = int(call["layer"])
                for expert_id in call["experts"]:
                    future_layers_by_expert.setdefault(expert_id, []).append(layer)

            cached: set[tuple[int, int]] = set()
            prefetched_future: set[tuple[int, int]] = set()
            for call in token:
                layer = int(call["layer"])
                experts = call["experts"]
                layer_calls += 1
                expert_requests += len(experts)
                for expert_id in experts:
                    key = (expert_id, layer)
                    if key in cached:
                        cache_hits += 1
                        continue
                    cache_misses += 1

                    future_layers = future_layers_by_expert.get(expert_id, [])
                    remaining = [l for l in future_layers if l > layer and (expert_id, l) not in prefetched_future]
                    if remaining:
                        path = str(next(iter(em_index[expert_id][layer].values()))["path"])
                        fd = fds[path]
                        for future_layer in remaining:
                            family_map = em_index[expert_id][future_layer]
                            for family in FAMILIES:
                                desc = family_map.get(family)
                                if desc is None:
                                    continue
                                nbytes = int(desc["nbytes"])
                                offset = int(desc["offset"])
                                bytes_read += read_exact(fd, offset, nbytes)
                                reads += 1
                            prefetched_future.add((expert_id, future_layer))
                            cached.add((expert_id, future_layer))

                    family_map = em_index[expert_id][layer]
                    for family in FAMILIES:
                        desc = family_map.get(family)
                        if desc is None:
                            continue
                        nbytes = int(desc["nbytes"])
                        offset = int(desc["offset"])
                        bytes_read += read_exact(fds[str(desc["path"])], offset, nbytes)
                        reads += 1
                    cached.add(key)
        wall = time.perf_counter() - start

    return BenchResult(
        mode="em-oracle-exact",
        tokens=len(tokens),
        layer_calls=layer_calls,
        expert_requests=expert_requests,
        reads=reads,
        bytes_read=bytes_read,
        wall_s=wall,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def bench_em_oracle_span(
    tokens: list[list[dict[str, Any]]],
    em_index: dict[int, dict[int, dict[str, dict[str, int | str]]]],
    bundle_ranges: dict[int, dict[int, dict[int, tuple[int, int]]]],
    no_cache: bool = False,
) -> BenchResult:
    paths = {
        str(next(iter(next(iter(layer_map.values())).values()))["path"])
        for layer_map in em_index.values()
        if layer_map and next(iter(layer_map.values()))
    }
    with ExitStack() as stack:
        fds = open_fds_for_paths(paths, stack, no_cache=no_cache)
        start = time.perf_counter()
        reads = 0
        bytes_read = 0
        layer_calls = 0
        expert_requests = 0
        cache_hits = 0
        cache_misses = 0

        for token in tokens:
            future_layers_by_expert: dict[int, list[int]] = {}
            for call in token:
                layer = int(call["layer"])
                for expert_id in call["experts"]:
                    future_layers_by_expert.setdefault(expert_id, []).append(layer)

            cached: set[tuple[int, int]] = set()
            for call in token:
                layer = int(call["layer"])
                experts = call["experts"]
                layer_calls += 1
                expert_requests += len(experts)
                for expert_id in experts:
                    key = (expert_id, layer)
                    if key in cached:
                        cache_hits += 1
                        continue
                    cache_misses += 1

                    future_layers = future_layers_by_expert.get(expert_id, [])
                    remaining = [l for l in future_layers if l >= layer and (expert_id, l) not in cached]
                    if not remaining:
                        continue
                    last_layer = max(remaining)
                    horizon = last_layer - layer + 1
                    if horizon in bundle_ranges.get(expert_id, {}).get(layer, {}):
                        offset, nbytes = bundle_ranges[expert_id][layer][horizon]
                        path = str(next(iter(em_index[expert_id][layer].values()))["path"])
                        bytes_read += read_exact(fds[path], offset, nbytes)
                        reads += 1
                        for covered_layer in em_index[expert_id]:
                            if layer <= covered_layer <= last_layer:
                                cached.add((expert_id, covered_layer))
                    else:
                        family_map = em_index[expert_id][layer]
                        for family in FAMILIES:
                            desc = family_map.get(family)
                            if desc is None:
                                continue
                            nbytes = int(desc["nbytes"])
                            offset = int(desc["offset"])
                            bytes_read += read_exact(fds[str(desc["path"])], offset, nbytes)
                            reads += 1
                        cached.add(key)
        wall = time.perf_counter() - start

    return BenchResult(
        mode="em-oracle-span",
        tokens=len(tokens),
        layer_calls=layer_calls,
        expert_requests=expert_requests,
        reads=reads,
        bytes_read=bytes_read,
        wall_s=wall,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def bench_em_oracle_next_layer(
    tokens: list[list[dict[str, Any]]],
    layer_index: dict[int, dict[str, dict[str, int | str]]],
    em_index: dict[int, dict[int, dict[str, dict[str, int | str]]]],
    no_cache: bool = False,
) -> BenchResult:
    layer_paths = {str(desc["path"]) for family_map in layer_index.values() for desc in family_map.values()}
    em_paths = {
        str(next(iter(next(iter(layer_map.values())).values()))["path"])
        for layer_map in em_index.values()
        if layer_map and next(iter(layer_map.values()))
    }
    with ExitStack() as stack:
        layer_fds = open_fds_for_paths(layer_paths, stack, no_cache=no_cache)
        em_fds = open_fds_for_paths(em_paths, stack, no_cache=no_cache)
        start = time.perf_counter()
        reads = 0
        bytes_read = 0
        layer_calls = 0
        expert_requests = 0
        cache_hits = 0
        cache_misses = 0

        for token in tokens:
            future_layers_by_expert: dict[int, list[int]] = {}
            for call in token:
                layer = int(call["layer"])
                for expert_id in call["experts"]:
                    future_layers_by_expert.setdefault(expert_id, []).append(layer)

            cached: set[tuple[int, int]] = set()
            prefetched_future: set[tuple[int, int]] = set()
            for call in token:
                layer = int(call["layer"])
                experts = call["experts"]
                layer_calls += 1
                expert_requests += len(experts)
                for expert_id in experts:
                    key = (expert_id, layer)
                    if key in cached:
                        cache_hits += 1
                        continue
                    cache_misses += 1

                    future_layers = future_layers_by_expert.get(expert_id, [])
                    next_layer = layer + 1
                    remaining = [l for l in future_layers if l == next_layer and (expert_id, l) not in prefetched_future]
                    if remaining:
                        # Reused in the immediate next layer: use EM exact slices for that short prefetch window.
                        path = str(next(iter(em_index[expert_id][layer].values()))["path"])
                        fd = em_fds[path]
                        for future_layer in remaining:
                            family_map = em_index[expert_id][future_layer]
                            for family in FAMILIES:
                                desc = family_map.get(family)
                                if desc is None:
                                    continue
                                nbytes = int(desc["nbytes"])
                                offset = int(desc["offset"])
                                bytes_read += read_exact(fd, offset, nbytes)
                                reads += 1
                            prefetched_future.add((expert_id, future_layer))
                            cached.add((expert_id, future_layer))

                        family_map = em_index[expert_id][layer]
                        for family in FAMILIES:
                            desc = family_map.get(family)
                            if desc is None:
                                continue
                            nbytes = int(desc["nbytes"])
                            offset = int(desc["offset"])
                            bytes_read += read_exact(fd, offset, nbytes)
                            reads += 1
                    else:
                        # No immediate next-layer reuse: stay on the existing layer-major path.
                        family_map = layer_index[layer]
                        for family in FAMILIES:
                            desc = family_map.get(family)
                            if desc is None:
                                continue
                            nbytes = int(desc["bytes_per_expert"])
                            offset = int(desc["base_offset"]) + expert_id * nbytes
                            bytes_read += read_exact(layer_fds[str(desc["path"])], offset, nbytes)
                            reads += 1
                    cached.add(key)
        wall = time.perf_counter() - start

    return BenchResult(
        mode="em-oracle-next-layer",
        tokens=len(tokens),
        layer_calls=layer_calls,
        expert_requests=expert_requests,
        reads=reads,
        bytes_read=bytes_read,
        wall_s=wall,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def bench_em_oracle_next_layer_bundle(
    tokens: list[list[dict[str, Any]]],
    layer_index: dict[int, dict[str, dict[str, int | str]]],
    em_index: dict[int, dict[int, dict[str, dict[str, int | str]]]],
    bundle_ranges: dict[int, dict[int, dict[int, tuple[int, int]]]],
    no_cache: bool = False,
) -> BenchResult:
    layer_paths = {str(desc["path"]) for family_map in layer_index.values() for desc in family_map.values()}
    em_paths = {
        str(next(iter(next(iter(layer_map.values())).values()))["path"])
        for layer_map in em_index.values()
        if layer_map and next(iter(layer_map.values()))
    }
    with ExitStack() as stack:
        layer_fds = open_fds_for_paths(layer_paths, stack, no_cache=no_cache)
        em_fds = open_fds_for_paths(em_paths, stack, no_cache=no_cache)
        start = time.perf_counter()
        reads = 0
        bytes_read = 0
        layer_calls = 0
        expert_requests = 0
        cache_hits = 0
        cache_misses = 0

        for token in tokens:
            future_layers_by_expert: dict[int, set[int]] = {}
            for call in token:
                layer = int(call["layer"])
                for expert_id in call["experts"]:
                    future_layers_by_expert.setdefault(expert_id, set()).add(layer)

            cached: set[tuple[int, int]] = set()
            for call in token:
                layer = int(call["layer"])
                experts = call["experts"]
                layer_calls += 1
                expert_requests += len(experts)
                for expert_id in experts:
                    key = (expert_id, layer)
                    if key in cached:
                        cache_hits += 1
                        continue
                    cache_misses += 1

                    next_layer = layer + 1
                    if next_layer in future_layers_by_expert.get(expert_id, set()):
                        horizons = bundle_ranges.get(expert_id, {}).get(layer, {})
                        if 2 in horizons:
                            offset, nbytes = horizons[2]
                            path = str(next(iter(em_index[expert_id][layer].values()))["path"])
                            bytes_read += read_exact(em_fds[path], offset, nbytes)
                            reads += 1
                            cached.add((expert_id, layer))
                            cached.add((expert_id, next_layer))
                            continue

                    family_map = layer_index[layer]
                    for family in FAMILIES:
                        desc = family_map.get(family)
                        if desc is None:
                            continue
                        nbytes = int(desc["bytes_per_expert"])
                        offset = int(desc["base_offset"]) + expert_id * nbytes
                        bytes_read += read_exact(layer_fds[str(desc["path"])], offset, nbytes)
                        reads += 1
                    cached.add(key)
        wall = time.perf_counter() - start

    return BenchResult(
        mode="em-oracle-next-layer-bundle",
        tokens=len(tokens),
        layer_calls=layer_calls,
        expert_requests=expert_requests,
        reads=reads,
        bytes_read=bytes_read,
        wall_s=wall,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def bench_em_oracle_gap_bundle(
    tokens: list[list[dict[str, Any]]],
    layer_index: dict[int, dict[str, dict[str, int | str]]],
    em_index: dict[int, dict[int, dict[str, dict[str, int | str]]]],
    bundle_ranges: dict[int, dict[int, dict[int, tuple[int, int]]]],
    max_holes: int,
    no_cache: bool = False,
) -> BenchResult:
    layer_paths = {str(desc["path"]) for family_map in layer_index.values() for desc in family_map.values()}
    em_paths = {
        str(next(iter(next(iter(layer_map.values())).values()))["path"])
        for layer_map in em_index.values()
        if layer_map and next(iter(layer_map.values()))
    }
    with ExitStack() as stack:
        layer_fds = open_fds_for_paths(layer_paths, stack, no_cache=no_cache)
        em_fds = open_fds_for_paths(em_paths, stack, no_cache=no_cache)
        start = time.perf_counter()
        reads = 0
        bytes_read = 0
        layer_calls = 0
        expert_requests = 0
        cache_hits = 0
        cache_misses = 0

        max_distance = max_holes + 1
        mode_name = f"em-oracle-hole{max_holes}-bundle"

        for token in tokens:
            future_layers_by_expert: dict[int, set[int]] = {}
            for call in token:
                layer = int(call["layer"])
                for expert_id in call["experts"]:
                    future_layers_by_expert.setdefault(expert_id, set()).add(layer)

            cached: set[tuple[int, int]] = set()
            for call in token:
                layer = int(call["layer"])
                experts = call["experts"]
                layer_calls += 1
                expert_requests += len(experts)
                for expert_id in experts:
                    key = (expert_id, layer)
                    if key in cached:
                        cache_hits += 1
                        continue
                    cache_misses += 1

                    future_layers = future_layers_by_expert.get(expert_id, set())
                    chosen_layer = None
                    for distance in range(1, max_distance + 1):
                        candidate = layer + distance
                        if candidate in future_layers:
                            chosen_layer = candidate
                            break

                    if chosen_layer is not None:
                        horizon = chosen_layer - layer + 1
                        horizons = bundle_ranges.get(expert_id, {}).get(layer, {})
                        if horizon in horizons:
                            offset, nbytes = horizons[horizon]
                            path = str(next(iter(em_index[expert_id][layer].values()))["path"])
                            bytes_read += read_exact(em_fds[path], offset, nbytes)
                            reads += 1
                            for covered_layer in range(layer, chosen_layer + 1):
                                cached.add((expert_id, covered_layer))
                            continue

                    family_map = layer_index[layer]
                    for family in FAMILIES:
                        desc = family_map.get(family)
                        if desc is None:
                            continue
                        nbytes = int(desc["bytes_per_expert"])
                        offset = int(desc["base_offset"]) + expert_id * nbytes
                        bytes_read += read_exact(layer_fds[str(desc["path"])], offset, nbytes)
                        reads += 1
                    cached.add(key)
        wall = time.perf_counter() - start

    return BenchResult(
        mode=mode_name,
        tokens=len(tokens),
        layer_calls=layer_calls,
        expert_requests=expert_requests,
        reads=reads,
        bytes_read=bytes_read,
        wall_s=wall,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def render_text(results: list[BenchResult]) -> str:
    lines = []
    header = (
        f"{'mode':<16} {'tok':>5} {'ms/tok':>8} {'tok/s':>7} {'GiB':>8} "
        f"{'GiB/s':>8} {'reads':>10} {'MiB/read':>9} {'hit':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        hit = (
            f"{(result.cache_hits / max(1, result.cache_hits + result.cache_misses)) * 100.0:5.1f}%"
            if result.cache_hits or result.cache_misses
            else "   n/a"
        )
        lines.append(
            f"{result.mode:<16} {result.tokens:>5} {result.ms_per_token:>8.2f} {result.tok_per_s:>7.2f} "
            f"{result.gib_read:>8.2f} {result.gib_per_s:>8.2f} {result.reads:>10} "
            f"{result.mib_per_read:>9.2f} {hit:>7}"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    modes = args.mode or [
        "layer-exact",
        "em-exact",
        "em-bundle",
        "em-oracle-exact",
        "em-oracle-span",
        "em-oracle-next-layer",
        "em-oracle-next-layer-bundle",
        "em-oracle-hole1-bundle",
        "em-oracle-hole2-bundle",
    ]
    horizons = args.horizon or [2, 4]

    layer_manifest_path, layer_manifest = load_manifest(args.layer_sidecar)
    em_manifest_path, em_manifest = load_manifest(args.em_sidecar)
    tokens = split_trace_tokens(args.trace, args.token_limit)

    layer_index = build_layer_index(layer_manifest, layer_manifest_path.parent)
    em_index, bundle_ranges = build_em_index(em_manifest, em_manifest_path.parent)

    results: list[BenchResult] = []

    for mode in modes:
        if args.drop_cache_between_runs:
            maybe_drop_cache()
        if mode == "layer-exact":
            results.append(bench_layer_exact(tokens, layer_index, no_cache=args.no_cache))
        elif mode == "em-exact":
            results.append(bench_em_exact(tokens, em_index, no_cache=args.no_cache))
        elif mode == "em-bundle":
            for horizon in horizons:
                if args.drop_cache_between_runs:
                    maybe_drop_cache()
                results.append(bench_em_bundle(tokens, em_index, bundle_ranges, horizon, no_cache=args.no_cache))
        elif mode == "em-oracle-exact":
            results.append(bench_em_oracle_exact(tokens, em_index, no_cache=args.no_cache))
        elif mode == "em-oracle-span":
            results.append(bench_em_oracle_span(tokens, em_index, bundle_ranges, no_cache=args.no_cache))
        elif mode == "em-oracle-next-layer":
            results.append(bench_em_oracle_next_layer(tokens, layer_index, em_index, no_cache=args.no_cache))
        elif mode == "em-oracle-next-layer-bundle":
            results.append(bench_em_oracle_next_layer_bundle(tokens, layer_index, em_index, bundle_ranges, no_cache=args.no_cache))
        elif mode == "em-oracle-hole1-bundle":
            results.append(bench_em_oracle_gap_bundle(tokens, layer_index, em_index, bundle_ranges, max_holes=1, no_cache=args.no_cache))
        elif mode == "em-oracle-hole2-bundle":
            results.append(bench_em_oracle_gap_bundle(tokens, layer_index, em_index, bundle_ranges, max_holes=2, no_cache=args.no_cache))
        else:
            raise SystemExit(f"unknown mode '{mode}'")

    if args.json:
        print(
            json.dumps(
                {
                    "trace": str(args.trace.expanduser().resolve()),
                    "tokens": len(tokens),
                    "results": [result.__dict__ for result in results],
                },
                indent=2,
            )
        )
    else:
        print(render_text(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
