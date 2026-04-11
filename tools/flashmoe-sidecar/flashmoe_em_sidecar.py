#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from flashmoe_sidecar import (  # type: ignore
    FAMILY_ORDER,
    filter_manifest_entries,
    load_manifest,
    parse_family_spec,
    parse_layer_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack a layer-major Flash-MoE sidecar into an expert-major sidecar."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="repack a layer-major sidecar into one file per expert",
    )
    extract.add_argument(
        "--sidecar",
        required=True,
        type=Path,
        help="input Flash-MoE sidecar directory or manifest.json",
    )
    extract.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="output expert-major sidecar directory",
    )
    extract.add_argument(
        "--layers",
        type=str,
        help="optional layer filter, e.g. '3-10'",
    )
    extract.add_argument(
        "--families",
        type=str,
        help="optional family filter, e.g. 'ffn_gate_exps,ffn_up_exps' or aliases 'routed,shared,all'",
    )
    extract.add_argument(
        "--experts",
        type=str,
        help="optional expert filter, e.g. '0,1,4-7'; defaults to all experts",
    )
    extract.add_argument(
        "--force",
        action="store_true",
        help="overwrite manifest and expert files in the output directory",
    )
    extract.set_defaults(func=cmd_extract)

    return parser.parse_args()


def parse_expert_spec(spec: str | None, expert_count: int) -> set[int] | None:
    if spec is None or spec.strip() == "" or spec.strip() == "all":
        return None

    experts: set[int] = set()
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise SystemExit(f"invalid expert range '{item}'")
            experts.update(range(start, end + 1))
        else:
            experts.add(int(item))

    for expert_id in experts:
        if expert_id < 0 or expert_id >= expert_count:
            raise SystemExit(
                f"expert id {expert_id} is out of range for expert_count={expert_count}"
            )
    return experts


def prepare_out_dir(out_dir: Path, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.iterdir())
    if not existing:
        return
    if not force:
        raise SystemExit(
            f"output directory '{out_dir}' is not empty; use --force to overwrite files"
        )
    for path in existing:
        if path.is_dir():
            raise SystemExit(
                f"refusing to overwrite directory '{path}'; clear '{out_dir}' manually first"
            )
        path.unlink()


def expand_entries_by_expert(
    manifest_entries: list[dict[str, Any]],
    expert_count: int,
    expert_filter: set[int] | None,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    selected_experts = (
        sorted(expert_filter) if expert_filter is not None else list(range(expert_count))
    )

    for entry in manifest_entries:
        bytes_per_expert = entry.get("bytes_per_expert")
        if not isinstance(bytes_per_expert, int) or bytes_per_expert <= 0:
            raise SystemExit(
                f"manifest entry '{entry.get('tensor_name', '<unknown>')}' has invalid bytes_per_expert"
            )

        exact_byte_length = int(entry["exact_byte_length"])
        shape = entry.get("shape") or []
        inferred_expert_count = shape[-1] if shape else None
        if inferred_expert_count is not None and inferred_expert_count != expert_count:
            raise SystemExit(
                f"manifest entry '{entry['tensor_name']}' expert dimension {inferred_expert_count} "
                f"does not match model expert_count {expert_count}"
            )
        if exact_byte_length % bytes_per_expert != 0:
            raise SystemExit(
                f"manifest entry '{entry['tensor_name']}' exact_byte_length is not divisible by bytes_per_expert"
            )

        for expert_id in selected_experts:
            expert_entry = dict(entry)
            expert_entry["expert_id"] = expert_id
            expert_entry["original_exact_byte_length"] = exact_byte_length
            expert_entry["original_shape"] = shape
            expert_entry["expert_shape"] = shape[:-1] if len(shape) >= 1 else shape
            expert_entry["exact_byte_length"] = bytes_per_expert
            expert_entry["source_offset"] = int(entry["source_offset"]) + expert_id * bytes_per_expert
            expert_entry["layer_major_source_file"] = entry["repacked_file"]
            expert_entry["layer_major_source_offset"] = int(entry["repacked_offset"]) + expert_id * bytes_per_expert
            expert_entry["repacked_file"] = None
            expert_entry["repacked_offset"] = None
            expanded.append(expert_entry)

    return sorted(
        expanded,
        key=lambda entry: (
            int(entry["expert_id"]),
            int(entry["layer"]),
            FAMILY_ORDER.get(str(entry["tensor_family"]), 99),
            str(entry["tensor_name"]),
        ),
    )


def copy_em_bytes(
    manifest_dir: Path,
    entries: list[dict[str, Any]],
    out_dir: Path,
    expert_count: int,
) -> None:
    width = max(3, len(str(max(expert_count - 1, 0))))
    with ExitStack() as stack:
        source_handles: dict[Path, Any] = {}
        expert_handles: dict[Path, Any] = {}

        for entry in entries:
            source_path = (manifest_dir / entry["layer_major_source_file"]).resolve()
            source_handle = source_handles.get(source_path)
            if source_handle is None:
                source_handle = stack.enter_context(source_path.open("rb"))
                source_handles[source_path] = source_handle

            expert_path = out_dir / f"expert_{int(entry['expert_id']):0{width}d}.bin"
            expert_handle = expert_handles.get(expert_path)
            if expert_handle is None:
                expert_handle = stack.enter_context(expert_path.open("wb"))
                expert_handles[expert_path] = expert_handle

            entry["repacked_file"] = expert_path.name
            entry["repacked_offset"] = expert_handle.tell()

            source_handle.seek(int(entry["layer_major_source_offset"]))
            remaining = int(entry["exact_byte_length"])
            while remaining > 0:
                chunk = source_handle.read(min(8 << 20, remaining))
                if not chunk:
                    raise SystemExit(
                        f"unexpected EOF while reading '{entry['tensor_name']}' "
                        f"expert {entry['expert_id']}"
                    )
                expert_handle.write(chunk)
                remaining -= len(chunk)


def summarize_by_expert(entries: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[int, dict[str, Any]] = {}
    for entry in entries:
        expert_id = int(entry["expert_id"])
        data = summary.setdefault(
            expert_id,
            {"slices": 0, "bytes": 0, "layers": set(), "families": set()},
        )
        data["slices"] += 1
        data["bytes"] += int(entry["exact_byte_length"])
        data["layers"].add(int(entry["layer"]))
        data["families"].add(str(entry["tensor_family"]))

    by_expert = {}
    for expert_id, values in sorted(summary.items()):
        by_expert[str(expert_id)] = {
            "slices": values["slices"],
            "bytes": values["bytes"],
            "layers": sorted(values["layers"]),
            "families": sorted(values["families"], key=lambda family: FAMILY_ORDER.get(family, 99)),
        }
    return by_expert


def cmd_extract(args: argparse.Namespace) -> int:
    manifest_path, manifest = load_manifest(args.sidecar)
    if manifest.get("layout") != "layer_major_whole_tensor":
        raise SystemExit(
            f"unsupported input layout '{manifest.get('layout')}'; expected layer_major_whole_tensor"
        )

    model_meta = manifest.get("model") or {}
    expert_count = model_meta.get("expert_count")
    if not isinstance(expert_count, int) or expert_count <= 0:
        raise SystemExit("manifest is missing model.expert_count")

    layer_filter = parse_layer_spec(args.layers)
    family_filter = parse_family_spec(args.families)
    expert_filter = parse_expert_spec(args.experts, expert_count)
    input_entries = filter_manifest_entries(
        manifest["entries"],
        layer_filter=layer_filter,
        family_filter=family_filter,
    )
    if not input_entries:
        raise SystemExit("no manifest entries matched the requested filters")

    out_dir = args.out_dir.expanduser().resolve()
    prepare_out_dir(out_dir, args.force)

    expanded_entries = expand_entries_by_expert(
        input_entries,
        expert_count=expert_count,
        expert_filter=expert_filter,
    )
    copy_em_bytes(
        manifest_dir=manifest_path.parent,
        entries=expanded_entries,
        out_dir=out_dir,
        expert_count=expert_count,
    )

    output_manifest = {
        "schema_version": 1,
        "sidecar_kind": "flashmoe_gguf",
        "layout": "expert_major_per_expert_slice",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_layout": manifest.get("layout"),
        "include_shared": bool(manifest.get("include_shared")),
        "filters": {
            "layers": sorted(layer_filter) if layer_filter is not None else None,
            "families": sorted(family_filter, key=lambda family: FAMILY_ORDER.get(family, 99))
            if family_filter is not None
            else None,
            "experts": sorted(expert_filter) if expert_filter is not None else None,
        },
        "source": {
            "input_sidecar": str(manifest_path.parent),
            "input_manifest": str(manifest_path),
            "model_files": manifest.get("source", {}).get("model_files"),
        },
        "model": model_meta,
        "entries": expanded_entries,
        "by_expert": summarize_by_expert(expanded_entries),
    }

    manifest_out = out_dir / "manifest.json"
    with manifest_out.open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")

    total_bytes = sum(int(entry["exact_byte_length"]) for entry in expanded_entries)
    expert_ids = sorted({int(entry["expert_id"]) for entry in expanded_entries})
    layer_ids = sorted({int(entry["layer"]) for entry in expanded_entries})

    print(
        f"wrote {len(expanded_entries)} expert slices for {len(expert_ids)} experts "
        f"across {len(layer_ids)} layers to {out_dir}"
    )
    print(f"total bytes copied: {total_bytes}")
    print(f"manifest: {manifest_out}")
    return 0


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
