#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import ctypes
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sys


LLAMA_ROOT = Path(__file__).resolve().parents[2]
if str(LLAMA_ROOT / "gguf-py") not in sys.path:
    sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))

from gguf import GGMLQuantizationType  # type: ignore


@dataclass(frozen=True)
class ExpertGeometry:
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    group_size: int
    bits: int

    @property
    def values_per_uint32(self) -> int:
        return 32 // self.bits

    @property
    def packed_hidden_size(self) -> int:
        return self.hidden_size // self.values_per_uint32

    @property
    def packed_moe_size(self) -> int:
        return self.moe_intermediate_size // self.values_per_uint32

    @property
    def hidden_groups(self) -> int:
        return self.hidden_size // self.group_size

    @property
    def moe_groups(self) -> int:
        return self.moe_intermediate_size // self.group_size

    @property
    def gate_weight_shape(self) -> tuple[int, int]:
        return (self.moe_intermediate_size, self.packed_hidden_size)

    @property
    def gate_scale_shape(self) -> tuple[int, int]:
        return (self.moe_intermediate_size, self.hidden_groups)

    @property
    def up_weight_shape(self) -> tuple[int, int]:
        return (self.moe_intermediate_size, self.packed_hidden_size)

    @property
    def up_scale_shape(self) -> tuple[int, int]:
        return (self.moe_intermediate_size, self.hidden_groups)

    @property
    def down_weight_shape(self) -> tuple[int, int]:
        return (self.hidden_size, self.packed_moe_size)

    @property
    def down_scale_shape(self) -> tuple[int, int]:
        return (self.hidden_size, self.moe_groups)

    @property
    def gate_weight_bytes(self) -> int:
        return int(np.prod(self.gate_weight_shape)) * np.dtype(np.uint32).itemsize

    @property
    def gate_scale_bytes(self) -> int:
        return int(np.prod(self.gate_scale_shape)) * np.dtype(np.uint16).itemsize

    @property
    def gate_bias_bytes(self) -> int:
        return self.gate_scale_bytes

    @property
    def up_weight_bytes(self) -> int:
        return int(np.prod(self.up_weight_shape)) * np.dtype(np.uint32).itemsize

    @property
    def up_scale_bytes(self) -> int:
        return int(np.prod(self.up_scale_shape)) * np.dtype(np.uint16).itemsize

    @property
    def up_bias_bytes(self) -> int:
        return self.up_scale_bytes

    @property
    def down_weight_bytes(self) -> int:
        return int(np.prod(self.down_weight_shape)) * np.dtype(np.uint32).itemsize

    @property
    def down_scale_bytes(self) -> int:
        return int(np.prod(self.down_scale_shape)) * np.dtype(np.uint16).itemsize

    @property
    def down_bias_bytes(self) -> int:
        return self.down_scale_bytes

    @property
    def gate_weight_offset(self) -> int:
        return 0

    @property
    def gate_scale_offset(self) -> int:
        return self.gate_weight_offset + self.gate_weight_bytes

    @property
    def gate_bias_offset(self) -> int:
        return self.gate_scale_offset + self.gate_scale_bytes

    @property
    def up_weight_offset(self) -> int:
        return self.gate_bias_offset + self.gate_bias_bytes

    @property
    def up_scale_offset(self) -> int:
        return self.up_weight_offset + self.up_weight_bytes

    @property
    def up_bias_offset(self) -> int:
        return self.up_scale_offset + self.up_scale_bytes

    @property
    def down_weight_offset(self) -> int:
        return self.up_bias_offset + self.up_bias_bytes

    @property
    def down_scale_offset(self) -> int:
        return self.down_weight_offset + self.down_weight_bytes

    @property
    def down_bias_offset(self) -> int:
        return self.down_scale_offset + self.down_scale_bytes

    @property
    def expert_size(self) -> int:
        return self.down_bias_offset + self.down_bias_bytes


GEOMETRY_397B_2BIT = ExpertGeometry(
    hidden_size=4096,
    moe_intermediate_size=1024,
    num_experts=512,
    group_size=64,
    bits=2,
)

GGML_BASE_DYLIB = LLAMA_ROOT / "build" / "bin" / "libggml-base.dylib"

FAMILY_TO_SOURCE = {
    "ffn_gate_exps": ("gate", GEOMETRY_397B_2BIT.gate_weight_shape, GEOMETRY_397B_2BIT.gate_scale_shape),
    "ffn_up_exps": ("up", GEOMETRY_397B_2BIT.up_weight_shape, GEOMETRY_397B_2BIT.up_scale_shape),
    "ffn_down_exps": ("down", GEOMETRY_397B_2BIT.down_weight_shape, GEOMETRY_397B_2BIT.down_scale_shape),
}

_THREAD_STATE = threading.local()


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not np.isfinite(seconds):
        return "--:--:--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the older affine 2-bit expert pack into a Flash-MoE GGUF sidecar."
    )
    parser.add_argument("--source-pack", required=True, type=Path, help="Directory with layer_XX.bin affine 2-bit experts")
    parser.add_argument(
        "--template-sidecar",
        required=True,
        type=Path,
        help="Existing Flash-MoE sidecar directory or manifest to use as the GGUF tensor template",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output Flash-MoE sidecar directory")
    parser.add_argument(
        "--layers",
        default=None,
        help="Optional layer filter, e.g. '0', '0-3', '0,3,5-7'",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of layers to convert in parallel",
    )
    parser.add_argument(
        "--expert-jobs",
        type=int,
        default=1,
        help="Number of worker threads to use within each layer across expert chunks",
    )
    parser.add_argument(
        "--expert-chunk-size",
        type=int,
        default=0,
        help="Experts per chunk for within-layer parallelism (0 = auto)",
    )
    parser.add_argument(
        "--target-quant-type",
        default=None,
        help="Optional GGML quant type override for all routed tensors, e.g. IQ2_XXS",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write a manifest-only sidecar that reads directly from the affine 2-bit layer files and requantizes at runtime",
    )
    return parser.parse_args()


def parse_layers(spec: str | None) -> set[int] | None:
    if spec is None:
        return None
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid layer range '{part}'")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def bf16_to_f32(bf16_u16: np.ndarray) -> np.ndarray:
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)


def unpack_2bit(packed: np.ndarray) -> np.ndarray:
    shape = packed.shape
    flat = packed.ravel()
    out = np.empty(flat.size * 16, dtype=np.uint8)
    for i in range(16):
        out[i::16] = ((flat >> (i * 2)) & 0x3).astype(np.uint8)
    return out.reshape(shape[:-1] + (shape[-1] * 16,))


def dequantize_affine_projection(
    expert_blob: memoryview,
    *,
    weight_offset: int,
    weight_shape: tuple[int, int],
    scale_offset: int,
    scale_shape: tuple[int, int],
    bias_offset: int,
) -> np.ndarray:
    weight_bytes = int(np.prod(weight_shape)) * np.dtype(np.uint32).itemsize
    scale_bytes = int(np.prod(scale_shape)) * np.dtype(np.uint16).itemsize

    packed = np.frombuffer(expert_blob[weight_offset : weight_offset + weight_bytes], dtype=np.uint32).reshape(weight_shape)
    scales = np.frombuffer(expert_blob[scale_offset : scale_offset + scale_bytes], dtype=np.uint16).reshape(scale_shape)
    biases = np.frombuffer(expert_blob[bias_offset : bias_offset + scale_bytes], dtype=np.uint16).reshape(scale_shape)

    values = unpack_2bit(packed).reshape(weight_shape[0], scale_shape[1], GEOMETRY_397B_2BIT.group_size).astype(np.float32)
    s = bf16_to_f32(scales)[:, :, np.newaxis]
    b = bf16_to_f32(biases)[:, :, np.newaxis]
    return (values * s + b).reshape(weight_shape[0], scale_shape[1] * GEOMETRY_397B_2BIT.group_size)


def load_template_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class NativeQuantizer:
    def __init__(self, dylib_path: Path):
        self.lib = ctypes.CDLL(str(dylib_path))
        self.lib.ggml_row_size.argtypes = [
            ctypes.c_int,
            ctypes.c_int64,
        ]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    def row_size(self, qtype: GGMLQuantizationType, n_per_row: int) -> int:
        return int(self.lib.ggml_row_size(int(qtype.value), int(n_per_row)))

    def quantize(self, matrix: np.ndarray, qtype: GGMLQuantizationType, expected_nbytes: int | None = None) -> bytes:
        arr = np.ascontiguousarray(matrix, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"expected 2D matrix, got shape {arr.shape!r}")
        if expected_nbytes is None:
            expected_nbytes = self.row_size(qtype, arr.shape[0]) * arr.shape[1]
        out = (ctypes.c_uint8 * expected_nbytes)()
        written = self.lib.ggml_quantize_chunk(
            int(qtype.value),
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out),
            0,
            arr.shape[0],
            arr.shape[1],
            None,
        )
        if written != expected_nbytes:
            raise ValueError(f"native quantizer wrote {written} bytes, expected {expected_nbytes}")
        return bytes(out)


def get_thread_quantizer(ggml_base_dylib: str) -> NativeQuantizer:
    cached = getattr(_THREAD_STATE, "native_quantizer", None)
    cached_path = getattr(_THREAD_STATE, "native_quantizer_path", None)
    if cached is None or cached_path != ggml_base_dylib:
        cached = NativeQuantizer(Path(ggml_base_dylib))
        _THREAD_STATE.native_quantizer = cached
        _THREAD_STATE.native_quantizer_path = ggml_base_dylib
    return cached


def choose_projection_buffer(expert_blob: memoryview, family: str) -> np.ndarray:
    if family == "ffn_gate_exps":
        return dequantize_affine_projection(
            expert_blob,
            weight_offset=GEOMETRY_397B_2BIT.gate_weight_offset,
            weight_shape=GEOMETRY_397B_2BIT.gate_weight_shape,
            scale_offset=GEOMETRY_397B_2BIT.gate_scale_offset,
            scale_shape=GEOMETRY_397B_2BIT.gate_scale_shape,
            bias_offset=GEOMETRY_397B_2BIT.gate_bias_offset,
        )
    if family == "ffn_up_exps":
        return dequantize_affine_projection(
            expert_blob,
            weight_offset=GEOMETRY_397B_2BIT.up_weight_offset,
            weight_shape=GEOMETRY_397B_2BIT.up_weight_shape,
            scale_offset=GEOMETRY_397B_2BIT.up_scale_offset,
            scale_shape=GEOMETRY_397B_2BIT.up_scale_shape,
            bias_offset=GEOMETRY_397B_2BIT.up_bias_offset,
        )
    if family == "ffn_down_exps":
        return dequantize_affine_projection(
            expert_blob,
            weight_offset=GEOMETRY_397B_2BIT.down_weight_offset,
            weight_shape=GEOMETRY_397B_2BIT.down_weight_shape,
            scale_offset=GEOMETRY_397B_2BIT.down_scale_offset,
            scale_shape=GEOMETRY_397B_2BIT.down_scale_shape,
            bias_offset=GEOMETRY_397B_2BIT.down_bias_offset,
        )
    raise ValueError(f"unsupported tensor family '{family}'")


def convert_expert_tensor(expert_blob: memoryview, entry: dict[str, Any], native_quantizer: NativeQuantizer) -> bytes:
    family = str(entry["tensor_family"])
    target_shape = tuple(int(v) for v in entry["shape"][:2])
    target_qtype = GGMLQuantizationType[str(entry["quant_type"])]

    source = choose_projection_buffer(expert_blob, family)
    if source.shape == target_shape:
        aligned = source
    elif source.T.shape == target_shape:
        aligned = source.T
    else:
        raise ValueError(
            f"source projection shape {source.shape!r} does not match target {target_shape!r} for {entry['tensor_name']}"
        )

    exact = int(entry["bytes_per_expert"])
    return native_quantizer.quantize(aligned, target_qtype, exact)


def parse_quant_type(name: str | None) -> GGMLQuantizationType | None:
    if not name:
        return None
    normalized = name.strip().upper()
    try:
        return GGMLQuantizationType[normalized]
    except KeyError as exc:
        raise SystemExit(f"unknown --target-quant-type '{name}'") from exc


def resolve_target_qtype(entry: dict[str, Any], forced_qtype: GGMLQuantizationType | None) -> GGMLQuantizationType:
    return forced_qtype if forced_qtype is not None else GGMLQuantizationType[str(entry["quant_type"])]


def bytes_per_expert_for(entry: dict[str, Any], qtype: GGMLQuantizationType, native_quantizer: NativeQuantizer) -> int:
    target_shape = tuple(int(v) for v in entry["shape"][:2])
    return native_quantizer.row_size(qtype, target_shape[0]) * target_shape[1]


def ensure_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if any(path.iterdir()):
            if not force:
                raise SystemExit(f"output directory '{path}' already exists and is not empty; pass --force to overwrite")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def convert_expert_chunk(
    layer_blob: bytes,
    entry: dict[str, Any],
    ggml_base_dylib: str,
    expert_start: int,
    expert_stop: int,
    quant_type_name: str | None,
) -> tuple[int, bytes, int]:
    native_quantizer = get_thread_quantizer(ggml_base_dylib)
    forced_qtype = parse_quant_type(quant_type_name)
    target_qtype = resolve_target_qtype(entry, forced_qtype)
    bytes_per_expert = bytes_per_expert_for(entry, target_qtype, native_quantizer)
    converted = bytearray(bytes_per_expert * (expert_stop - expert_start))

    for local_index, expert_id in enumerate(range(expert_start, expert_stop)):
        start = expert_id * GEOMETRY_397B_2BIT.expert_size
        stop = start + GEOMETRY_397B_2BIT.expert_size
        entry_override = dict(entry)
        entry_override["quant_type"] = target_qtype.name
        entry_override["bytes_per_expert"] = bytes_per_expert
        quantized = convert_expert_tensor(memoryview(layer_blob)[start:stop], entry_override, native_quantizer)
        dst_start = local_index * bytes_per_expert
        converted[dst_start : dst_start + bytes_per_expert] = quantized

    return expert_start, bytes(converted), expert_stop - expert_start


def convert_layer(
    source_pack: str,
    out_dir: str,
    layer: int,
    layer_entries: list[dict[str, Any]],
    ggml_base_dylib: str,
    expert_jobs: int,
    expert_chunk_size: int,
    quant_type_name: str | None,
) -> tuple[int, list[dict[str, Any]], int, str]:
    native_quantizer = NativeQuantizer(Path(ggml_base_dylib))
    forced_qtype = parse_quant_type(quant_type_name)
    layer_start = time.monotonic()
    total_units = len(layer_entries) * GEOMETRY_397B_2BIT.num_experts
    completed_units = 0
    next_report = layer_start + 5.0

    layer_path = Path(source_pack) / f"layer_{layer:02d}.bin"
    if not layer_path.exists():
        raise FileNotFoundError(f"missing source layer file: {layer_path}")

    layer_blob = layer_path.read_bytes()
    expected_size = GEOMETRY_397B_2BIT.expert_size * GEOMETRY_397B_2BIT.num_experts
    if len(layer_blob) != expected_size:
        raise ValueError(
            f"source layer file {layer_path} has {len(layer_blob)} bytes, expected {expected_size}"
        )

    out_name = f"layer_{layer:03d}.bin"
    out_path = Path(out_dir) / out_name
    offset = 0
    converted_entries: list[dict[str, Any]] = []
    total_bytes = 0

    print(
        f"[convert] layer {layer:02d}: start ({len(layer_entries)} tensors x {GEOMETRY_397B_2BIT.num_experts} experts)",
        flush=True,
    )

    with out_path.open("wb") as out_handle:
        for entry in layer_entries:
            target_qtype = resolve_target_qtype(entry, forced_qtype)
            bytes_per_expert = bytes_per_expert_for(entry, target_qtype, native_quantizer)
            converted = bytearray(bytes_per_expert * GEOMETRY_397B_2BIT.num_experts)
            family = str(entry["tensor_family"])
            if expert_jobs <= 1:
                for expert_id in range(GEOMETRY_397B_2BIT.num_experts):
                    start = expert_id * GEOMETRY_397B_2BIT.expert_size
                    stop = start + GEOMETRY_397B_2BIT.expert_size
                    entry_override = dict(entry)
                    entry_override["quant_type"] = target_qtype.name
                    entry_override["bytes_per_expert"] = bytes_per_expert
                    quantized = convert_expert_tensor(memoryview(layer_blob)[start:stop], entry_override, native_quantizer)
                    dst_start = expert_id * bytes_per_expert
                    converted[dst_start : dst_start + bytes_per_expert] = quantized
                    completed_units += 1

                    now = time.monotonic()
                    if now >= next_report or expert_id + 1 == GEOMETRY_397B_2BIT.num_experts:
                        elapsed = now - layer_start
                        rate = completed_units / elapsed if elapsed > 0 else 0.0
                        remaining = total_units - completed_units
                        eta = remaining / rate if rate > 0 else None
                        pct = 100.0 * completed_units / total_units if total_units else 100.0
                        print(
                            "[convert] "
                            f"layer {layer:02d} {family}: "
                            f"{expert_id + 1}/{GEOMETRY_397B_2BIT.num_experts} experts, "
                            f"{completed_units}/{total_units} steps ({pct:.1f}%), "
                            f"elapsed {format_duration(elapsed)}, eta {format_duration(eta)}",
                            flush=True,
                        )
                        next_report = now + 5.0
            else:
                if expert_chunk_size > 0:
                    chunk_size = expert_chunk_size
                else:
                    chunk_size = max(1, GEOMETRY_397B_2BIT.num_experts // (expert_jobs * 4))
                    chunk_size = min(chunk_size, 32)
                futures = []
                with ThreadPoolExecutor(max_workers=expert_jobs) as executor:
                    for expert_start in range(0, GEOMETRY_397B_2BIT.num_experts, chunk_size):
                        expert_stop = min(GEOMETRY_397B_2BIT.num_experts, expert_start + chunk_size)
                        futures.append(
                            executor.submit(
                                convert_expert_chunk,
                                layer_blob,
                                entry,
                                ggml_base_dylib,
                                expert_start,
                                expert_stop,
                                quant_type_name,
                            )
                        )

                    for future in as_completed(futures):
                        expert_start, chunk_bytes, chunk_count = future.result()
                        dst_start = expert_start * bytes_per_expert
                        converted[dst_start : dst_start + len(chunk_bytes)] = chunk_bytes
                        completed_units += chunk_count

                        now = time.monotonic()
                        if now >= next_report or completed_units == total_units:
                            elapsed = now - layer_start
                            rate = completed_units / elapsed if elapsed > 0 else 0.0
                            remaining = total_units - completed_units
                            eta = remaining / rate if rate > 0 else None
                            pct = 100.0 * completed_units / total_units if total_units else 100.0
                            done_for_family = min(
                                GEOMETRY_397B_2BIT.num_experts,
                                completed_units - (len(converted_entries) * GEOMETRY_397B_2BIT.num_experts),
                            )
                            print(
                                "[convert] "
                                f"layer {layer:02d} {family}: "
                                f"{done_for_family}/{GEOMETRY_397B_2BIT.num_experts} experts, "
                                f"{completed_units}/{total_units} steps ({pct:.1f}%), "
                                f"elapsed {format_duration(elapsed)}, eta {format_duration(eta)}",
                                flush=True,
                            )
                            next_report = now + 5.0

            out_handle.write(converted)
            updated = dict(entry)
            updated["repacked_file"] = out_name
            updated["repacked_offset"] = offset
            updated["quant_type"] = target_qtype.name
            updated["bytes_per_expert"] = bytes_per_expert
            updated["exact_byte_length"] = len(converted)
            converted_entries.append(updated)
            offset += len(converted)
            total_bytes += len(converted)

            family_elapsed = time.monotonic() - layer_start
            print(
                f"[convert] layer {layer:02d}: finished {family} ({len(converted)} bytes written, elapsed {format_duration(family_elapsed)})",
                flush=True,
            )

    layer_elapsed = time.monotonic() - layer_start
    print(
        f"[convert] layer {layer:02d}: done in {format_duration(layer_elapsed)} ({total_bytes} bytes)",
        flush=True,
    )
    return layer, converted_entries, total_bytes, str(out_path)


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.expert_jobs < 1:
        raise SystemExit("--expert-jobs must be >= 1")
    if args.expert_chunk_size < 0:
        raise SystemExit("--expert-chunk-size must be >= 0")
    parse_quant_type(args.target_quant_type)
    layer_filter = parse_layers(args.layers)
    manifest = load_template_manifest(args.template_sidecar)
    if not GGML_BASE_DYLIB.exists():
        raise SystemExit(f"missing ggml quantizer dylib: {GGML_BASE_DYLIB}")
    entries = [entry for entry in manifest["entries"] if layer_filter is None or int(entry["layer"]) in layer_filter]
    if not entries:
        raise SystemExit("no template manifest entries matched the requested layer filter")

    ensure_output_dir(args.out_dir, args.force)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(int(entry["layer"]), []).append(dict(entry))

    converted_entries: list[dict[str, Any]] = []
    total_bytes = 0
    total_layers = len(grouped)
    completed_layers = 0
    overall_start = time.monotonic()
    sorted_layers = sorted(grouped.items())
    forced_qtype = parse_quant_type(args.target_quant_type)

    def report_overall_progress(layer: int, layer_total_bytes: int, out_path: str) -> None:
        nonlocal completed_layers, total_bytes
        completed_layers += 1
        elapsed = time.monotonic() - overall_start
        rate = completed_layers / elapsed if elapsed > 0 else 0.0
        remaining = total_layers - completed_layers
        eta = remaining / rate if rate > 0 else None
        total_bytes += layer_total_bytes
        print(
            "[convert] overall: "
            f"{completed_layers}/{total_layers} layers complete "
            f"({100.0 * completed_layers / total_layers:.1f}%), "
            f"last=layer {layer:02d}, elapsed {format_duration(elapsed)}, eta {format_duration(eta)}, "
            f"bytes {total_bytes}, out={out_path}",
            flush=True,
        )

    if args.manifest_only:
        native_quantizer = NativeQuantizer(GGML_BASE_DYLIB)
        for layer, layer_entries in sorted_layers:
            source_layer_path = (args.source_pack / f"layer_{layer:02d}.bin").resolve()
            if not source_layer_path.exists():
                raise SystemExit(f"missing source layer file: {source_layer_path}")

            layer_total_bytes = 0
            for entry in layer_entries:
                target_qtype = resolve_target_qtype(entry, forced_qtype)
                bytes_per_expert = bytes_per_expert_for(entry, target_qtype, native_quantizer)
                updated = dict(entry)
                updated["repacked_file"] = str(source_layer_path)
                updated["repacked_offset"] = 0
                updated["quant_type"] = target_qtype.name
                updated["bytes_per_expert"] = bytes_per_expert
                updated["exact_byte_length"] = bytes_per_expert * GEOMETRY_397B_2BIT.num_experts
                converted_entries.append(updated)
                layer_total_bytes += int(updated["exact_byte_length"])

            print(
                f"[convert] manifest-only layer {layer:02d}: mapped {len(layer_entries)} tensors to {source_layer_path}",
                flush=True,
            )
            report_overall_progress(layer, layer_total_bytes, str(source_layer_path))
    else:
        if args.jobs == 1:
            for layer, layer_entries in sorted_layers:
                layer, layer_converted_entries, layer_total_bytes, out_path = convert_layer(
                    str(args.source_pack),
                    str(args.out_dir),
                    layer,
                    layer_entries,
                    str(GGML_BASE_DYLIB),
                    args.expert_jobs,
                    args.expert_chunk_size,
                    args.target_quant_type,
                )
                converted_entries.extend(layer_converted_entries)
                report_overall_progress(layer, layer_total_bytes, out_path)
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                for layer, layer_entries in sorted_layers:
                    futures.append(
                        executor.submit(
                            convert_layer,
                            str(args.source_pack),
                            str(args.out_dir),
                            layer,
                            layer_entries,
                            str(GGML_BASE_DYLIB),
                            args.expert_jobs,
                            args.expert_chunk_size,
                            args.target_quant_type,
                        )
                    )

                for future in as_completed(futures):
                    layer, layer_converted_entries, layer_total_bytes, out_path = future.result()
                    converted_entries.extend(layer_converted_entries)
                    report_overall_progress(layer, layer_total_bytes, out_path)

    converted_entries.sort(key=lambda entry: (int(entry["layer"]), str(entry["tensor_family"]), str(entry["tensor_name"])))

    out_manifest = {
        "schema_version": 1,
        "sidecar_kind": "flashmoe_affine_2bit_qwen397b" if args.manifest_only else "flashmoe_gguf",
        "layout": "layer_major_whole_tensor",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "affine_pack": str(args.source_pack),
            "template_sidecar": str(args.template_sidecar),
            "note": (
                "Manifest-only runtime sidecar: reads legacy affine 2-bit packed experts and requantizes on slot install."
                if args.manifest_only else
                "Converted from legacy affine 2-bit packed experts into GGUF-compatible routed tensor bytes."
            ),
        },
        "model": manifest.get("model", {}),
        "entries": converted_entries,
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(out_manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"wrote {len(converted_entries)} entries across {len(grouped)} layers to {args.out_dir}")
    print(f"total bytes: {total_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
