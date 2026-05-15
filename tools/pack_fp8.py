"""Lossless FP8 safetensors pack writer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pcketlm.core.model_import.fp8_pack_plan import plan_model_dir_to_fp8_pack
from pcketlm.core.model_import.q4_plan import (
    _canonical_dtype,
    _is_fp8_dtype,
    _read_safetensors_header,
    _scale_partner_name,
    _tensor_layer_and_expert,
    _weight_map,
)
from pcketlm.core.storage.paths import state_root

DEFAULT_PACK_BYTES = 16 * 1024**3
FORMAT = "pcketlm-fp8-pack-v1"
SCALE_SUFFIXES = (".scale_inv", "_scale_inv", ".scale", "_scale")


@dataclass(frozen=True, slots=True)
class PackTask:
    name: str
    shard_name: str
    dtype: str
    shape: list[int]
    data_offset_start: int
    data_offset_end: int
    tensor_role: str
    scale_name: str | None
    weight_name: str | None
    layer_index: int | None
    expert_index: int | None
    pack_order: int

    @property
    def byte_length(self) -> int:
        return int(self.data_offset_end) - int(self.data_offset_start)


def fp8_pack_state_path(model_id: str) -> Path:
    safe_model_id = model_id.replace("/", "_").replace("\\", "_")
    return state_root() / "fp8_packs" / f"{safe_model_id}.json"


def pack_model_dir_to_fp8(
    model_dir: Path,
    output_dir: Path | None = None,
    *,
    model_id: str | None = None,
    pack_bytes: int = DEFAULT_PACK_BYTES,
    resume: bool = True,
    check_disk_space: bool = True,
    max_new_tensors: int | None = None,
) -> dict:
    """Write or resume a lossless FP8 pack artifact."""
    started = time.perf_counter()
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else model_dir / "artifacts" / "fp8_pack"
    model_id = model_id or model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = fp8_pack_state_path(model_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pack_manifest.json"

    plan = plan_model_dir_to_fp8_pack(model_dir, output_dir)
    if not plan.get("ready_for_pack"):
        return _write_state(
            state_path,
            {
                "model_id": model_id,
                "status": "blocked",
                "source_model_dir": str(model_dir),
                "output_dir": str(output_dir),
                "plan": plan,
                "blockers": _plan_blockers(plan),
                "updated_at": _now(),
            },
        )
    required = int(plan["estimated_packed_bytes"]) + int(plan["estimated_manifest_overhead_bytes"])
    if check_disk_space:
        free_bytes = _free_bytes_for(output_dir)
        if free_bytes < required:
            return _write_state(
                state_path,
                {
                    "model_id": model_id,
                    "status": "blocked",
                    "source_model_dir": str(model_dir),
                    "output_dir": str(output_dir),
                    "plan": plan,
                    "free_bytes": free_bytes,
                    "required_bytes": required,
                    "blockers": [
                        f"Not enough free disk space for FP8 pack: need {required} bytes, have {free_bytes} bytes."
                    ],
                    "updated_at": _now(),
                },
            )

    tasks = _build_pack_tasks(model_dir)
    manifest = _read_manifest(manifest_path) if resume else None
    if manifest is None:
        manifest = _empty_manifest(model_dir, output_dir, plan, int(pack_bytes))
    completed = set(str(name) for name in (manifest.get("tensors") or {}))
    next_pack_index = _next_pack_index(manifest)
    pack_file = _open_tmp_pack(output_dir, next_pack_index)
    current_size = 0
    new_tensors = 0
    finalized_any = False

    try:
        for task in tasks:
            if task.name in completed:
                continue
            if max_new_tensors is not None and new_tensors >= max_new_tensors:
                break
            if current_size > 0 and current_size + task.byte_length > int(pack_bytes):
                _finalize_pack(output_dir, next_pack_index, pack_file)
                manifest["pack_files"].append(_pack_file_record(output_dir, next_pack_index))
                _write_manifest(manifest_path, manifest)
                finalized_any = True
                next_pack_index += 1
                pack_file = _open_tmp_pack(output_dir, next_pack_index)
                current_size = 0
            source_path = model_dir / task.shard_name
            source_offset = _source_data_base_offset(source_path) + task.data_offset_start
            _copy_exact(source_path, source_offset, pack_file, task.byte_length)
            entry = _manifest_entry(task, next_pack_index, current_size)
            manifest["tensors"][task.name] = entry
            if task.weight_name and task.weight_name in manifest["tensors"]:
                weight_entry = manifest["tensors"][task.weight_name]
                weight_entry["scale_name"] = task.name
                weight_entry["scale_dtype"] = task.dtype
                weight_entry["scale_shape"] = list(task.shape)
                weight_entry["scale_pack_file_index"] = next_pack_index
                weight_entry["scale_byte_offset"] = current_size
                weight_entry["scale_byte_length"] = task.byte_length
            if task.scale_name and task.scale_name in manifest["tensors"]:
                scale_entry = manifest["tensors"][task.scale_name]
                entry["scale_pack_file_index"] = scale_entry["pack_file_index"]
                entry["scale_byte_offset"] = scale_entry["byte_offset"]
                entry["scale_byte_length"] = scale_entry["byte_length"]
                entry["scale_dtype"] = scale_entry["dtype"]
                entry["scale_shape"] = list(scale_entry["shape"])
            current_size += task.byte_length
            new_tensors += 1
            completed.add(task.name)
    except Exception as exc:
        pack_file.close()
        return _write_state(
            state_path,
            {
                **_state_base(model_id, model_dir, output_dir, plan, manifest, len(tasks), started),
                "status": "error",
                "error": str(exc),
                "updated_at": _now(),
            },
        )

    if current_size > 0:
        _finalize_pack(output_dir, next_pack_index, pack_file)
        manifest["pack_files"].append(_pack_file_record(output_dir, next_pack_index))
        _write_manifest(manifest_path, manifest)
        finalized_any = True
    else:
        pack_file.close()
        tmp = output_dir / f"pack_{next_pack_index:04d}.bin.tmp"
        if tmp.exists():
            tmp.unlink()

    complete = len(manifest["tensors"]) >= len(tasks)
    status = "complete" if complete else "paused"
    return _write_state(
        state_path,
        {
            **_state_base(model_id, model_dir, output_dir, plan, manifest, len(tasks), started),
            "status": status,
            "new_tensors_written": int(new_tensors),
            "finalized_pack_file": bool(finalized_any),
            "manifest_path": str(manifest_path),
            "updated_at": _now(),
        },
    )


def load_fp8_pack_state(model_id: str) -> dict | None:
    path = fp8_pack_state_path(model_id)
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _build_pack_tasks(model_dir: Path) -> list[PackTask]:
    weight_map = _weight_map(model_dir / "model.safetensors.index.json")
    all_names = set(weight_map)
    headers: dict[str, dict] = {}
    tasks: list[PackTask] = []
    for name, shard_name in weight_map.items():
        if shard_name not in headers:
            headers[shard_name] = _read_safetensors_header(model_dir / shard_name)
        metadata = headers[shard_name].get(name)
        if metadata is None:
            continue
        dtype = _canonical_dtype(str(metadata.get("dtype", "")))
        shape = [int(value) for value in metadata.get("shape", [])]
        offsets = metadata.get("data_offsets") or [0, 0]
        weight_name = _scale_partner_name(name, all_names)
        scale_name = None if weight_name is not None else _scale_name_for_weight(name, all_names)
        layer, expert = _tensor_layer_and_expert(name)
        role = "scale_companion" if weight_name is not None else "weight"
        tasks.append(
            PackTask(
                name=name,
                shard_name=shard_name,
                dtype=dtype,
                shape=shape,
                data_offset_start=int(offsets[0]),
                data_offset_end=int(offsets[1]),
                tensor_role=role,
                scale_name=scale_name,
                weight_name=weight_name,
                layer_index=layer,
                expert_index=expert,
                pack_order=0,
            )
        )
    tasks.sort(key=lambda task: _pack_sort_key(task))
    return [replace(task, pack_order=index) for index, task in enumerate(tasks)]


def _pack_sort_key(task: PackTask) -> tuple:
    layer = -1 if task.layer_index is None else int(task.layer_index)
    expert = -1 if task.expert_index is None else int(task.expert_index)
    lowered = task.name.lower()
    if task.name in {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} or task.layer_index is None:
        group = 0
    elif ".self_attn." in lowered:
        group = 1
    elif ".mlp.gate." in lowered:
        group = 2
    elif ".mlp.experts." in lowered:
        group = 3
    elif ".shared_experts." in lowered:
        group = 4
    elif ".mlp." in lowered:
        group = 5
    else:
        group = 6
    projection_order = 0
    for index, marker in enumerate(("gate_proj", "up_proj", "down_proj", "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "o_proj")):
        if marker in lowered:
            projection_order = index
            break
    role_order = 1 if task.tensor_role == "scale_companion" else 0
    return (layer, group, expert, projection_order, _base_name(task.name), role_order, task.name)


def _scale_name_for_weight(name: str, all_names: set[str]) -> str | None:
    for suffix in SCALE_SUFFIXES:
        candidate = f"{name}{suffix}"
        if candidate in all_names:
            return candidate
    return None


def _base_name(name: str) -> str:
    for suffix in SCALE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _manifest_entry(task: PackTask, pack_index: int, offset: int) -> dict:
    return {
        "name": task.name,
        "dtype": task.dtype,
        "shape": list(task.shape),
        "tensor_role": task.tensor_role,
        "pack_file_index": int(pack_index),
        "byte_offset": int(offset),
        "byte_length": int(task.byte_length),
        "scale_name": task.scale_name,
        "scale_pack_file_index": None,
        "scale_byte_offset": None,
        "scale_byte_length": None,
        "scale_dtype": None,
        "scale_shape": None,
        "weight_name": task.weight_name,
        "layer_index": task.layer_index,
        "expert_index": task.expert_index,
        "pack_order": int(task.pack_order),
    }


def _empty_manifest(model_dir: Path, output_dir: Path, plan: dict, pack_bytes: int) -> dict:
    return {
        "format": FORMAT,
        "source_model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "pack_bytes": int(pack_bytes),
        "plan": plan,
        "pack_files": [],
        "tensors": {},
        "created_at": _now(),
        "updated_at": _now(),
    }


def _state_base(model_id: str, model_dir: Path, output_dir: Path, plan: dict, manifest: dict, total: int, started: float) -> dict:
    completed = len(manifest.get("tensors") or {})
    return {
        "model_id": model_id,
        "source_model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "plan": plan,
        "completed_tensor_count": int(completed),
        "source_tensor_count": int(total),
        "completed_pack_file_count": len(manifest.get("pack_files") or []),
        "pack_output_bytes": int(sum(int(item.get("byte_length", 0)) for item in manifest.get("pack_files") or [])),
        "progress_pct": None if total <= 0 else round(min(completed / total, 1.0) * 100, 2),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _next_pack_index(manifest: dict) -> int:
    pack_files = manifest.get("pack_files") or []
    if not pack_files:
        return 0
    return max(int(item.get("index", 0)) for item in pack_files) + 1


def _open_tmp_pack(output_dir: Path, index: int):
    tmp = output_dir / f"pack_{int(index):04d}.bin.tmp"
    if tmp.exists():
        tmp.unlink()
    return tmp.open("wb")


def _finalize_pack(output_dir: Path, index: int, handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    tmp = output_dir / f"pack_{int(index):04d}.bin.tmp"
    final = output_dir / f"pack_{int(index):04d}.bin"
    tmp.replace(final)


def _pack_file_record(output_dir: Path, index: int) -> dict:
    path = output_dir / f"pack_{int(index):04d}.bin"
    return {
        "index": int(index),
        "file_name": path.name,
        "byte_length": int(path.stat().st_size),
    }


def _copy_exact(source_path: Path, source_offset: int, output_handle, byte_length: int) -> None:
    remaining = int(byte_length)
    with source_path.open("rb") as source:
        source.seek(int(source_offset))
        while remaining > 0:
            chunk = source.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise OSError(f"Unexpected EOF while reading {source_path}.")
            output_handle.write(chunk)
            remaining -= len(chunk)


def _source_data_base_offset(path: Path) -> int:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
    return 8 + int(header_size)


def _read_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = _now()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_state(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plan_blockers(plan: dict) -> list[str]:
    missing = plan.get("missing_shards") or []
    blockers = [f"Missing source shards: {', '.join(missing)}."] if missing else []
    if not plan.get("fp8_native"):
        blockers.append("Source is not FP8-native.")
    return blockers or ["FP8 pack plan is not ready."]


def _free_bytes_for(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a lossless pcketlm FP8 pack artifact.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--pack-gb", type=float, default=16.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-disk-check", action="store_true")
    parser.add_argument("--max-new-tensors", type=int, default=None, help="Test hook: stop after writing this many new tensors.")
    args = parser.parse_args()
    state = pack_model_dir_to_fp8(
        args.model_dir,
        args.output_dir,
        model_id=args.model_id,
        pack_bytes=max(1, int(args.pack_gb * 1024**3)),
        resume=not args.no_resume,
        check_disk_space=not args.skip_disk_check,
        max_new_tensors=args.max_new_tensors,
    )
    print(
        json.dumps(
            {
                key: state.get(key)
                for key in (
                    "model_id",
                    "status",
                    "progress_pct",
                    "completed_tensor_count",
                    "source_tensor_count",
                    "completed_pack_file_count",
                    "pack_output_bytes",
                    "manifest_path",
                    "elapsed_seconds",
                    "blockers",
                    "error",
                )
                if key in state
            },
            indent=2,
        )
    )
    return 0 if state.get("status") in {"complete", "paused"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
