"""Check whether this machine can run the local DeepSeek V3 FP8 GPU path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcketlm.core.runtime.deepseek_gpu_residency import (  # noqa: E402
    estimate_deepseek_gpu_residency,
    run_local_deepseek_paged_decode_loop,
)
from pcketlm.core.runtime.fp8_pack import default_fp8_pack_dir  # noqa: E402
from pcketlm.core.runtime.tensor_catalog import build_tensor_catalog, load_tensor_catalog  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check CUDA, DeepSeek tensor catalog, FP8 pack, and optionally run the local GPU pager."
    )
    parser.add_argument("--model-id", default="deepseek-v3")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--build-catalog", action="store_true")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--budget-gb", type=float, default=12.0)
    parser.add_argument("--prefetch-window", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--tail-chunk-rows", type=int, default=8192)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-cpu-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = check_deepseek_gpu_ready(
        model_id=args.model_id,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        build_catalog=args.build_catalog,
        start_layer=args.layer,
        layer_count=args.layers,
        resident_budget_bytes=int(float(args.budget_gb) * 1024**3),
        prefetch_window=args.prefetch_window,
        max_new_tokens=args.max_new_tokens,
        token_id=args.token_id,
        tail_chunk_rows=args.tail_chunk_rows,
        run=args.run,
        allow_cpu_run=args.allow_cpu_run,
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plain(payload)
    return _exit_code(payload)


def check_deepseek_gpu_ready(
    *,
    model_id: str = "deepseek-v3",
    model_dir: Path | None = None,
    build_catalog: bool = False,
    start_layer: int = 3,
    layer_count: int = 3,
    resident_budget_bytes: int = 12 * 1024**3,
    prefetch_window: int = 0,
    max_new_tokens: int = 1,
    token_id: int = 0,
    tail_chunk_rows: int = 8192,
    run: bool = False,
    allow_cpu_run: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable readiness report for local DeepSeek GPU execution."""
    blockers: list[str] = []
    cuda = _cuda_status()

    if model_dir is not None and build_catalog:
        try:
            build_tensor_catalog(str(model_id), Path(model_dir))
        except Exception as exc:  # noqa: BLE001 - report the exact readiness blocker.
            blockers.append(f"Could not build tensor catalog: {type(exc).__name__}: {exc}")

    catalog_payload: dict[str, Any]
    pack_payload: dict[str, Any]
    estimate_payload: dict[str, Any] | None = None
    run_payload: dict[str, Any] | None = None
    try:
        catalog = load_tensor_catalog(str(model_id))
        catalog_payload = {
            "ready": bool(catalog.ready),
            "model_id": str(catalog.model_id),
            "model_dir": str(catalog.model_dir),
            "catalog_path": str(catalog.catalog_path),
            "tensor_count": int(catalog.tensor_count),
            "shard_count": int(catalog.shard_count),
            "config_hidden_layers": int(catalog.num_hidden_layers or catalog.layer_count or 0),
            "fp8_weight_count": int(catalog.fp8_weight_count),
            "fp8_scale_count": int(catalog.fp8_scale_count),
            "fp8_weight_bytes": int(catalog.fp8_weight_bytes),
            "fp8_scale_bytes": int(catalog.fp8_scale_bytes),
            "blockers": list(catalog.blockers),
        }
        blockers.extend(str(item) for item in catalog.blockers)
        if not catalog.ready:
            blockers.append("Tensor catalog is not ready.")
        pack_payload = _pack_status(catalog.model_dir)
        if not pack_payload["manifest_exists"]:
            blockers.append(f"FP8 pack manifest is missing: {pack_payload['manifest_path']}")
        if int(pack_payload["pack_file_count"]) <= 0:
            blockers.append(f"FP8 pack files are missing: {pack_payload['pack_dir']}")
        estimate = estimate_deepseek_gpu_residency(
            str(model_id),
            start_layer=int(start_layer),
            layer_count=int(layer_count),
            resident_budget_bytes=int(resident_budget_bytes),
        )
        estimate_payload = estimate.to_dict()
        blockers.extend(str(item) for item in estimate.blockers)
    except Exception as exc:  # noqa: BLE001 - readiness is a diagnostic command.
        catalog_payload = {
            "ready": False,
            "model_id": str(model_id),
            "model_dir": "" if model_dir is None else str(model_dir),
            "catalog_path": "",
            "tensor_count": 0,
            "shard_count": 0,
            "config_hidden_layers": 0,
            "fp8_weight_count": 0,
            "fp8_scale_count": 0,
            "fp8_weight_bytes": 0,
            "fp8_scale_bytes": 0,
            "blockers": [f"{type(exc).__name__}: {exc}"],
        }
        pack_payload = _pack_status(model_dir)
        blockers.append(f"Could not load readiness inputs: {type(exc).__name__}: {exc}")

    if not cuda["available"]:
        blockers.append("CUDA is not available on this machine.")

    if run:
        if not cuda["available"] and not allow_cpu_run:
            run_payload = {
                "passed": False,
                "blockers": ["Local pager run skipped because CUDA is missing. Use --allow-cpu-run only for diagnostics."],
            }
        else:
            device = "cuda" if cuda["available"] else "cpu"
            dtype = torch.float16 if cuda["available"] else torch.float32
            result = run_local_deepseek_paged_decode_loop(
                str(model_id),
                [int(token_id)],
                start_layer=int(start_layer),
                layer_count=max(1, int(layer_count)),
                max_new_tokens=max(0, int(max_new_tokens)),
                resident_budget_bytes=int(resident_budget_bytes),
                prefetch_window=max(0, int(prefetch_window)),
                tail_chunk_rows=max(1, int(tail_chunk_rows)),
                device=device,
                require_cuda=bool(cuda["available"]),
                dtype=dtype,
            )
            run_payload = result.to_dict()
            if not result.passed:
                blockers.extend(str(item) for item in result.blockers)

    blockers = list(dict.fromkeys(item for item in blockers if item))
    ready = bool(
        cuda["available"]
        and catalog_payload.get("ready")
        and pack_payload.get("manifest_exists")
        and int(pack_payload.get("pack_file_count", 0)) > 0
        and not blockers
    )
    return {
        "ready": ready,
        "blockers": blockers,
        "cuda": cuda,
        "catalog": catalog_payload,
        "pack": pack_payload,
        "estimate": estimate_payload,
        "run": run_payload,
        "next_action": _next_action(ready=ready, blockers=blockers, run=run, run_payload=run_payload),
    }


def _cuda_status() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if available else 0
    devices = []
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": int(index),
                "name": str(torch.cuda.get_device_name(index)),
                "total_memory_bytes": int(props.total_memory),
            }
        )
    return {
        "available": available,
        "device_count": device_count,
        "devices": devices,
        "torch_version": str(torch.__version__),
    }


def _pack_status(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None or not str(model_dir):
        pack_dir = Path()
    else:
        pack_dir = default_fp8_pack_dir(Path(model_dir))
    manifest_path = pack_dir / "pack_manifest.json" if str(pack_dir) else Path("pack_manifest.json")
    pack_files = sorted(pack_dir.glob("pack_*.bin")) if pack_dir.exists() else []
    return {
        "pack_dir": str(pack_dir),
        "manifest_path": str(manifest_path),
        "manifest_exists": bool(manifest_path.exists()),
        "pack_file_count": len(pack_files),
        "pack_bytes": int(sum(path.stat().st_size for path in pack_files)),
    }


def _next_action(*, ready: bool, blockers: list[str], run: bool, run_payload: dict[str, Any] | None) -> str:
    if ready and run_payload and run_payload.get("passed"):
        return "GPU worker is ready and the local DeepSeek pager run passed."
    if ready and not run:
        return "Run again with --run on this GPU worker to execute the local resident-pager decode loop."
    if any("CUDA is not available" in item for item in blockers):
        return "Move this DeepSeek source/pack to a CUDA machine, or attach this drive to one, then rerun this command."
    if any("catalog" in item.lower() for item in blockers):
        return "Build the tensor catalog with --model-dir <DeepSeek path> --build-catalog."
    if any("pack" in item.lower() for item in blockers):
        return "Build or attach the lossless FP8 pack before running the local GPU pager."
    if run and run_payload and not run_payload.get("passed"):
        return "Fix the local pager blocker shown in the run section, then rerun with --run."
    return "Fix the listed blockers, then rerun the readiness check."


def _exit_code(payload: dict[str, Any]) -> int:
    if payload.get("ready"):
        run_payload = payload.get("run")
        if run_payload is None or run_payload.get("passed"):
            return 0
        return 1
    if any("CUDA is not available" in str(item) for item in payload.get("blockers", [])):
        return 2
    return 1


def _print_plain(payload: dict[str, Any]) -> None:
    cuda = payload["cuda"]
    catalog = payload["catalog"]
    pack = payload["pack"]
    estimate = payload.get("estimate") or {}
    print(f"DeepSeek GPU ready: {payload['ready']}")
    print(f"CUDA: {cuda['available']} ({cuda['device_count']} device(s), torch {cuda['torch_version']})")
    print(f"Catalog: {catalog['ready']} ({catalog['tensor_count']} tensors, {catalog['model_dir']})")
    print(f"FP8 pack: {pack['manifest_exists']} ({pack['pack_file_count']} files, {pack['pack_bytes']} bytes)")
    if estimate:
        print(
            "Estimate: "
            f"{estimate.get('projected_hot_seconds_per_token', 0.0):.3f}s/token projected, "
            f"{estimate.get('layers_fit_by_dequantized_bytes', 0)} active layers fit by dequantized bytes"
        )
    run_payload = payload.get("run")
    if run_payload is not None:
        print(f"Run: {run_payload.get('passed')} ({run_payload.get('elapsed_seconds', 0.0)}s)")
    if payload["blockers"]:
        print("Blockers:")
        for item in payload["blockers"]:
            print(f"- {item}")
    print(f"Next: {payload['next_action']}")


if __name__ == "__main__":
    raise SystemExit(main())
