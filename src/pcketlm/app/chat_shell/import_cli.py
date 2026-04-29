"""Small import command for local model folders."""

from __future__ import annotations

import argparse
import fnmatch
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pcketlm.core.model_import.service import ImportRequest, import_model
from pcketlm.core.storage.paths import state_root


ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.txt",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer*",
    "vocab*",
    "merges.txt",
]
IGNORE_PATTERNS = [
    "*.bin",
    "*.gguf",
    "*.onnx",
    "*.msgpack",
    "onnx/*",
]


def _download_status_path(model_id: str) -> Path:
    safe_model_id = model_id.replace("/", "_").replace("\\", "_")
    return state_root() / "downloads" / f"{safe_model_id}.json"


def _matches_patterns(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern) for pattern in patterns)


def _wanted_repo_file(path: str) -> bool:
    return _matches_patterns(path, ALLOW_PATTERNS) and not _matches_patterns(path, IGNORE_PATTERNS)


def _expected_repo_files(repo_id: str) -> list[dict]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    files = []
    for sibling in info.siblings:
        name = str(sibling.rfilename)
        if not _wanted_repo_file(name):
            continue
        files.append({"path": name, "size": int(sibling.size or 0)})
    return files


def _bytes_under(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _present_expected_files(target_dir: Path, expected_files: list[dict]) -> int:
    return sum(1 for item in expected_files if (target_dir / str(item["path"])).exists())


def _write_download_status(
    *,
    model_id: str,
    repo_id: str,
    target_dir: Path,
    expected_files: list[dict],
    status: str,
    started_at: str,
    error: str | None = None,
) -> dict:
    expected_bytes = sum(int(item["size"]) for item in expected_files)
    bytes_on_disk = _bytes_under(target_dir)
    progress_pct = None if expected_bytes <= 0 else round(min(bytes_on_disk / expected_bytes, 1.0) * 100, 2)
    payload = {
        "model_id": model_id,
        "repo_id": repo_id,
        "target_dir": str(target_dir),
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "bytes_on_disk": bytes_on_disk,
        "bytes_on_disk_gb": round(bytes_on_disk / (1024**3), 2),
        "expected_bytes": expected_bytes,
        "expected_bytes_gb": round(expected_bytes / (1024**3), 2),
        "progress_pct": progress_pct,
        "expected_file_count": len(expected_files),
        "present_expected_file_count": _present_expected_files(target_dir, expected_files),
        "error": error,
    }
    path = _download_status_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _format_download_line(payload: dict) -> str:
    return (
        f"{payload['model_id']} download {payload['progress_pct']}% "
        f"({payload['bytes_on_disk_gb']} / {payload['expected_bytes_gb']} GB), "
        f"files {payload['present_expected_file_count']}/{payload['expected_file_count']}, "
        f"status={payload['status']}"
    )


def download_huggingface_snapshot(repo_id: str, target_dir: Path, *, model_id: str, progress_interval_seconds: int = 30) -> Path:
    """Download a safetensors text-model snapshot into the requested local folder."""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    expected_files = _expected_repo_files(repo_id)
    started_at = datetime.now(timezone.utc).isoformat()
    stop_event = threading.Event()

    def monitor() -> None:
        while not stop_event.wait(progress_interval_seconds):
            payload = _write_download_status(
                model_id=model_id,
                repo_id=repo_id,
                target_dir=target_dir,
                expected_files=expected_files,
                status="downloading",
                started_at=started_at,
            )
            print(_format_download_line(payload), flush=True)

    payload = _write_download_status(
        model_id=model_id,
        repo_id=repo_id,
        target_dir=target_dir,
        expected_files=expected_files,
        status="starting",
        started_at=started_at,
    )
    print(_format_download_line(payload), flush=True)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
    except Exception as exc:
        _write_download_status(
            model_id=model_id,
            repo_id=repo_id,
            target_dir=target_dir,
            expected_files=expected_files,
            status="error",
            started_at=started_at,
            error=str(exc),
        )
        raise
    finally:
        stop_event.set()
        thread.join(timeout=2)
    payload = _write_download_status(
        model_id=model_id,
        repo_id=repo_id,
        target_dir=target_dir,
        expected_files=expected_files,
        status="complete",
        started_at=started_at,
    )
    print(_format_download_line(payload), flush=True)
    return target_dir


def build_import_request(args: argparse.Namespace) -> ImportRequest:
    """Build an import request from CLI arguments."""
    return ImportRequest(
        model_id=args.model_id,
        label=args.label or args.model_id,
        family=args.family,
        source_path=args.source_path,
        repo_id=args.repo_id,
        source_origin=args.source_origin,
        source_kind="local-folder",
        model_type=args.model_type,
        format_name=args.format_name,
    )


def main() -> int:
    """Run the local import command."""
    parser = argparse.ArgumentParser(description="Import a local model folder into pcketlm.")
    parser.add_argument("model_id", help="Stable pcketlm model id")
    parser.add_argument("source_path", type=Path, help="Path to the local model folder")
    parser.add_argument("--label", default=None, help="Human-friendly model label")
    parser.add_argument("--family", default="qwen", help="Model family")
    parser.add_argument("--model-type", default="dense", help="Model type")
    parser.add_argument("--repo-id", default=None, help="Original upstream repo id if known")
    parser.add_argument("--source-origin", default="local", help="Origin label for the model source")
    parser.add_argument("--format-name", default="unknown", help="Expected source format name")
    parser.add_argument("--download", action="store_true", help="Download --repo-id into source_path before import")
    args = parser.parse_args()

    if args.download:
        if not args.repo_id:
            parser.error("--download requires --repo-id")
        download_huggingface_snapshot(args.repo_id, args.source_path, model_id=args.model_id)

    record = import_model(build_import_request(args))
    print(json.dumps(record.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
