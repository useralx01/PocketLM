"""Build pcketlm native helper DLLs with the local Windows C++ toolchain."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = PROJECT_ROOT / "src" / "pcketlm" / "native"
VS_ROOT = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC")


def find_vcvars64() -> Path:
    path = VS_ROOT / "Auxiliary" / "Build" / "vcvars64.bat"
    if path.exists():
        return path
    matches = list(VS_ROOT.glob("**/vcvars64.bat"))
    if matches:
        return matches[0]
    raise FileNotFoundError("vcvars64.bat was not found under Visual Studio Build Tools.")


def build_cpp(source: Path, *, force: bool = False) -> Path:
    source = source.resolve()
    if source.suffix.lower() != ".cpp":
        raise ValueError(f"Native source must be a .cpp file: {source}")
    dll = source.with_suffix(".dll")
    if dll.exists() and not force and dll.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return dll
    vcvars = find_vcvars64()
    batch = (
        "@echo off\n"
        f'call "{vcvars}" >nul\n'
        f'cl.exe /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /openmp /LD "{source}" /Fe:"{dll}"\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="utf-8") as handle:
        handle.write(batch)
        batch_path = Path(handle.name)
    try:
        completed = subprocess.run([str(batch_path)], cwd=str(source.parent), text=True, capture_output=True)
    finally:
        batch_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Native build failed for "
            f"{source}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return dll


def build_all(*, force: bool = False) -> list[Path]:
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    return [build_cpp(source, force=force) for source in sorted(NATIVE_DIR.glob("*.cpp"))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pcketlm native helper DLLs.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    built = build_all(force=args.force)
    for path in built:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
