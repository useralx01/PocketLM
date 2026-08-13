"""Build pcketlm native helper DLLs with the local Windows C++ toolchain."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = PROJECT_ROOT / "src" / "pcketlm" / "native"
OPENBLAS_DIR = PROJECT_ROOT / "vendor" / "openblas"
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
    include_flags = ""
    link_flags = ""
    post_copy = ""
    if source.name == "fp16_matmul.cpp":
        openblas_include = OPENBLAS_DIR / "include"
        openblas_lib = OPENBLAS_DIR / "lib" / "libopenblas.lib"
        openblas_dll = OPENBLAS_DIR / "bin" / "libopenblas.dll"
        if not openblas_include.exists() or not openblas_lib.exists() or not openblas_dll.exists():
            raise FileNotFoundError(
                "OpenBLAS vendor files are missing. Expected "
                f"{openblas_include}, {openblas_lib}, and {openblas_dll}."
            )
        include_flags = f' /I"{openblas_include}"'
        link_flags = f' /link "{openblas_lib}"'
        post_copy = f'copy /Y "{openblas_dll}" "{NATIVE_DIR / "libopenblas.dll"}" >nul\n'
    arch_flag = "/arch:AVX512" if source.name == "fp8_linear_avx512.cpp" else "/arch:AVX2"
    batch = (
        "@echo off\n"
        f'call "{vcvars}" >nul\n'
        f'cl.exe /nologo /O2 /EHsc /std:c++17 {arch_flag} /openmp{include_flags} /LD "{source}" /Fe:"{dll}"{link_flags}\n'
        f"{post_copy}"
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
