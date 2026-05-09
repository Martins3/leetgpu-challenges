#!/usr/bin/env python3
"""
Generate a compile_commands.json for CUDA files in this repository.

This is mainly for clangd/editor diagnostics. It uses clang++ with CUDA flags
because clangd understands these arguments better than raw nvcc commands.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
CUDA_INCLUDE = CUDA_HOME / "include"
OUTPUT = PROJECT_ROOT / "compile_commands.json"


def detect_cuda_arch() -> str:
    env_arch = os.environ.get("LOCAL_CUDA_ARCH")
    if env_arch:
        return env_arch

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)
        if first_line and "." in first_line:
            major, minor = first_line.split(".", 1)
            if major.isdigit() and minor.isdigit():
                return f"sm_{major}{minor}"

    return "sm_75"


def collect_cuda_files() -> list[Path]:
    excluded = {".git", ".venv", "__pycache__"}
    files = []
    for path in PROJECT_ROOT.rglob("*.cu"):
        if any(part in excluded for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def build_entry(path: Path, compiler: str, arch: str) -> dict:
    return {
        "directory": str(path.parent),
        "file": str(path),
        "arguments": [
            compiler,
            "--cuda-path=" + str(CUDA_HOME),
            "--cuda-gpu-arch=" + arch,
            "-x",
            "cuda",
            "-std=c++17",
            "-Wno-unknown-cuda-version",
            "-I" + str(CUDA_INCLUDE),
            "-I" + str(PROJECT_ROOT),
            "-c",
            str(path),
        ],
    }


def main() -> int:
    compiler = shutil.which("clang++")
    if compiler is None:
        print("[ERROR] clang++ not found in PATH.")
        return 1
    if not CUDA_INCLUDE.exists():
        print(f"[ERROR] CUDA include directory not found: {CUDA_INCLUDE}")
        return 1

    arch = detect_cuda_arch()
    files = collect_cuda_files()
    entries = [build_entry(path, compiler, arch) for path in files]
    OUTPUT.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"[OK] Wrote {OUTPUT} with {len(entries)} CUDA file entries (arch={arch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
