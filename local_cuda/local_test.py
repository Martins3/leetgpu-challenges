#!/usr/bin/env python3
"""
Local test runner for LeetGPU CUDA challenges.

Compiles a CUDA solution into a shared library and runs it against the
challenge's own reference implementation and test cases. No WebSocket /
online platform required.

Usage:
    cd /path/to/leetgpu-challenges
    source .venv/bin/activate
    python local_cuda/local_test.py challenges/easy/1_vector_add/local.cu
    python local_cuda/local_test.py challenges/easy/1_vector_add/local.cu --functional
"""

import argparse
import ctypes
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
CHALLENGES_CORE = PROJECT_ROOT / "challenges" / "core"
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
NVCC = CUDA_HOME / "bin" / "nvcc"


def maybe_reexec_with_project_python() -> None:
    """Prefer the repository's virtualenv interpreter when available."""
    if os.environ.get("LOCAL_CUDA_USING_PROJECT_VENV") == "1":
        return

    if not PROJECT_VENV_PYTHON.exists():
        return

    current_python = Path(sys.executable).absolute()
    project_python = PROJECT_VENV_PYTHON.absolute()
    if current_python == project_python:
        return

    env = os.environ.copy()
    env["LOCAL_CUDA_USING_PROJECT_VENV"] = "1"
    os.execvpe(str(project_python), [str(project_python), __file__, *sys.argv[1:]], env)


def locate_libstdcpp_path() -> Path | None:
    """Best-effort lookup for libstdc++.so.6."""
    for candidate in [
        Path("/lib64/libstdc++.so.6"),
        Path("/usr/lib64/libstdc++.so.6"),
        Path("/lib/x86_64-linux-gnu/libstdc++.so.6"),
        Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6"),
    ]:
        if candidate.exists():
            return candidate

    result = subprocess.run(
        ["gcc", "-print-file-name=libstdc++.so.6"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    candidate = Path(result.stdout.strip())
    if candidate.exists():
        return candidate
    return None


def locate_system_libcuda() -> Path | None:
    """Best-effort lookup for the real NVIDIA driver library."""
    for candidate in [
        Path("/lib64/libcuda.so.1"),
        Path("/usr/lib64/libcuda.so.1"),
        Path("/run/opengl-driver/lib/libcuda.so.1"),
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
    ]:
        if candidate.exists():
            return candidate
    return None


maybe_reexec_with_project_python()

libcuda_path = os.environ.get("LOCAL_CUDA_LIBCUDA_PATH")
if libcuda_path is None:
    found = locate_system_libcuda()
    libcuda_path = str(found) if found is not None else None
if libcuda_path is not None:
    try:
        ctypes.CDLL(str(libcuda_path), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

# We need torch for reference_impl and tensor handling
try:
    import torch
except ImportError as exc:
    print("[ERROR] Failed to import PyTorch:")
    print(f"  {exc}")
    print("Install it first, for example:")
    print("  uv pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print("If torch is already installed, check your runtime libraries and active Python environment.")
    raise SystemExit(1) from exc

def fail(message: str) -> "NoReturn":
    print(f"[ERROR] {message}")
    raise SystemExit(1)


def load_challenge(challenge_dir: Path):
    """Dynamically import challenge.py from the given directory."""
    challenge_py = challenge_dir / "challenge.py"
    if not challenge_py.exists():
        fail(f"No challenge.py found in {challenge_dir}")

    # Ensure challenges/core is on the path so "from core.challenge_base import ..." works
    if str(CHALLENGES_CORE.parent) not in sys.path:
        sys.path.insert(0, str(CHALLENGES_CORE.parent))

    spec = importlib.util.spec_from_file_location("challenge", challenge_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    challenge = module.Challenge()
    return challenge


def validate_challenge_dir(challenge_dir: Path) -> Path:
    """Ensure the supplied path looks like a LeetGPU challenge directory."""
    challenge_dir = challenge_dir.resolve()
    challenge_py = challenge_dir / "challenge.py"
    if not challenge_py.exists():
        fail(
            f"{challenge_dir} is not a challenge directory. "
            "Expected a challenge.py file there."
        )
    return challenge_dir


def infer_challenge_dir(cu_path: Path) -> Path:
    """Walk upward from a .cu file until we find its challenge directory."""
    for parent in [cu_path.parent, *cu_path.parents]:
        challenge_py = parent / "challenge.py"
        if challenge_py.exists():
            return parent
        if parent == PROJECT_ROOT:
            break
    fail(
        "Could not infer the challenge directory from "
        f"{cu_path}. The .cu file must live inside a challenge directory or "
        "you must pass --challenge-dir."
    )


def discover_cuda_source(challenge_dir: Path) -> Path:
    """Pick the default CUDA source for a challenge directory."""
    starter_cu = challenge_dir / "starter" / "starter.cu"
    if starter_cu.exists():
        return starter_cu

    fail(
        f"No starter CUDA source found in {challenge_dir}. "
        "Pass a .cu file path directly or add starter/starter.cu."
    )


def resolve_paths(target: Path, explicit_challenge_dir: Path | None) -> tuple[Path, Path]:
    """Resolve the CUDA source file and challenge directory for a run."""
    target = target.resolve()
    if not target.exists():
        fail(f"Path does not exist: {target}")

    if target.is_file():
        if target.suffix != ".cu":
            fail(f"Expected a .cu file, got: {target}")
        cu_path = target
        if explicit_challenge_dir is None:
            challenge_dir = infer_challenge_dir(cu_path)
        else:
            challenge_dir = validate_challenge_dir(explicit_challenge_dir)
        return cu_path, challenge_dir

    if not target.is_dir():
        fail(f"Expected a .cu file or challenge directory, got: {target}")

    challenge_dir = validate_challenge_dir(
        explicit_challenge_dir if explicit_challenge_dir is not None else target
    )
    cu_path = discover_cuda_source(target)
    return cu_path, challenge_dir


def detect_cuda_arch() -> str | None:
    """Detect the first visible GPU's CUDA arch for nvcc."""
    env_arch = os.environ.get("LOCAL_CUDA_ARCH")
    if env_arch:
        return env_arch

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)
    if first_line is None or "." not in first_line:
        return None

    major, minor = first_line.split(".", 1)
    if not (major.isdigit() and minor.isdigit()):
        return None
    return f"sm_{major}{minor}"


def compile_solution(cu_path: Path, so_path: Path) -> None:
    """Compile a .cu file into a shared library with nvcc."""
    if not NVCC.exists():
        fail(f"nvcc not found at {NVCC}. Set CUDA_HOME correctly.")

    cmd = [
        str(NVCC),
        "-O2",
        "-std=c++17",
        "-shared",
        "-Xcompiler", "-fPIC",
        "-o", str(so_path),
        str(cu_path),
    ]
    arch = detect_cuda_arch()
    if arch is not None:
        cmd[1:1] = ["-arch", arch]
    print(f"[BUILD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] nvcc compilation failed for {cu_path}:")
        print(result.stderr.rstrip())
        raise SystemExit(1)
    print(f"[BUILD] OK -> {so_path}")


def ctype_to_ctypes_arg(ctype_spec, tensor_val: torch.Tensor):
    """Convert a torch tensor into the ctypes pointer expected by solve."""
    ptr = tensor_val.data_ptr()
    return ctypes.cast(ptr, ctype_spec)


def deep_clone_test_case(test_case: dict) -> dict:
    """Clone a test-case dict so reference_impl can mutate a copy."""
    cloned = {}
    for k, v in test_case.items():
        if isinstance(v, torch.Tensor):
            cloned[k] = v.clone()
        else:
            cloned[k] = v
    return cloned


def run_test(challenge, solution_so: Path, test_case: dict, test_name: str) -> bool:
    """Run one test case through the compiled solution and compare with reference."""
    # 1. Compute reference result on a clone
    ref_case = deep_clone_test_case(test_case)
    challenge.reference_impl(**ref_case)

    # 2. Prepare ctypes call
    lib = ctypes.CDLL(str(solution_so))
    sig = challenge.get_solve_signature()

    # Build argtypes list in declaration order
    argtypes = []
    argvals = []
    for param_name, (ctype_spec, direction) in sig.items():
        argtypes.append(ctype_spec)
        val = test_case[param_name]
        if isinstance(val, torch.Tensor):
            argvals.append(ctype_to_ctypes_arg(ctype_spec, val))
        elif ctype_spec == ctypes.c_float:
            argvals.append(ctypes.c_float(val))
        elif ctype_spec == ctypes.c_int:
            argvals.append(ctypes.c_int(val))
        elif ctype_spec == ctypes.c_size_t:
            argvals.append(ctypes.c_size_t(val))
        else:
            raise ValueError(f"Unsupported ctype for parameter {param_name}: {ctype_spec}")

    lib.solve.argtypes = argtypes
    lib.solve.restype = None

    # 3. Call the user's solve (mutates the original test_case tensors in-place)
    lib.solve(*argvals)
    torch.cuda.synchronize()

    # 4. Compare outputs (any param marked "out" or "inout")
    ok = True
    for param_name, (ctype_spec, direction) in sig.items():
        if direction not in ("out", "inout"):
            continue
        actual = test_case[param_name]
        expected = ref_case[param_name]
        if not torch.allclose(actual, expected, atol=challenge.atol, rtol=challenge.rtol):
            ok = False
            diff = (actual - expected).abs()
            max_diff = diff.max().item()
            max_idx = diff.argmax().item()
            print(f"  [FAIL] {param_name}: mismatch at flat index {max_idx}, max_diff={max_diff:.6e}")
            # Print a few values around the mismatch for debugging
            flat_actual = actual.flatten()
            flat_expected = expected.flatten()
            start = max(0, max_idx - 2)
            end = min(len(flat_actual), max_idx + 3)
            for i in range(start, end):
                mark = " <-" if i == max_idx else ""
                print(f"    [{i}] got={flat_actual[i].item():.6f} expected={flat_expected[i].item():.6f}{mark}")
        else:
            print(f"  [OK]   {param_name}")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Local runner for LeetGPU CUDA challenges")
    parser.add_argument(
        "target",
        type=Path,
        help="Path to a CUDA source file, or a challenge directory to run starter/starter.cu",
    )
    parser.add_argument(
        "--challenge-dir",
        type=Path,
        default=None,
        help="Path to the challenge directory when the .cu file lives elsewhere",
    )
    parser.add_argument(
        "--functional",
        action="store_true",
        help="Run all functional tests instead of just the example test",
    )
    args = parser.parse_args()

    cu_path, challenge_dir = resolve_paths(args.target, args.challenge_dir)
    challenge = load_challenge(challenge_dir)

    print(f"[INFO] Challenge : {challenge.name}")
    print(f"[INFO] Directory : {challenge_dir}")
    print(f"[INFO] Source    : {cu_path}")

    # Compile to shared library in a temp directory
    with tempfile.TemporaryDirectory(prefix="leetgpu_local_") as tmpdir:
        so_path = Path(tmpdir) / "solution.so"
        compile_solution(cu_path, so_path)

        # Gather tests
        if args.functional:
            test_cases = challenge.generate_functional_test()
            print(f"[INFO] Running {len(test_cases)} functional test(s)...\n")
        else:
            test_cases = [challenge.generate_example_test()]
            print(f"[INFO] Running example test...\n")

        passed = 0
        failed = 0
        for idx, test_case in enumerate(test_cases):
            test_name = f"test_{idx}"
            if args.functional and "name" in test_case:
                test_name = test_case["name"]
            print(f"[{test_name}]")
            if run_test(challenge, so_path, test_case, test_name):
                passed += 1
            else:
                failed += 1

    print(f"\n{'='*40}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
