import os
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    IS_WINDOWS,
    BuildExtension,
    CUDAExtension,
)


def detect_gpu_archs() -> tuple[bool, bool, bool]:
    """
    Query all visible CUDA devices via torch and return three booleans:
      - has_sm100: major version numbers == 10 with minor == 0  (sm100)
      - has_sm103: major version numbers == 10 with minor == 3  (sm103)
      - has_sm90:  major version numbers == 9  with minor == 0  (sm90a)
    Returns (has_sm100, has_sm103, has_sm90)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, False, False
        has_sm100 = False
        has_sm103 = False
        has_sm90 = False
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            major, minor = prop.major, prop.minor
            print(f"  GPU {i}: {prop.name}, compute capability sm_{major}{minor}")
            if major == 10 and minor == 0:
                has_sm100 = True
            if major == 10 and minor == 3:
                has_sm103 = True
            if major == 9 and minor == 0:
                has_sm90 = True
        return has_sm100, has_sm103, has_sm90
    except Exception as e:
        print(f"Warning: failed to detect GPU architectures via torch: {e}")
        return False, False, False


def resolve_disable_flag(env_name: str, detected: bool) -> bool:
    """
    Resolve whether to disable a given SM target.
    - If the environment variable is explicitly set, honour it.
    - Otherwise, disable the target when no matching GPU is detected.
    """
    env_val = os.getenv(env_name)
    if env_val is not None:
        return env_val.lower() in ["true", "1", "y", "yes"]
    # Auto-detect: disable if no matching device found
    disable = not detected
    if disable:
        print(f"  No matching GPU detected; auto-setting {env_name}=1 (disable). Set {env_name}=0 to override.")
    return disable


def get_features_args():
    features_args = []
    return features_args


USE_FAST_MATH = os.getenv("CULA_USE_FAST_MATH", "1") == "1"

print("Detecting GPU architectures...")
_has_sm100, _has_sm103, _has_sm90 = detect_gpu_archs()
DISABLE_SM100 = resolve_disable_flag("CULA_DISABLE_SM100", _has_sm100)
DISABLE_SM103 = resolve_disable_flag("CULA_DISABLE_SM103", _has_sm103)
DISABLE_SM90 = resolve_disable_flag("CULA_DISABLE_SM90", _has_sm90)


def get_nvcc_version() -> tuple[int, int]:
    """Return the NVCC (major, minor) version tuple, e.g. ``(12, 9)``."""
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    nvcc_version_number = nvcc_version.split("release ")[1].split(",")[0].strip()
    major, minor = map(int, nvcc_version_number.split("."))
    return major, minor


def nvcc_supports_blackwell() -> bool:
    """Return ``True`` if the current NVCC version is >= 12.9."""
    major, minor = get_nvcc_version()
    return major > 12 or (major == 12 and minor >= 9)


def assert_blackwell_build_env() -> None:
    if not nvcc_supports_blackwell():
        assert DISABLE_SM100 and DISABLE_SM103, (
            "sm100/sm103 compilation requires NVCC 12.9 or higher. "
            "Please set CULA_DISABLE_SM100=1 and CULA_DISABLE_SM103=1 to disable them, "
            "or update your environment."
        )


def get_arch_flags():
    major, minor = get_nvcc_version()
    print(f"Compiling using NVCC {major}.{minor}")

    assert_blackwell_build_env()

    arch_flags = []
    if not DISABLE_SM100:
        arch_flags.extend(["-gencode", "arch=compute_100a,code=sm_100a"])
        arch_flags.extend(["-DCULA_SM100_ENABLED"])
    if not DISABLE_SM103:
        arch_flags.extend(["-gencode", "arch=compute_103a,code=sm_103a"])
        arch_flags.extend(["-DCULA_SM103_ENABLED"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
        arch_flags.extend(["-DCULA_SM90A_ENABLED"])
    return arch_flags


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


# =====================================================================
# ccache setup — wrap NVCC so that unchanged .cu files compile instantly
# =====================================================================
this_dir = os.path.dirname(os.path.abspath(__file__))

ccache_path = shutil.which("ccache")
if ccache_path and os.environ.get("CULA_DISABLE_CCACHE", "0") != "1":
    nvcc_real = os.path.join(CUDA_HOME, "bin", "nvcc")
    ccache_wrapper = os.path.join(this_dir, "scripts", "nvcc-ccache")
    os.makedirs(os.path.dirname(ccache_wrapper), exist_ok=True)
    with open(ccache_wrapper, "w") as f:
        f.write(f'#!/bin/bash\nexec "{ccache_path}" "{nvcc_real}" "${{@}}"\n')
    os.chmod(ccache_wrapper, 0o755)

    os.environ.setdefault("PYTORCH_NVCC", ccache_wrapper)
    os.environ.setdefault("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "1")
    os.environ.setdefault("CCACHE_NOHASHDIR", "1")
    print(f"  ccache enabled for NVCC: {ccache_wrapper}")
else:
    print("  ccache not enabled (install ccache or set CULA_DISABLE_CCACHE=0)")


subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"]


# =====================================================================
# Main extension: everything except the standalone backward-intra kernel
# =====================================================================
cuda_sources = [
    "csrc/api/pybind.cu",
]
if not DISABLE_SM100 or not DISABLE_SM103:
    cuda_sources.extend(
        [
            "csrc/api/kda_sm100.cu",
            "csrc/kda/sm100/kda_fwd_sm100.cu",
        ]
    )
if not DISABLE_SM90:
    cuda_sources.extend(
        [
            "csrc/api/kda_sm90.cu",
            "csrc/kda/sm90/kda_fwd_sm90.cu",
            "csrc/kda/sm90/kda_fwd_sm90_safe_gate.cu",
            # NOTE: backward intra is now in its own extension for faster iteration
        ]
    )

common_nvcc_flags = (
    [
        "-O3",
        "-std=c++20",
        "-DNDEBUG",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
        "-diag-suppress=3189",
    ]
    + get_features_args()
    + get_arch_flags()
    + get_nvcc_thread_args()
    + (["--use_fast_math"] if USE_FAST_MATH else [])
)

common_include_dirs = [
    Path(this_dir) / "csrc",
    Path(this_dir) / "csrc" / "kerutils" / "include",
    Path(this_dir) / "csrc" / "cutlass" / "include",
    Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="cula.cudac",
        sources=cuda_sources,
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": common_nvcc_flags,
        },
        include_dirs=common_include_dirs,
    )
)

# =====================================================================
# Standalone extension: SM90 backward intra (fast iteration)
# =====================================================================
if not DISABLE_SM90:
    ext_modules.append(
        CUDAExtension(
            name="cula._kda_bwd_intra_sm90",
            sources=[
                "csrc/api/kda_bwd_sm90_standalone.cu",
                "csrc/kda/sm90/bwd/kda_bwd_intra_sm90.cu",
            ],
            extra_compile_args={
                "cxx": cxx_args + get_features_args(),
                "nvcc": common_nvcc_flags + ["-maxrregcount=152"],
            },
            include_dirs=common_include_dirs,
        )
    )

# =====================================================================
# Standalone extension: SM90 backward dqkg (fast iteration)
# =====================================================================
if not DISABLE_SM90:
    ext_modules.append(
        CUDAExtension(
            name="cula._kda_bwd_dqkg_sm90",
            sources=[
                "csrc/api/kda_bwd_dqkg_sm90.cu",
                "csrc/kda/sm90/bwd/dqkg/chunk_kda_bwd_sm90_dqkg.cu",
            ],
            extra_compile_args={
                "cxx": cxx_args + get_features_args(),
                "nvcc": common_nvcc_flags + ["-maxrregcount=152"],
            },
            include_dirs=common_include_dirs,
        )
    )

setup(
    name="cuda-linear-attention",
    packages=find_packages(include=["cula", "cula.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
