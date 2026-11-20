from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


_ROOT = Path(__file__).resolve().parent
# A wheel cannot hold symlinks, so the extra names below are copies. A copy
# loads as a second image, but every FakeGPU library resolves the process's
# first fakegpu_process_global_state_v1 provider, so state stays shared.
if sys.platform == "darwin":
    _REQUIRED_LIBS = (
        "libnccl.dylib",
        "libcublas.dylib",
        "libcudart.dylib",
        "libcuda.dylib",
        "libnvidia-ml.dylib",
        # nvidia-ml-py loads CDLL("libnvidia-ml.so.1") on every non-Windows
        # platform.
        "libnvidia-ml.so.1",
    )
else:
    _REQUIRED_LIBS = (
        "libnccl.so.2",
        "libcublas.so.12",
        "libcudart.so.12",
        "libcuda.so.1",
        "libnvidia-ml.so.1",
        # nvitop loads CDLL("libcudart.so") on Linux.
        "libcudart.so",
    )

_REQUIRED_CMAKE_TARGETS = (
    "fake_gpu",
    "fake_cuda",
    "fake_cudart",
    "fake_cublas",
    "fake_nccl",
    "fakegpu_coordinator",
)

_REQUIRED_BINARIES = ("fakegpu-coordinator",)


def _cmake_build(build_dir: Path) -> None:
    cfg = os.environ.get("CMAKE_BUILD_TYPE", "Release")
    subprocess.check_call(
        ["cmake", "-S", str(_ROOT), "-B", str(build_dir), f"-DCMAKE_BUILD_TYPE={cfg}"],
    )
    # Packaging needs the runtime libraries and coordinator executable copied
    # into fakegpu/_native.
    # Avoid building probes/tests here so wheel builds stay isolated from optional targets.
    subprocess.check_call(
        ["cmake", "--build", str(build_dir), "--config", cfg, "--target", *_REQUIRED_CMAKE_TARGETS],
    )


def _copy_native_artifacts(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for artifact in (*_REQUIRED_LIBS, *_REQUIRED_BINARIES):
        src = src_dir / artifact
        if not src.exists():
            raise FileNotFoundError(f"Missing built native artifact: {src}")
        shutil.copy2(src.resolve(), dst_dir / artifact)


def _copy_profile_catalog(dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    profile_root = _ROOT / "profiles"
    for profile in sorted(profile_root.rglob("*.yaml")):
        destination = dst_dir / profile.relative_to(profile_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile, destination)


class build_py(_build_py):
    def run(self) -> None:
        if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
            raise RuntimeError("fakegpu currently supports Linux and macOS only")

        build_base = Path(self.get_finalized_command("build").build_base)

        prebuilt_dir = os.environ.get("FAKEGPU_BUILD_DIR")
        if prebuilt_dir and Path(prebuilt_dir).is_dir():
            native_out = Path(prebuilt_dir)
        else:
            native_out = build_base / "fakegpu_native"
            _cmake_build(native_out)

        super().run()

        package_root = Path(self.build_lib) / "fakegpu"
        for cache_dir in list(package_root.rglob("__pycache__")):
            shutil.rmtree(cache_dir)

        package_native = Path(self.build_lib) / "fakegpu" / "_native"
        _copy_native_artifacts(native_out, package_native)
        package_profiles = Path(self.build_lib) / "fakegpu" / "_profiles"
        _copy_profile_catalog(package_profiles)


class sdist(_sdist):
    def make_archive(self, *args, **kwargs):
        archive = super().make_archive(*args, **kwargs)
        if archive.endswith(".tar.gz"):
            # Tar headers otherwise expose the local builder's account name
            # and numeric user/group IDs even when file contents are clean.
            path = Path(archive)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tar.gz") as temp:
                with tarfile.open(path, "r:gz") as source, tarfile.open(temp.name, "w:gz") as target:
                    for entry in source:
                        entry.uid = entry.gid = 0
                        entry.uname = entry.gname = ""
                        if entry.isfile():
                            with source.extractfile(entry) as stream:
                                target.addfile(entry, stream)
                        else:
                            target.addfile(entry)
                shutil.copyfile(temp.name, path)
        return archive


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        # We ship ELF .so files as package data; the wheel is not "pure".
        # Force installation into platlib so wheels containing ELF .so files are
        # platlib-compliant (required by auditwheel).
        return True


class bdist_wheel(_bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        _python, _abi, plat = super().get_tag()
        # The Python wrapper is pure; only the platform tag matters.
        return ("py3", "none", plat)


cmdclass = {"build_py": build_py, "bdist_wheel": bdist_wheel, "sdist": sdist}


setup(cmdclass=cmdclass, distclass=BinaryDistribution)
