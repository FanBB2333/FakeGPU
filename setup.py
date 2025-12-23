from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:  # pragma: no cover - wheel may be absent in some build contexts
    _bdist_wheel = None


_ROOT = Path(__file__).resolve().parent
_REQUIRED_LIBS = (
    "libcublas.so.12",
    "libcudart.so.12",
    "libcuda.so.1",
    "libnvidia-ml.so.1",
)


def _cmake_build(build_dir: Path) -> None:
    cfg = os.environ.get("CMAKE_BUILD_TYPE", "Release")
    subprocess.check_call(
        ["cmake", "-S", str(_ROOT), "-B", str(build_dir), f"-DCMAKE_BUILD_TYPE={cfg}"],
    )
    subprocess.check_call(["cmake", "--build", str(build_dir), "--config", cfg])


def _copy_native_libs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for lib in _REQUIRED_LIBS:
        src = src_dir / lib
        if not src.exists():
            raise FileNotFoundError(f"Missing built library: {src}")
        shutil.copy2(src.resolve(), dst_dir / lib)


class build_py(_build_py):
    def run(self) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("fakegpu currently supports Linux only")

        build_base = Path(self.get_finalized_command("build").build_base)

        prebuilt_dir = os.environ.get("FAKEGPU_BUILD_DIR")
        if prebuilt_dir and Path(prebuilt_dir).is_dir():
            native_out = Path(prebuilt_dir)
        else:
            native_out = build_base / "fakegpu_native"
            _cmake_build(native_out)

        super().run()

        package_native = Path(self.build_lib) / "fakegpu" / "_native"
        _copy_native_libs(native_out, package_native)


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        # Force installation into platlib so wheels containing ELF .so files are
        # platlib-compliant (required by auditwheel).
        return True


if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self) -> None:
            super().finalize_options()
            # We ship ELF .so files as package data; the wheel is not "pure".
            self.root_is_pure = False

        def get_tag(self) -> tuple[str, str, str]:
            _python, _abi, plat = super().get_tag()
            # The Python wrapper is pure; only the platform tag matters.
            return ("py3", "none", plat)


    cmdclass = {"build_py": build_py, "bdist_wheel": bdist_wheel}
else:
    cmdclass = {"build_py": build_py}


setup(cmdclass=cmdclass, distclass=BinaryDistribution)
