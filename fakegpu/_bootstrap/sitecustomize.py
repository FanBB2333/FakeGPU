"""Start FakeCUDA in every Python process of a ``fakegpu run``.

``fakegpu run`` puts this directory first on ``PYTHONPATH``, so Python imports
this module at startup in the command and in every Python process it starts,
such as torchrun workers. The module then runs the ``sitecustomize`` it
shadows, if the environment has one.
"""

import os
import sys


def _start_fakegpu() -> None:
    if not os.environ.get("FAKEGPU_RUN_DIR"):
        return
    try:
        from fakegpu._run_bootstrap import install
    except ImportError as exc:
        # A Python without fakegpu, e.g. another virtual environment: only
        # worth a word if this process goes on to use torch.
        sys.meta_path.insert(0, _WarnOnTorch(f"{sys.executable} cannot import fakegpu ({exc})"))
        return
    install()


class _WarnOnTorch:
    def __init__(self, reason: str):
        self.reason = reason

    def find_spec(self, fullname, path, target=None):
        if fullname == "torch":
            sys.meta_path.remove(self)
            print(f"fakegpu run: no simulated GPUs in this process: {self.reason}", file=sys.stderr)
        return None


def _run_shadowed_sitecustomize() -> None:
    import importlib.machinery
    import importlib.util

    # Skip this module and any other FakeGPU install's copy of it, e.g. in a
    # nested run: two copies would otherwise run each other forever.
    bootstrap = os.path.join("fakegpu", "_bootstrap", "sitecustomize.py")
    for entry in sys.path:
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [entry or os.curdir])
        if spec is None or spec.loader is None or str(spec.origin).endswith(bootstrap):
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["sitecustomize"] = module
        spec.loader.exec_module(module)
        return


_start_fakegpu()
_run_shadowed_sitecustomize()
