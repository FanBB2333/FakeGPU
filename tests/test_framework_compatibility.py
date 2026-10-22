from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _run(code, *, hide_cuda=True):
    # An empty CUDA_VISIBLE_DEVICES keeps TensorFlow and scikit-learn on the
    # CPU. FakeCUDA honors the mask like CUDA does, so code that simulates
    # GPUs runs without it.
    environment = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    if hide_cuda:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={
            **environment,
            "FAKEGPU_TERMINAL_REPORT": "0",
            "SCIPY_ARRAY_API": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TF_NUM_INTEROP_THREADS": "2",
            "TF_NUM_INTRAOP_THREADS": "2",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("framework,modules", [
    ("tensorflow", ["tensorflow"]),
    ("sklearn-cpu", ["sklearn"]),
    ("sklearn-array-api", ["sklearn", "torch"]),
])
def test_framework_numerical_probe(framework, modules):
    for module in modules:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"optional framework dependency {module} is not installed")
    result = _run(
        "import runpy,sys; "
        f"sys.argv=['framework_compatibility.py','--framework',{framework!r}]; "
        "runpy.run_path('tests/support/framework_compatibility.py',run_name='__main__')",
        hide_cuda=framework != "sklearn-array-api",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "passed"


def test_tensorflow_adapter_import_is_optional():
    result = _run("import sys; import fakegpu.tensorflow; assert 'tensorflow' not in sys.modules")
    assert result.returncode == 0, result.stderr


def test_tensorflow_scope_validation_and_restoration():
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("optional TensorFlow dependency is not installed")
    result = _run("""
import tensorflow as tf
from fakegpu.tensorflow import simulate
original = tf.device
for count in (0, -1, True, 1.5):
    try:
        with simulate(device_count=count):
            raise AssertionError('invalid device count was accepted')
    except ValueError:
        pass
with simulate(device_count=2):
    outer = tf.device
    with simulate(device_count=1):
        with tf.device('/GPU:0'):
            assert 'CPU:0' in tf.constant(1).device
    assert tf.device is outer
    for device in ('/GPU:2', '/GPU:99'):
        try:
            with tf.device(device):
                raise AssertionError('invalid device index was accepted')
        except ValueError:
            pass
    with tf.device('/CPU:0'):
        assert 'CPU:0' in tf.constant(1).device
    with tf.device(None):
        assert 'CPU:0' in tf.constant(1).device
assert tf.device is original
try:
    with simulate():
        raise RuntimeError('leave scope')
except RuntimeError:
    pass
assert tf.device is original
assert not tf.config.list_logical_devices('GPU')
""")
    assert result.returncode == 0, result.stdout + result.stderr


def test_torch_asarray_preserves_device_copy_and_gradient_semantics():
    pytest.importorskip("torch")
    result = _run("""
import fakegpu
fakegpu.init(runtime='fakecuda', device_count=2, profile='a100')
import torch
import numpy as np
array = np.array([1., 2.], dtype=np.float32)
host = torch.asarray(array, copy=False, device='cpu')
host[0] = 3
assert array[0] == 3
gpu = torch.asarray(array, device='cuda:1')
gpu[0] = 5
assert array[0] == 3
try:
    torch.asarray(array, device='cuda:1', copy=False)
except ValueError:
    pass
else:
    raise AssertionError('host to GPU copy=False must fail')
with torch.device('cuda:1'):
    assert torch.asarray([1., 2.]).device == torch.device('cuda:1')
    assert torch.asarray(host).device == torch.device('cuda:1')
with torch.device('meta'):
    assert torch.asarray(host).device == torch.device('meta')
torch.set_default_device('cuda:1')
assert torch.asarray([1., 2.]).device == torch.device('cuda:1')
assert torch.asarray(host).device == torch.device('cuda:1')
torch.set_default_device(None)
source = torch.tensor([1., 2.], device='cuda:1')
shared = torch.asarray(source, copy=False)
assert shared.device == source.device
shared[0] = 3
assert source[0].item() == 3
copy = torch.asarray(source, copy=True)
copy[0] = 7
assert source[0].item() == 3
assert copy.device == source.device
host = torch.asarray(source, device='cpu')
assert host.device.type == 'cpu'
host[0] = 9
assert source[0].item() == 3
peer = torch.asarray(source, device='cuda:0')
assert peer.device == torch.device('cuda:0')
peer[0] = 11
assert source[0].item() == 3
with torch.device('cuda:0'):
    assert torch.asarray(source).device == torch.device('cuda:0')
with torch.device('cpu'):
    assert torch.asarray(source).device.type == 'cpu'
    try:
        torch.asarray(source, copy=False)
    except ValueError:
        pass
    else:
        raise AssertionError('default CPU device must require a copy')
torch.set_default_device('cpu')
assert torch.asarray(source).device.type == 'cpu'
torch.set_default_device(None)
torch.cuda.set_device(1)
assert torch.asarray(source, device='cuda', copy=False).device == source.device
for target in ('cpu', 'cuda:0'):
    try:
        torch.asarray(source, device=target, copy=False)
    except ValueError:
        pass
    else:
        raise AssertionError('cross-device copy=False must fail')
try:
    torch.asarray(source, device='cuda:2')
except RuntimeError as error:
    assert 'invalid device ordinal' in str(error)
else:
    raise AssertionError('invalid device must fail')
leaf = torch.asarray([2., 3.], device='cuda:1', requires_grad=True)
(leaf * leaf).sum().backward()
torch.testing.assert_close(leaf.grad.cpu(), torch.tensor([4., 6.]))
assert not torch.asarray(leaf).requires_grad
""", hide_cuda=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_torch_asarray_records_host_and_peer_transfers():
    pytest.importorskip("torch")
    result = _run("""
import fakegpu
fakegpu.init(runtime='fakecuda', device_count=2, profile='a100')
import torch
from fakegpu.torch_patch import _transfer_snapshot
torch.set_default_device('cuda:0')
first = torch.asarray([1., 2.], dtype=torch.float32)
torch.set_default_device(None)
second = torch.asarray(obj=first, device='cuda:1')
host = torch.asarray(second, device='cpu')
torch.asarray(second, copy=False)
snapshot = _transfer_snapshot()
for index, kind in ((0, 'host_to_device'), (0, 'peer_send'), (1, 'peer_receive'), (1, 'device_to_host')):
    copy = snapshot[index]['copies'][kind]
    assert copy['calls'] == 1
    assert copy['bytes'] == 8
""", hide_cuda=False)
    assert result.returncode == 0, result.stdout + result.stderr
