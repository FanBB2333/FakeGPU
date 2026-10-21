"""``fakegpu run``: unmodified commands on simulated GPUs, with a labeled summary."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import textwrap

import pytest

from fakegpu import run_cli
from fakegpu._run_bootstrap import visible_gpus
from fakegpu._run_summary import build_report, render
from fakegpu.machine.layout import build_layout_state
from fakegpu.profile_catalog import get_profile

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
needs_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")


def _environment(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env["FAKEGPU_HOME"] = str(tmp_path / "home")
    for name in ("FAKEGPU_PROFILE", "FAKEGPU_PROFILES", "FAKEGPU_DEVICE_COUNT", "FAKEGPU_RUN_DIR"):
        env.pop(name, None)
    return env


def _run(tmp_path: Path, *args: str, timeout: float = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fakegpu", "run", *args],
        cwd=tmp_path,
        env=_environment(tmp_path),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _script(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("a100x2", ["a100", "a100"]),
        ("2xa100", ["a100", "a100"]),
        ("A100:2", ["a100", "a100"]),
        ("h100:1, a100:2", ["h100", "a100", "a100"]),
        ("a100x", ["a100x"]),
        ("rtx-a4000x2", ["rtx-a4000", "rtx-a4000"]),
        ("2xrtx4090", ["rtx4090", "rtx4090"]),  # not GPU '2xrt' x 4090
        ("t4", ["t4"]),
    ],
)
def test_gpu_spec_accepts_the_usual_spellings(spec: str, expected: list[str]) -> None:
    assert run_cli.parse_gpu_spec(spec) == expected


def test_gpu_spec_suggests_profiles_for_a_typo() -> None:
    with pytest.raises(run_cli.GpuSpecError, match=r"unknown GPU 'a100-80g'.*Did you mean: .*a100"):
        run_cli.parse_gpu_spec("a100-80g")
    with pytest.raises(run_cli.GpuSpecError, match="at least 1"):
        run_cli.parse_gpu_spec("a100:0")


def test_layout_without_gpu_option_follows_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKEGPU_HOME", str(tmp_path))
    for name in ("FAKEGPU_PROFILE", "FAKEGPU_PROFILES", "FAKEGPU_DEVICE_COUNT"):
        monkeypatch.delenv(name, raising=False)
    assert run_cli._resolve_layout(None) == (["a100"] * 8, False)
    monkeypatch.setenv("FAKEGPU_PROFILE", "h100")
    monkeypatch.setenv("FAKEGPU_DEVICE_COUNT", "2")
    assert run_cli._resolve_layout(None) == (["h100", "h100"], False)
    monkeypatch.setenv("FAKEGPU_PROFILES", "t4,l4:2")
    assert run_cli._resolve_layout(None) == (["t4", "l4", "l4"], False)
    assert run_cli._resolve_layout("a100x1") == (["a100"], False)


def _record(index: int, peak: int, *, total: int = 80 * GIB, profile: str = "a100", **extra) -> dict:
    device = {
        "index": index, "profile_id": profile, "name": "NVIDIA A100-SXM4-80GB", "total_memory": total,
        "peak_memory": peak, "peak_reserved_memory": peak, "allocation_count": 3, "oom_events": 0,
        "last_oom": None, "tracking_confidence": "C3_torch_dispatch_lifetime",
    }
    device.update(extra.pop("device", {}))
    return {
        "pid": 100 + index, "argv": ["train.py"], "snapshot": {
            "tracking_confidence": device["tracking_confidence"], "devices": [device],
        },
        **extra,
    }


LAYOUT = [
    {"index": 0, "profile_id": "a100", "name": "NVIDIA A100-SXM4-80GB", "total_memory": 80 * GIB},
    {"index": 1, "profile_id": "a100", "name": "NVIDIA A100-SXM4-80GB", "total_memory": 80 * GIB},
]


def test_summary_adds_processes_on_one_gpu_and_labels_every_number() -> None:
    report = build_report(
        command=["torchrun", "train.py"], exit_code=0, duration_seconds=1.0, layout=LAYOUT,
        records=[
            _record(0, 30 * GIB),
            _record(0, 20 * GIB, device={"tracking_confidence": "C2_torch_tensor_lifetime"}),
        ],
    )
    gpu = report["devices"][0]
    assert (gpu["peak_memory"], gpu["process_count"]) == (50 * GIB, 2)
    assert gpu["tracking_confidence"] == "C2_torch_tensor_lifetime"  # the weaker level
    assert report["verdict"] == "fits"
    text = render(report)
    assert "Fits. Largest peak: 50.00 GiB on GPU 0, 62.5% of its capacity." in text.replace("\n           ", " ")
    assert "[C2]" in text and "[catalog]" in text and "Excludes" in text
    assert all(len(line) <= 72 for line in text.splitlines())


def test_summary_reports_a_simulated_out_of_memory_error() -> None:
    oom = {"requested_bytes": 20 * GIB, "allocated_bytes": 0, "reserved_bytes": 2 * GIB}
    report = build_report(
        command=["python", "train.py"], exit_code=1, duration_seconds=1.0, layout=LAYOUT[:1],
        records=[_record(0, 2 * GIB, total=16 * GIB, device={"oom_events": 1, "last_oom": oom},
                         exception={"type": "OutOfMemoryError", "message": "CUDA out of memory."})],
    )
    assert report["verdict"] == "does_not_fit"
    assert report["verdict_text"] == (
        "Does not fit. GPU 0 ran out of memory: tried to allocate 20.00 GiB with 2.00 GiB reserved of 16.00 GiB."
    )


def test_summary_without_gpu_use_failures_or_with_caught_ooms() -> None:
    idle = build_report(command=["python", "x.py"], exit_code=0, duration_seconds=0.1, layout=LAYOUT, records=[])
    assert idle["verdict"] == "no_gpu_use" and "-I, -E or -S" in idle["verdict_text"]
    isolated = build_report(command=["python", "-I", "x.py"], exit_code=1, duration_seconds=0.1, layout=LAYOUT,
                            records=[])
    assert isolated["verdict"] == "unresolved" and "-I, -E or -S" in isolated["verdict_text"]
    failed = build_report(
        command=["python", "x.py"], exit_code=3, duration_seconds=0.1, layout=LAYOUT,
        records=[_record(0, GIB, exception={"type": "ValueError", "message": "bad batch"})],
    )
    assert failed["verdict"] == "unresolved"
    assert "failed (exit 3)" in failed["verdict_text"] and "ValueError: bad batch" in failed["verdict_text"]
    caught = build_report(
        command=["python", "x.py"], exit_code=0, duration_seconds=0.1, layout=LAYOUT,
        records=[_record(0, GIB, device={"oom_events": 2})],
    )
    assert caught["verdict"] == "fits" and "caught 2 simulated out-of-memory" in caught["verdict_text"]


def test_summary_shows_the_gpus_a_script_chose_itself() -> None:
    report = build_report(
        command=["python", "x.py"], exit_code=0, duration_seconds=0.1, layout=LAYOUT,
        records=[_record(0, GIB, total=16 * GIB, profile="t4")],
    )
    assert report["devices"][0]["profile_id"] == "t4"
    assert report["devices"][0]["total_memory"] == 16 * GIB
    assert report["notes"]


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (None, None),
        ("0,1,2", None),
        ("2,0", (["h100", "a100"], [2, 0])),
        ("1,7,2", (["a100"], [1])),  # CUDA stops at the first GPU it does not have
        ("-1", ([], [])),
        ("", ([], [])),
    ],
)
def test_cuda_visible_devices_picks_gpus_of_the_run(mask: str | None, expected: tuple | None) -> None:
    assert visible_gpus("a100,a100,h100", mask) == expected


def test_layout_state_lists_idle_gpus_for_viewers(tmp_path: Path) -> None:
    from fakegpu._smi_state import LAYOUT_ONLY_KEY, build_inventory

    state = build_layout_state(["a100", "t4"], {}, pid=os.getpid(), process_name="fakegpu run")
    assert state[LAYOUT_ONLY_KEY] is True
    inventory = build_inventory([state])
    assert [device["profile_id"] for device in inventory["devices"]] == ["a100", "t4"]
    assert inventory["devices"][1]["memory"]["total_bytes"] == get_profile("t4").memory_bytes


def test_bootstrap_runs_the_sitecustomize_it_shadows(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text(
        "import os\nos.environ['SHADOWED_RAN'] = str(int(os.environ.get('SHADOWED_RAN', '0')) + 1)\n",
        encoding="utf-8",
    )
    env = _environment(tmp_path)
    bootstrap = ROOT / "fakegpu" / "_bootstrap"
    # A second FakeGPU install's bootstrap, as in a nested run, must not run this one again.
    second = tmp_path / "second" / "fakegpu" / "_bootstrap"
    second.mkdir(parents=True)
    shutil.copy(bootstrap / "sitecustomize.py", second)
    env["PYTHONPATH"] = os.pathsep.join([str(bootstrap), str(second), str(other), env["PYTHONPATH"]])
    env["FAKEGPU_RUN_DIR"] = str(tmp_path / "run")
    completed = subprocess.run(
        [sys.executable, "-c", "import os, sys; print(os.environ.get('SHADOWED_RAN'), 'torch' in sys.modules)"],
        env=env, text=True, capture_output=True, timeout=60,
    )
    assert completed.stdout.split() == ["1", "False"], completed.stderr


@needs_torch
def test_run_streams_output_and_prints_a_labeled_verdict(tmp_path: Path) -> None:
    script = _script(tmp_path, "train.py", """
        import shutil, subprocess, torch
        listed = subprocess.run(["nvidia-smi", "--query-gpu=count,index,name", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip().splitlines()
        print("before", listed)
        model = torch.nn.Linear(256, 256).cuda()
        model(torch.randn(8, 256, device="cuda")).sum().backward()
        print("cuda", torch.cuda.device_count(), torch.cuda.get_device_name(1))
        print("smi", shutil.which("nvidia-smi") is not None)
    """)
    completed = _run(tmp_path, "--gpu", "a100x2", "--report-dir", str(tmp_path / "report"), "--",
                     sys.executable, script)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == (
        "before ['2, 0, NVIDIA A100-SXM4-80GB', '2, 1, NVIDIA A100-SXM4-80GB']"
    )
    assert lines[1:] == ["cuda 2 NVIDIA A100-SXM4-80GB", "smi True"]
    assert "Verdict   Fits." in completed.stderr
    assert "[C3]" in completed.stderr and "[catalog]" in completed.stderr
    report = json.loads((tmp_path / "report" / "run_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "fits"
    assert report["devices"][0]["process_count"] == 1
    assert report["tracking_confidence"] == "C3_torch_dispatch_lifetime"


@needs_torch
def test_run_keeps_the_exit_status_and_reports_out_of_memory(tmp_path: Path) -> None:
    script = _script(tmp_path, "big.py", """
        import torch
        torch.empty(20 * 2**30, dtype=torch.uint8, device="cuda")
    """)
    completed = _run(tmp_path, "--gpu", "t4", "--", sys.executable, script)
    assert completed.returncode == 1
    assert "OutOfMemoryError" in completed.stderr
    assert "Does not fit. GPU 0 ran out of memory: tried to allocate 20.00 GiB" in " ".join(completed.stderr.split())

    exits = _script(tmp_path, "exits.py", "import sys, torch\ntorch.ones(1, device='cuda')\nsys.exit(3)\n")
    completed = _run(tmp_path, "--gpu", "t4", "--no-summary", "--", sys.executable, exits)
    assert completed.returncode == 3
    assert "FakeGPU run summary" not in completed.stderr


@needs_torch
def test_run_rejects_unknown_gpus_and_missing_commands(tmp_path: Path) -> None:
    completed = _run(tmp_path, "--gpu", "a100-80g", "--", "true")
    assert completed.returncode == 2 and "Did you mean" in completed.stderr
    completed = _run(tmp_path, "--gpu", "t4", "--", "no-such-command-for-fakegpu")
    assert completed.returncode == 127 and "command not found" in completed.stderr


@needs_torch
def test_run_joins_torchrun_workers_with_real_collectives(tmp_path: Path) -> None:
    script = _script(tmp_path, "ddp.py", """
        import os, torch, torch.distributed as dist
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        assert (rank, world) == (int(os.environ["RANK"]), 2)
        torch.cuda.set_device(rank)
        values = torch.full((2,), float(rank + 1), device="cuda")
        dist.all_reduce(values)
        mean = torch.full((2,), float(rank + 1), device="cuda")
        dist.all_reduce(mean, op=dist.ReduceOp.AVG)
        gathered = torch.empty(4, device="cuda")
        dist.all_gather_into_tensor(gathered, torch.full((2,), float(rank), device="cuda"))
        part = torch.empty(2, device="cuda")
        dist.reduce_scatter_tensor(part, torch.arange(4, dtype=torch.float32, device="cuda") * (rank + 1))
        objects = [None, None]
        dist.all_gather_object(objects, rank)
        torch.manual_seed(rank)  # different weights until DDP broadcasts rank 0's
        model = torch.nn.Linear(4, 1).cuda()
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        ddp(torch.full((1, 4), float(rank + 1), device="cuda")).sum().backward()
        weights = [torch.empty_like(model.weight) for _ in range(2)]
        dist.all_gather(weights, model.weight.detach())

        class Branches(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(2, 1, bias=False)
                self.b = torch.nn.Linear(2, 1, bias=False)

            def forward(self, x, use_a):
                return (self.a if use_a else self.b)(x)

        # Rank 0 trains only a, rank 1 only b: each gradient averages with zeros.
        branches = torch.nn.parallel.DistributedDataParallel(
            Branches().cuda(), device_ids=[rank], find_unused_parameters=True)
        branches(torch.full((1, 2), float(rank + 1), device="cuda"), rank == 0).sum().backward()
        ints = torch.tensor([3 * rank + 1], device="cuda")
        dist.all_reduce(ints, op=dist.ReduceOp.AVG)
        reduced = torch.full((1,), float(rank + 1), device="cuda")
        dist.reduce(reduced, group_dst=1)
        fields = [values.tolist(), mean.tolist(), gathered.tolist(), part.tolist(), objects,
                  torch.equal(weights[0], weights[1]), model.weight.grad.tolist(),
                  branches.module.a.weight.grad.tolist(), branches.module.b.weight.grad.tolist(),
                  ints.tolist(), reduced.tolist() if rank == 1 else None]
        # One write per rank: torchrun runs workers unbuffered on a shared pipe.
        os.write(1, ("rank " + " ".join(map(str, [rank, *fields])) + "\\n").encode())
        dist.destroy_process_group()
    """)
    completed = _run(
        tmp_path, "--gpu", "a100x2", "--", sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", "2", "--master_addr", "127.0.0.1", "--master_port", str(_free_port()), script,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = sorted(line for line in completed.stdout.splitlines() if line.startswith("rank "))
    assert lines == [
        "rank 0 [3.0, 3.0] [1.5, 1.5] [0.0, 0.0, 1.0, 1.0] [0.0, 3.0] [0, 1] True [[1.5, 1.5, 1.5, 1.5]]"
        " [[0.5, 0.5]] [[1.0, 1.0]] [2] None",
        "rank 1 [3.0, 3.0] [1.5, 1.5] [0.0, 0.0, 1.0, 1.0] [6.0, 9.0] [0, 1] True [[1.5, 1.5, 1.5, 1.5]]"
        " [[0.5, 0.5]] [[1.0, 1.0]] [2] [3.0]",
    ]
    assert "2 processes used simulated GPUs" in completed.stderr


@needs_torch
def test_run_keeps_the_memory_of_ranks_torchrun_stops(tmp_path: Path) -> None:
    script = _script(tmp_path, "fails.py", """
        import time, torch, torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        held = torch.empty(2**30, dtype=torch.uint8, device="cuda")
        dist.barrier()
        if rank == 1:
            raise ValueError("rank 1 failed")
        time.sleep(120)  # torchrun stops this rank with SIGTERM
    """)
    completed = _run(
        tmp_path, "--gpu", "a100x2", "--report-dir", str(tmp_path / "report"), "--", sys.executable, "-m",
        "torch.distributed.run", "--nproc_per_node", "2", "--master_addr", "127.0.0.1",
        "--master_port", str(_free_port()), script,
    )
    assert completed.returncode != 0
    report = json.loads((tmp_path / "report" / "run_report.json").read_text(encoding="utf-8"))
    assert [(gpu["process_count"], gpu["peak_memory"] >= GIB) for gpu in report["devices"]] == [(1, True), (1, True)]
    assert report["verdict"] == "unresolved"
    assert "ValueError: rank 1 failed" in report["verdict_text"]


@needs_torch
def test_run_limits_a_process_to_its_cuda_visible_devices(tmp_path: Path) -> None:
    script = _script(tmp_path, "pinned.py", """
        import subprocess, time, torch
        held = torch.empty(2**30, dtype=torch.uint8, device="cuda")
        time.sleep(1)  # let the process publish its memory
        query = ["nvidia-smi", "-i", "1", "--query-gpu=count,index,memory.used", "--format=csv,noheader,nounits"]
        print("visible", torch.cuda.device_count(), subprocess.run(query, capture_output=True, text=True).stdout)
    """)
    completed = _run(tmp_path, "--gpu", "a100x2", "--report-dir", str(tmp_path / "report"), "--",
                     "env", "CUDA_VISIBLE_DEVICES=1", sys.executable, script)
    assert completed.returncode == 0, completed.stderr
    count, smi = completed.stdout.split(maxsplit=2)[1:]
    assert count == "1"
    total, index, used = (int(value) for value in smi.split(","))
    assert (total, index) == (2, 1) and used >= 1024
    report = json.loads((tmp_path / "report" / "run_report.json").read_text(encoding="utf-8"))
    assert [gpu["process_count"] for gpu in report["devices"]] == [0, 1]
    assert report["notes"] == []


@needs_torch
def test_nested_run_reports_to_the_outer_run(tmp_path: Path) -> None:
    script = _script(tmp_path, "alloc.py", "import torch\ntorch.ones(1024, device='cuda')\n")
    completed = _run(tmp_path, "--gpu", "t4", "--", sys.executable, "-m", "fakegpu", "run", "--gpu", "t4",
                     "--no-summary", "--", sys.executable, script)
    assert completed.returncode == 0, completed.stderr
    assert "Verdict   Fits." in completed.stderr


@needs_torch
def test_accelerate_launch_sees_every_process(tmp_path: Path) -> None:
    pytest.importorskip("accelerate")
    script = _script(tmp_path, "accel.py", """
        import os, torch
        from accelerate import Accelerator
        accelerator = Accelerator()
        model = accelerator.prepare(torch.nn.Linear(8, 8))
        model(torch.randn(2, 8, device=accelerator.device)).sum().backward()
        line = f"process {accelerator.process_index} {accelerator.num_processes} {accelerator.device}"
        os.write(1, (line + "\\n").encode())
    """)
    completed = _run(
        tmp_path, "--gpu", "a100x2", "--", sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
        "--num_processes", "2", "--num_machines", "1", "--mixed_precision", "no", "--dynamo_backend", "no",
        "--main_process_ip", "127.0.0.1", "--main_process_port", str(_free_port()), script,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = sorted(line for line in completed.stdout.splitlines() if line.startswith("process "))
    assert lines == ["process 0 2 cuda:0", "process 1 2 cuda:1"]


@needs_torch
def test_only_launched_ranks_wait_for_their_peers(monkeypatch: pytest.MonkeyPatch) -> None:
    from fakegpu import _upstream

    for name in ("TORCHELASTIC_RUN_ID", "FAKEGPU_DIST_COLLECTIVES"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {"WORLD_SIZE": "8", "RANK": "0", "MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "29500"}.items():
        monkeypatch.setenv(name, value)
    # A cluster pod's environment alone: no peers will start, so simulate.
    assert _upstream._real_launch(-1, -1, None, None) is None
    assert _upstream._real_launch(-1, -1, "env://", None) is None
    assert _upstream._real_launch(2, 1, "tcp://127.0.0.1:29501", None) == (2, 1)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run")
    assert _upstream._real_launch(-1, -1, None, None) == (8, 0)
    monkeypatch.setenv("FAKEGPU_DIST_COLLECTIVES", "simulate")
    assert _upstream._real_launch(-1, -1, None, None) is None
    monkeypatch.delenv("TORCHELASTIC_RUN_ID")
    monkeypatch.setenv("FAKEGPU_DIST_COLLECTIVES", "real")
    assert _upstream._real_launch(-1, -1, None, None) == (8, 0)
