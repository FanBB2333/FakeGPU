"""torch.distributed under FakeCUDA, checked against the installed torch and real gloo runs.

Expected values come from running the same calls without FakeGPU: plain
``torchrun`` with gloo on CPU tensors, torch's own single-process ``fake``
backend, or the installed torch's source where NCCL differs from gloo.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")

# One simulated process that names an 8-rank job, as a researcher sizing a large run does.
SIMULATED = """
import json, os
for name in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "TORCHELASTIC_RUN_ID", "FAKEGPU_DIST_COLLECTIVES"):
    os.environ.pop(name, None)
os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"
import fakegpu
fakegpu.init(runtime="fakecuda", profile="a100", device_count=8)
import torch, torch.distributed as dist
dist.init_process_group("nccl", world_size=8, rank=5)
torch.cuda.set_device(5)
results = {}
def probe(name, fn):
    try:
        results[name] = fn()
    except Exception as exc:
        results[name] = [type(exc).__name__, str(exc).splitlines()[0] if str(exc) else ""]
"""


def _environment(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env["FAKEGPU_HOME"] = str(tmp_path / "home")
    for name in ("FAKEGPU_PROFILE", "FAKEGPU_PROFILES", "FAKEGPU_DEVICE_COUNT", "FAKEGPU_RUN_DIR"):
        env.pop(name, None)
    return env


def _simulate(tmp_path: Path, body: str) -> dict:
    code = SIMULATED + textwrap.dedent(body) + "\nprint('RESULTS ' + json.dumps(results, sort_keys=True))\n"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=_environment(tmp_path), text=True,
        capture_output=True, timeout=240,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULTS "))
    return json.loads(line[len("RESULTS "):])


def test_patched_functions_accept_every_argument_of_the_installed_torch(tmp_path: Path) -> None:
    results = _simulate(tmp_path, """
        import inspect
        torch_dist = dist.distributed_c10d
        for name in ("init_process_group", "barrier", "all_reduce", "broadcast", "all_gather",
                     "all_gather_into_tensor", "all_gather_object", "reduce", "gather", "scatter",
                     "reduce_scatter", "reduce_scatter_tensor", "all_to_all", "all_to_all_single",
                     "broadcast_object_list", "get_rank", "get_world_size", "new_group"):
            original = inspect.signature(getattr(torch_dist, name)).parameters
            patched = inspect.signature(getattr(dist, name)).parameters
            open_ended = any(p.kind is p.VAR_KEYWORD for p in patched.values())
            results[name] = sorted(set(original) - set(patched)) if not open_ended else []
        import datetime
        probe("call_barrier", lambda: dist.barrier(timeout=datetime.timedelta(seconds=5)))
        probe("call_all_gather_object", lambda: dist.all_gather_object([None] * 8, 1, weights_only=True))
        probe("call_broadcast_object_list", lambda: dist.broadcast_object_list([1], src=0, weights_only=True))
        probe("init_typo", lambda: dist.init_process_group("nccl", world_size=8, rank=5, wolrd_size=8))
    """)
    names = ("init_process_group", "barrier", "all_reduce", "broadcast", "all_gather", "all_gather_into_tensor",
             "all_gather_object", "reduce", "gather", "scatter", "reduce_scatter", "reduce_scatter_tensor",
             "all_to_all", "all_to_all_single", "broadcast_object_list", "get_rank", "get_world_size", "new_group")
    assert {name: results[name] for name in names} == {name: [] for name in names}
    assert results["call_barrier"] is None
    assert results["call_all_gather_object"] is None
    assert results["call_broadcast_object_list"] is None
    assert results["init_typo"][0] == "TypeError"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _torchrun(tmp_path: Path, body: str, nproc: int = 3) -> dict[int, dict]:
    """Run ``body`` on ``nproc`` FakeCUDA ranks under ``fakegpu run`` and torchrun; return each rank's results."""

    script = tmp_path / "ranks.py"
    script.write_text(textwrap.dedent("""
        import datetime, json, os, warnings
        import torch, torch.distributed as dist
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
        results = {}
        def probe(name, fn):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results[name] = fn()
            except Exception as exc:
                results[name] = [type(exc).__name__, str(exc).splitlines()[0] if str(exc) else ""]
            dist.barrier()
        def t(*values):
            return torch.tensor(values, dtype=torch.float32, device="cuda")
    """) + textwrap.dedent(body) + textwrap.dedent("""
        os.write(1, ("RESULTS " + json.dumps([rank, results], sort_keys=True) + "\\n").encode())
        dist.destroy_process_group()
    """), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "fakegpu", "run", "--gpu", f"a100x{nproc}", "--no-summary", "--",
         sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
         "--master_addr", "127.0.0.1", "--master_port", str(_free_port()), str(script)],
        cwd=tmp_path, env=_environment(tmp_path), text=True, capture_output=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    ranks = [json.loads(line[len("RESULTS "):]) for line in completed.stdout.splitlines() if line.startswith("RESULTS ")]
    return {rank: results for rank, results in ranks}


def test_launched_ranks_match_gloo(tmp_path: Path) -> None:
    ranks = _torchrun(tmp_path, """
        probe("barrier_timeout", lambda: dist.barrier(timeout=datetime.timedelta(seconds=60)))
        def objects():
            gathered = [None] * world
            dist.all_gather_object(gathered, {"r": rank}, weights_only=True)
            sent = [{"r": rank}]
            dist.broadcast_object_list(sent, src=1, weights_only=True)
            return [gathered, sent]
        probe("weights_only", objects)
        pair = dist.new_group([0, 1])  # rank 2 is not a member
        def outside(call):
            def run():
                out = t(-1.0)
                returned = call(out)
                return [out.tolist(), returned]
            return run
        probe("pair/all_gather_into_tensor", outside(
            lambda out: dist.all_gather_into_tensor(torch.empty(2, device="cuda") if rank < 2 else out,
                                                    t(rank * 1.0), group=pair)))
        probe("pair/reduce_scatter_tensor", outside(
            lambda out: dist.reduce_scatter_tensor(out, t(rank + 1.0, 10.0 * (rank + 1)), group=pair)))
        probe("pair/reduce_scatter", outside(
            lambda out: dist.reduce_scatter(out, [t(rank + 1.0), t(10.0 * (rank + 1))], group=pair)))
        probe("pair/reduce_scatter_avg", outside(
            lambda out: dist.reduce_scatter(out, [t(rank + 1.0), t(10.0 * (rank + 1))], op=dist.ReduceOp.AVG,
                                            group=pair)))
        probe("pair/all_reduce_avg_async", outside(
            lambda out: dist.all_reduce(out, op=dist.ReduceOp.AVG, group=pair, async_op=True) is None))
        probe("pair/reduce_avg_async", outside(
            lambda out: dist.reduce(out, dst=0, op=dist.ReduceOp.AVG, group=pair, async_op=True) is None))
        def work_handles():
            handles = []
            x = t(rank + 1.0)
            gathered = torch.empty(3, device="cuda")
            part = torch.empty(1, device="cuda")
            for output, work in ((x, dist.all_reduce(x, async_op=True)),
                                 (gathered, dist.all_gather_into_tensor(gathered, t(rank * 1.0), async_op=True)),
                                 (part, dist.reduce_scatter_tensor(part, t(1.0, 2.0, 3.0), async_op=True))):
                future = work.get_future()
                handles.append([isinstance(work, dist.Work), work.wait(), work.is_completed(),
                                future.wait()[0] is output, output.tolist()])
            return handles
        probe("work", work_handles)
        probe("band", lambda: dist.all_reduce(torch.tensor([rank + 1], device="cuda"), op=dist.ReduceOp.BAND))
        everyone = dist.new_group([0, 1, 2], backend="gloo")
        def gloo_band():
            flags = torch.tensor([rank + 1, 7], device="cuda")
            dist.all_reduce(flags, op=dist.ReduceOp.BAND, group=everyone)
            return flags.tolist()
        probe("gloo_band", gloo_band)
    """)
    for rank in range(3):
        assert ranks[rank]["barrier_timeout"] is None
        assert ranks[rank]["weights_only"] == [[{"r": 0}, {"r": 1}, {"r": 2}], [{"r": 1}]]
    # gloo without FakeGPU: members exchange values, rank 2 keeps its tensor and gets None.
    assert ranks[0]["pair/reduce_scatter_tensor"] == [[3.0], None]
    assert ranks[1]["pair/reduce_scatter_tensor"] == [[30.0], None]
    assert ranks[0]["pair/reduce_scatter"] == [[3.0], None]
    assert ranks[1]["pair/reduce_scatter_avg"] == [[15.0], None]
    for name in ("pair/all_gather_into_tensor", "pair/reduce_scatter_tensor", "pair/reduce_scatter",
                 "pair/reduce_scatter_avg"):
        assert ranks[2][name] == [[-1.0], None], name
    assert ranks[2]["pair/all_reduce_avg_async"] == [[-1.0], True]
    assert ranks[2]["pair/reduce_avg_async"] == [[-1.0], True]
    # Finished work whose future holds the outputs, as gloo's and NCCL's work does.
    for rank in range(3):
        assert ranks[rank]["work"] == [
            [True, True, True, True, [6.0]],
            [True, True, True, True, [0.0, 1.0, 2.0]],
            [True, True, True, True, [3.0 * (rank + 1)]],
        ]
        # NCCL refuses bitwise reductions; a gloo group computes them (1 & 2 & 3 == 0).
        assert ranks[rank]["band"] == ["ValueError", "Cannot use ReduceOp.BAND with NCCL"]
        assert ranks[rank]["gloo_band"] == [0, 7]


def test_simulated_subgroups_report_their_own_rank_and_size(tmp_path: Path) -> None:
    # Expected values: the same calls on torch's single-process "fake" backend at rank 5 of 8.
    results = _simulate(tmp_path, """
        tensor_parallel = dist.new_group([4, 5, 6, 7])
        other = dist.new_group([0, 1])
        results["world"] = [dist.get_rank(), dist.get_world_size()]
        results["tensor_parallel"] = [dist.get_rank(tensor_parallel), dist.get_world_size(tensor_parallel)]
        results["other"] = [dist.get_rank(other), dist.get_world_size(other)]
        current, _ = dist.new_subgroups(group_size=2)
        results["subgroup"] = [dist.get_rank(current), dist.get_world_size(current)]
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
        results["mesh_tp"] = [dist.get_rank(mesh.get_group("tp")), dist.get_world_size(mesh.get_group("tp"))]
    """)
    assert results == {"world": [5, 8], "tensor_parallel": [1, 4], "other": [-1, -1], "subgroup": [1, 2],
                       "mesh_tp": [1, 4]}


def test_simulated_subgroups_may_name_a_backend(tmp_path: Path) -> None:
    # Megatron-style setup code names NCCL or gloo for its groups; a real 8-rank job creates them.
    results = _simulate(tmp_path, """
        for backend in ("nccl", "gloo", None):
            probe(str(backend), lambda: dist.get_world_size(dist.new_group([4, 5, 6, 7], backend=backend)))
    """)
    assert results == {"nccl": 4, "gloo": 4, "None": 4}


def test_simulated_uneven_exchanges_keep_their_shapes(tmp_path: Path) -> None:
    # NCCL and torch's fake backend accept splits whose totals differ, as in MoE token dispatch.
    results = _simulate(tmp_path, """
        received = torch.zeros(9, device="cuda")
        def exchange():
            dist.all_to_all_single(received, torch.arange(6.0, device="cuda"), [1, 2, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 0, 0])
            return list(received.shape)
        probe("all_to_all_single", exchange)
        pieces = [torch.zeros(size, device="cuda") for size in (1, 2, 3, 4, 5, 6, 7, 8)]
        probe("all_gather", lambda: [dist.all_gather(pieces, torch.ones(6, device="cuda")),
                                     [list(piece.shape) for piece in pieces]])
    """)
    assert results == {"all_to_all_single": [9],
                        "all_gather": [None, [[1], [2], [3], [4], [5], [6], [7], [8]]]}


def test_nccl_rejects_the_reductions_it_lacks(tmp_path: Path) -> None:
    # torch v2.14.1 ProcessGroupNCCL.cpp getNcclReduceOp: no bitwise ops, no AVG of booleans.
    results = _simulate(tmp_path, """
        flags = torch.tensor([1, 2], device="cuda")
        for name in ("BAND", "BOR", "BXOR"):
            op = getattr(dist.ReduceOp, name)
            probe("all_reduce/" + name, lambda: dist.all_reduce(flags, op=op))
            probe("reduce/" + name, lambda: dist.reduce(flags, dst=0, op=op))
        probe("reduce_scatter_tensor/BOR", lambda: dist.reduce_scatter_tensor(
            torch.zeros(1, dtype=torch.int64, device="cuda"), torch.ones(8, dtype=torch.int64, device="cuda"),
            op=dist.ReduceOp.BOR))
        probe("all_reduce/AVG/bool", lambda: dist.all_reduce(torch.ones(2, dtype=torch.bool, device="cuda"),
                                                             op=dist.ReduceOp.AVG))
        probe("all_reduce/MAX", lambda: dist.all_reduce(flags, op=dist.ReduceOp.MAX))
        gloo = dist.new_group(list(range(8)), backend="gloo")
        probe("gloo_group/BAND", lambda: [dist.get_backend(gloo), dist.all_reduce(flags, op=dist.ReduceOp.BAND,
                                                                                    group=gloo)])
    """)
    for name in ("BAND", "BOR", "BXOR"):
        assert results["all_reduce/" + name] == ["ValueError", f"Cannot use ReduceOp.{name} with NCCL"]
        assert results["reduce/" + name] == ["ValueError", f"Cannot use ReduceOp.{name} with NCCL"]
    assert results["reduce_scatter_tensor/BOR"] == ["ValueError", "Cannot use ReduceOp.BOR with NCCL"]
    assert results["all_reduce/AVG/bool"] == ["TypeError", "Cannot use ReduceOp.AVG with boolean inputs"]
    assert results["all_reduce/MAX"] is None
    assert results["gloo_group/BAND"] == ["gloo", None]


def test_simulated_async_work_completes_with_its_outputs(tmp_path: Path) -> None:
    results = _simulate(tmp_path, """
        x = torch.ones(2, device="cuda")
        work = dist.all_reduce(x, async_op=True)
        results["all_reduce"] = [isinstance(work, dist.Work), work.wait(), work.is_completed(),
                                 work.get_future().wait()[0] is x]
        work = dist.barrier(async_op=True)
        results["barrier"] = [work.is_completed(), work.get_future().wait()]
    """)
    assert results == {"all_reduce": [True, True, True, True], "barrier": [True, []]}
