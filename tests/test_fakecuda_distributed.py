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

        class Counter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(1, 1)
                self.register_buffer("count", torch.full((1,), 10.0 * rank))
            def forward(self, x):
                self.count += rank + 1
                return self.fc(x)
        def counter(**options):
            torch.manual_seed(100 + rank)
            return torch.nn.parallel.DistributedDataParallel(Counter().cuda(), device_ids=[rank], **options)
        def buffer_syncs():
            model, x, counts = counter(), torch.ones(1, 1, device="cuda"), []
            model(x).sum().backward()
            counts.append(model.module.count.item())
            with torch.no_grad():
                for _ in range(2):
                    model(x)
                    counts.append(model.module.count.item())
            with model.no_sync():
                model(x).sum().backward()
                counts.append(model.module.count.item())
            for _ in range(2):
                model(x).sum().backward()
                counts.append(model.module.count.item())
            return counts
        probe("buffer_syncs", buffer_syncs)
        def initial_state():
            built = {}
            for name, options in {"default": {}, "broadcast_buffers": {"broadcast_buffers": False},
                                  "init_sync": {"init_sync": False},
                                  "forward_sync_buffers": {"forward_sync_buffers": False}}.items():
                model = counter(**options)
                built[name] = [model.module.count.item(), model.module.fc.weight.item()]
            return built
        probe("initial_state", initial_state)
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
    # Real DDP syncs buffers before the first forward and after a forward that synced
    # gradients: not after no_grad or no_sync forwards, even before a training one.
    assert [ranks[rank]["buffer_syncs"] for rank in range(3)] == [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 5.0, 7.0, 9.0, 7.0], [3.0, 4.0, 7.0, 10.0, 13.0, 8.0]]
    # broadcast_buffers=False keeps each rank's buffers at construction; init_sync=False keeps everything.
    weights = {name: ranks[0]["initial_state"][name][1] for name in ranks[0]["initial_state"]}
    for rank in range(3):
        built = ranks[rank]["initial_state"]
        assert built["default"] == [0.0, weights["default"]]
        assert built["broadcast_buffers"] == [10.0 * rank, weights["broadcast_buffers"]]
        assert built["forward_sync_buffers"] == [0.0, weights["forward_sync_buffers"]]
        assert built["init_sync"][0] == 10.0 * rank
        assert (built["init_sync"][1] == weights["init_sync"]) == (rank == 0)


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


def test_nccl_checks_the_sizes_of_tensor_collectives(tmp_path: Path) -> None:
    # torch v2.14.1 ProcessGroupNCCL.cpp all_gather_single / reduce_scatter_single.
    results = _simulate(tmp_path, """
        tensor_parallel = dist.new_group([4, 5, 6, 7])
        def gather(output, source, group=None):
            return lambda: [dist.all_gather_into_tensor(output, source, group=group), list(output.shape)]
        def scatter(output, source, group=None):
            return lambda: [dist.reduce_scatter_tensor(output, source, group=group), list(output.shape)]
        ones = torch.ones(4, device="cuda")
        probe("gather/concatenated", gather(torch.empty(32, device="cuda"), ones))
        probe("gather/stacked", gather(torch.empty(8, 4, device="cuda"), ones))
        probe("gather/group", gather(torch.empty(16, device="cuda"), ones, tensor_parallel))
        probe("gather/world_buffer_for_group", gather(torch.empty(32, device="cuda"), ones, tensor_parallel))
        probe("gather/dtype", gather(torch.empty(32, dtype=torch.float16, device="cuda"), ones))
        probe("scatter/even", scatter(torch.empty(2, device="cuda"), torch.ones(16, device="cuda")))
        probe("scatter/group", scatter(torch.empty(2, device="cuda"), torch.ones(8, device="cuda"), tensor_parallel))
        probe("scatter/short", scatter(torch.empty(2, device="cuda"), torch.ones(8, device="cuda")))
        probe("scatter/dtype", scatter(torch.empty(2, dtype=torch.bfloat16, device="cuda"),
                                       torch.ones(16, device="cuda")))
    """)
    assert results["gather/concatenated"] == [None, [32]]
    assert results["gather/stacked"] == [None, [8, 4]]
    assert results["gather/group"] == [None, [16]]
    size_error = ["ValueError", "output tensor size must be equal to world_size times input tensor size"]
    assert results["gather/world_buffer_for_group"] == size_error
    assert results["gather/dtype"] == ["TypeError", "output tensor must have the same type as input tensor"]
    assert results["scatter/even"] == [None, [2]]
    assert results["scatter/group"] == [None, [2]]
    assert results["scatter/short"] == [
        "ValueError", "input tensor must be the same size as output size times world size"]
    assert results["scatter/dtype"] == ["TypeError", "input tensor must be the same type as the output tensor."]


def test_ddp_validates_its_arguments_like_torch(tmp_path: Path) -> None:
    # Expected values: real DDP over gloo with the same arguments.
    results = _simulate(tmp_path, """
        from torch.nn.parallel import DistributedDataParallel as DDP
        def build(*args, **kwargs):
            def run():
                model = DDP(torch.nn.Linear(2, 2).cuda(), *args, **kwargs)
                return [model.broadcast_buffers, model.find_unused_parameters, model.static_graph,
                        model.gradient_as_bucket_view]
            return run
        probe("typo", build(find_unused_parameter=True))
        probe("positional", build([5], None, 0, False))
        probe("keywords", build(device_ids=[5], find_unused_parameters=True, static_graph=True,
                                gradient_as_bucket_view=True))
        probe("two_devices", build(device_ids=[4, 5]))
        frozen = torch.nn.Linear(2, 2).cuda().requires_grad_(False)
        probe("frozen", lambda: DDP(frozen) and None)
    """)
    assert results["typo"][0] == "TypeError"
    assert "find_unused_parameter" in results["typo"][1]
    assert results["positional"] == [False, False, False, False]
    assert results["keywords"] == [True, True, True, True]
    assert results["two_devices"] == ["ValueError", "device_ids can only be None or contain a single element."]
    assert results["frozen"] == [
        "RuntimeError",
        "DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.",
    ]


def test_ddp_fails_on_unused_parameters_like_its_reducer(tmp_path: Path) -> None:
    # Expected outcomes: real DDP over gloo; it raises at the next forward with gradients enabled.
    results = _simulate(tmp_path, """
        from torch.nn.parallel import DistributedDataParallel as DDP
        class Heads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b, self.c = (torch.nn.Linear(2, 1) for _ in range(3))
            def forward(self, x):
                return self.b(x)
        x = torch.ones(1, 2, device="cuda")
        def steps(count, **options):
            def run():
                model = DDP(Heads().cuda(), **options)
                for _ in range(count):
                    model(x).sum().backward()
                return "trained"
            return run
        probe("default", steps(2))
        probe("find_unused_parameters", steps(3, find_unused_parameters=True))
        probe("static_graph", steps(3, static_graph=True))
        def evaluate_then_train():
            model = DDP(Heads().cuda())
            model(x).sum().backward()
            with torch.no_grad():
                model(x)
            with model.no_sync():
                try:
                    model(x)
                except RuntimeError:
                    return "failed in no_sync"
            return "evaluated"
        probe("evaluate_then_train", evaluate_then_train)
        def accumulate():
            model = DDP(Heads().cuda())
            with model.no_sync():
                model(x).sum().backward()
                model(x).sum().backward()
            model(x).sum().backward()
            return "accumulated"
        probe("accumulate", accumulate)
    """)
    assert results["default"][0] == "RuntimeError"
    assert results["default"][1].startswith(
        "Expected to have finished reduction in the prior iteration before starting a new one.")
    assert results["find_unused_parameters"] == "trained"
    assert results["static_graph"] == "trained"
    assert results["evaluate_then_train"] == "failed in no_sync"
    assert results["accumulate"] == "accumulated"


def test_ddp_names_the_parameters_without_gradients(tmp_path: Path) -> None:
    results = _simulate(tmp_path, """
        from torch.nn.parallel import DistributedDataParallel as DDP
        class Heads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b, self.c = (torch.nn.Linear(2, 1) for _ in range(3))
            def forward(self, x):
                return self.b(x)
        model = DDP(Heads().cuda())
        model(torch.ones(1, 2, device="cuda")).sum().backward()
        try:
            model(torch.ones(1, 2, device="cuda"))
        except RuntimeError as exc:
            results["message"] = str(exc)
    """)
    # Real gloo DDP with the same module reports "... for rank 0: 0 1 4 5" on rank 0.
    assert results["message"].startswith("Expected to have finished reduction in the prior iteration")
    assert "\nParameter indices which did not receive grad for rank 5: 0 1 4 5\n" in results["message"]


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


# A single-rank job, as the real-GPU oracle ran (init_process_group("nccl") at world size 1).
SINGLE_RANK = """
import json, os, warnings
for name in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "TORCHELASTIC_RUN_ID", "FAKEGPU_DIST_COLLECTIVES"):
    os.environ.pop(name, None)
os.environ["FAKEGPU_TERMINAL_REPORT"] = "0"
import fakegpu
fakegpu.init(runtime="fakecuda", profile="a100", device_count=1)
import torch, torch.distributed as dist
results = {}
def probe(name, fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results[name] = fn()
    except Exception as exc:
        results[name] = [type(exc).__name__, str(exc).splitlines()[0] if str(exc) else ""]
"""


def _single_rank(tmp_path: Path, body: str) -> dict:
    code = SINGLE_RANK + textwrap.dedent(body) + "\nprint('RESULTS ' + json.dumps(results, sort_keys=True))\n"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=_environment(tmp_path), text=True,
        capture_output=True, timeout=240,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULTS "))
    return json.loads(line[len("RESULTS "):])


NO_CPU_BACKEND = ["RuntimeError", "No backend type associated with device type cpu"]
NOT_INITIALIZED = ["ValueError",
                   "Default process group has not been initialized, please make sure to call init_process_group."]


def test_cpu_tensors_need_a_cpu_backend(tmp_path: Path) -> None:
    # Oracle: init_process_group("nccl") on a GPU (torch 2.9.1); torch 2.14.1 maps devices the same way.
    results = _single_rank(tmp_path, """
        def all_reduced(tensor):
            dist.all_reduce(tensor)
            return tensor.tolist()
        cpu = torch.ones(1)
        for backend in ("nccl", None, "cpu:gloo,cuda:nccl", "gloo"):
            name = str(backend)
            dist.init_process_group(backend, world_size=1, rank=0)
            probe(name + "/get_backend", lambda: str(dist.get_backend()))
            probe(name + "/cpu", lambda: all_reduced(torch.ones(1)))
            probe(name + "/cuda", lambda: all_reduced(torch.ones(1, device="cuda")))
            if backend == "nccl":
                probe("nccl/async", lambda: dist.all_reduce(torch.ones(1), async_op=True))
                probe("nccl/broadcast", lambda: dist.broadcast(torch.ones(1), src=0))
                probe("nccl/all_gather", lambda: dist.all_gather([torch.zeros(1)], torch.ones(1)))
                probe("nccl/mixed_all_gather",
                      lambda: dist.all_gather([torch.zeros(1)], torch.ones(1, device="cuda")))
                probe("nccl/mixed_into_tensor",
                      lambda: dist.all_gather_into_tensor(torch.zeros(1), torch.ones(1, device="cuda")))
                probe("nccl/reduce_scatter_tensor",
                      lambda: dist.reduce_scatter_tensor(torch.zeros(1), torch.ones(1)))
                probe("nccl/all_to_all_single", lambda: dist.all_to_all_single(torch.zeros(1), torch.ones(1)))
                probe("nccl/gather", lambda: dist.gather(torch.ones(1), [torch.zeros(1)], dst=0))
                probe("nccl/barrier", lambda: dist.barrier())
                probe("nccl/all_gather_object", lambda: dist.all_gather_object([None], {"a": 1}))
                gloo = dist.new_group([0], backend="gloo")
                probe("nccl/gloo_group", lambda: (dist.all_reduce(cpu, group=gloo), cpu.tolist())[1])
                probe("nccl/nccl_group", lambda: dist.all_reduce(torch.ones(1), group=dist.new_group([0])))
            dist.destroy_process_group()
    """)
    assert results["nccl/get_backend"] == "nccl"
    for name in ("nccl/cpu", "nccl/async", "nccl/broadcast", "nccl/all_gather", "nccl/reduce_scatter_tensor",
                 "nccl/all_to_all_single", "nccl/gather", "nccl/nccl_group"):
        assert results[name] == NO_CPU_BACKEND, name
    assert results["nccl/cuda"] == [1.0]
    assert results["nccl/mixed_all_gather"] is None
    assert results["nccl/mixed_into_tensor"] == ["ValueError", "Tensors must be CUDA and dense"]
    assert results["nccl/barrier"] is None
    assert results["nccl/all_gather_object"] is None
    assert results["nccl/gloo_group"] == [1.0]
    # No backend named: torch reports "undefined" and gives only CUDA its NCCL backend.
    assert results["None/get_backend"] == "undefined"
    assert results["None/cpu"] == NO_CPU_BACKEND
    assert results["None/cuda"] == [1.0]
    for backend in ("cpu:gloo,cuda:nccl", "gloo"):
        assert results[backend + "/get_backend"] == backend
        assert results[backend + "/cpu"] == [1.0]
        assert results[backend + "/cuda"] == [1.0]


def test_monitored_barrier_needs_a_cpu_backend(tmp_path: Path) -> None:
    results = _single_rank(tmp_path, """
        for backend in ("nccl", None, "cpu:gloo,cuda:nccl", "gloo"):
            dist.init_process_group(backend, world_size=1, rank=0)
            probe(str(backend), lambda: dist.monitored_barrier())
            dist.destroy_process_group()
        results["new_check"] = torch.torch_version.TorchVersion(torch.__version__) >= (2, 14)
    """)
    if results["new_check"]:
        # torch 2.14 distributed_c10d.monitored_barrier: the group needs a CPU device type.
        refused = ["ValueError", "monitored_barrier requires a group with a CPU-capable backend (e.g. GLOO)."]
        assert results["cpu:gloo,cuda:nccl"] is None
    else:
        # Oracle, torch 2.9.1 on a GPU: only a gloo group qualifies.
        refused = ["ValueError", "monitored_barrier is only implemented for GLOO backend."]
        assert results["cpu:gloo,cuda:nccl"] == refused
    assert results["nccl"] == refused
    assert results["None"] == refused
    assert results["gloo"] is None


def test_premul_sum_scales_before_summing(tmp_path: Path) -> None:
    # Oracle at world size 1 on a GPU: factor 0.5 turns [2, 4] into [1, 2]; ProcessGroupNCCL getNcclReduceOp.
    results = _single_rank(tmp_path, """
        dist.init_process_group("nccl", world_size=1, rank=0)
        def reduced(dtype, factor, call="all_reduce"):
            def run():
                tensor = torch.tensor([2.0, 4.0], device="cuda").to(dtype)
                op = dist._make_nccl_premul_sum(factor)
                if call == "all_reduce":
                    dist.all_reduce(tensor, op=op)
                    return tensor.float().tolist()
                out = torch.empty(2, device="cuda", dtype=dtype)
                dist.reduce_scatter_tensor(out, tensor, op=op)
                return out.float().tolist()
            return run
        for dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16, torch.int32):
            probe(str(dtype), reduced(dtype, 0.5))
        probe("tensor_factor", reduced(torch.float32, torch.tensor([0.25], device="cuda")))
        probe("tensor_factor_dtype",
              reduced(torch.float32, torch.tensor([0.25], device="cuda", dtype=torch.float64)))
        probe("reduce_scatter_tensor", reduced(torch.float32, 0.5, "reduce_scatter_tensor"))
        probe("bare_enum", lambda: dist.all_reduce(torch.ones(1, device="cuda"), op=dist.ReduceOp.PREMUL_SUM))
        probe("cpu", lambda: dist.all_reduce(torch.ones(1), op=dist._make_nccl_premul_sum(0.5)))
        results["fixed_bf16"] = torch.torch_version.TorchVersion(torch.__version__) >= (2, 14)
    """)
    for dtype in ("torch.float32", "torch.float16", "torch.float64"):
        assert results[dtype] == [1.0, 2.0], dtype
    # torch < 2.14 handed NCCL a float factor for bfloat16, which read its low two bytes: 0 for 0.5,
    # as the GPU run with torch 2.9.1 shows. torch 2.14 passes a bfloat16 factor.
    assert results["torch.bfloat16"] == ([1.0, 2.0] if results["fixed_bf16"] else [0.0, 0.0])
    assert results["torch.int32"] == ["TypeError", "PreMulSum Data type must be half, float, bfloat16 or double"]
    assert results["tensor_factor"] == [0.5, 1.0]
    assert results["tensor_factor_dtype"] == ["RuntimeError", "expected scalar type Float but found Double"]
    assert results["reduce_scatter_tensor"] == [1.0, 2.0]
    assert results["bare_enum"][0] == "TypeError"
    assert results["cpu"] == NO_CPU_BACKEND


def test_launched_premul_sum_matches_nccl(tmp_path: Path) -> None:
    # NCCL computes sum(factor * x_r); a gloo group still rejects PREMUL_SUM, as gloo does.
    ranks = _torchrun(tmp_path, """
        def reduced(make_op, call="all_reduce"):
            def run():
                tensor = t(rank + 1.0, 10.0 * (rank + 1))
                if call == "all_reduce":
                    dist.all_reduce(tensor, op=make_op())
                    return tensor.tolist()
                if call == "reduce":
                    dist.reduce(tensor, dst=1, op=make_op())
                    return tensor.tolist()
                out = torch.empty(1, device="cuda")
                if call == "reduce_scatter":
                    dist.reduce_scatter(out, [t(rank + 1.0), t(10.0 * (rank + 1)), t(100.0)], op=make_op())
                else:
                    dist.reduce_scatter_tensor(out, t(rank + 1.0, 10.0 * (rank + 1), 100.0), op=make_op())
                return out.tolist()
            return run
        half = lambda: dist._make_nccl_premul_sum(0.5)
        probe("all_reduce", reduced(half))
        probe("tensor_factor", reduced(lambda: dist._make_nccl_premul_sum(torch.tensor([0.5], device="cuda"))))
        probe("reduce", reduced(half, "reduce"))
        probe("reduce_scatter", reduced(half, "reduce_scatter"))
        probe("reduce_scatter_tensor", reduced(half, "reduce_scatter_tensor"))
        try:
            dist.ReduceOp.PREMUL_SUM(0.5)  # FSDP2's form in torch 2.14
            probe("enum_call", reduced(lambda: dist.ReduceOp.PREMUL_SUM(0.5)))
        except TypeError:
            pass
        everyone = dist.new_group([0, 1, 2], backend="gloo")
        probe("gloo_group", lambda: dist.all_reduce(t(1.0), op=dist._make_nccl_premul_sum(0.5), group=everyone))
    """)
    for rank in range(3):
        results = ranks[rank]
        assert results["all_reduce"] == [3.0, 30.0]
        assert results["tensor_factor"] == [3.0, 30.0]
        assert results["reduce"] == ([3.0, 30.0] if rank == 1 else [rank + 1.0, 10.0 * (rank + 1)])
        assert results["reduce_scatter"] == [[3.0], [30.0], [150.0]][rank]
        assert results["reduce_scatter_tensor"] == [[3.0], [30.0], [150.0]][rank]
        if "enum_call" in results:
            assert results["enum_call"] == [3.0, 30.0]
        assert results["gloo_group"][0] == "RuntimeError"
        assert "PREMUL_SUM" in results["gloo_group"][1]


def test_queries_before_init_process_group_fail_like_torch(tmp_path: Path) -> None:
    # Oracle: torch 2.9.1 on a GPU raises for each query before init and after destroy.
    results = _single_rank(tmp_path, """
        probe("get_rank", dist.get_rank)
        probe("get_world_size", dist.get_world_size)
        probe("get_backend", dist.get_backend)
    """)
    assert results == {"get_rank": NOT_INITIALIZED, "get_world_size": NOT_INITIALIZED,
                       "get_backend": NOT_INITIALIZED}


def test_simulated_gather_and_scatter_use_every_slot(tmp_path: Path) -> None:
    # gloo, 3 ranks: dst gathers one tensor per rank, rank r receives scatter_list[r], and a list on the
    # wrong rank fails. A simulated rank's peers send what it sends.
    results = _simulate(tmp_path, """
        def gather(dst):
            def run():
                slots = [torch.full((2,), -1.0, device="cuda") for _ in range(8)] if dst == 5 else None
                dist.gather(torch.full((2,), 5.0, device="cuda"), slots, dst=dst)
                return None if slots is None else [slot.tolist() for slot in slots]
            return run
        probe("gather/dst", gather(5))
        probe("gather/missing_list", lambda: dist.gather(torch.ones(2, device="cuda"), None, dst=5))
        probe("gather/list_elsewhere", lambda: dist.gather(torch.ones(2, device="cuda"),
                                                           [torch.ones(2, device="cuda")] * 8, dst=1))
        def scatter():
            out = torch.full((2,), -1.0, device="cuda")
            dist.scatter(out, [torch.full((2,), 10.0 * r, device="cuda") for r in range(8)], src=5)
            return out.tolist()
        probe("scatter/src", scatter)
        probe("scatter/list_elsewhere", lambda: dist.scatter(torch.ones(2, device="cuda"),
                                                             [torch.ones(2, device="cuda")] * 8, src=0))
        group = dist.new_group([4, 5, 6, 7])
        def group_scatter():
            out = torch.zeros(1, device="cuda")
            dist.scatter(out, [torch.full((1,), float(r), device="cuda") for r in range(4)], src=5, group=group)
            return out.tolist()
        probe("scatter/group", group_scatter)
    """)
    assert results["gather/dst"] == [[5.0, 5.0]] * 8
    assert results["gather/missing_list"] == [
        "ValueError", "Argument ``gather_list`` must be specified on destination rank."]
    assert results["gather/list_elsewhere"] == [
        "ValueError", "Argument ``gather_list`` must NOT be specified on non-destination ranks."]
    assert results["scatter/src"] == [50.0, 50.0]
    assert results["scatter/list_elsewhere"] == [
        "ValueError", "Argument ``scatter_list`` must NOT be specified on non-source ranks."]
    assert results["scatter/group"] == [1.0]


def test_new_group_binds_positional_arguments_like_torch(tmp_path: Path) -> None:
    results = _simulate(tmp_path, """
        import datetime, warnings
        group = dist.new_group([4, 5, 6, 7], datetime.timedelta(seconds=60), "gloo", None, False, "tensor_parallel")
        results["group"] = [dist.get_backend(group), dist.get_world_size(group), group.group_desc]
        probe("too_many", lambda: dist.new_group([4, 5, 6, 7], None, None, None, False, None, None, True, 1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist.all_gather_into_tensor(torch.empty(2, device="cuda"), torch.ones(1, device="cuda"),
                                        group=dist.new_group([0, 1]))
        results["warning"] = [str(w.message) for w in caught if "does not belong" in str(w.message)]
        results["single_name"] = hasattr(dist.distributed_c10d, "all_gather_single")
    """)
    assert results["group"] == ["gloo", 4, "tensor_parallel"]
    assert results["too_many"][0] == "TypeError"
    # torch names the operation after the function that warns (all_gather_single since 2.10).
    name = "all_gather_single" if results["single_name"] else "all_gather_into_tensor"
    assert results["warning"] == [f"Running {name} on global rank 5 which does not belong to the given group."]

