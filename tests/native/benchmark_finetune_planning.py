"""CPU planning/storage benchmark; requires the supported HF native environment.

Use separate baseline and current checkouts via PYTHONPATH, or compare the
current planner with and without --reuse-profile. Synthetic rows and repeated
report content are not representative throughput or compression guarantees.
CUDA is hidden before importing Torch. Outputs are outside the source tree.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import gc
import hashlib
import json
import time
import tracemalloc
from types import SimpleNamespace
import torch
from accelerate.data_loader import SeedableRandomSampler
from fakegpu._finetune_sampler import plan_native_batches
from fakegpu.finetune_data import profile_tokenized_records
from fakegpu.structured_io import write_json, load_mapping
import argparse

parser = argparse.ArgumentParser(
    description="Compare duplicate versus reused native dataset scans on lazy synthetic rows."
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--rows", type=int, default=100000)
parser.add_argument("--reuse-profile", action="store_true")
parser.add_argument("--storage-repetitions", type=int, default=2500)
a = parser.parse_args()
if a.rows < 1 or a.storage_repetitions < 1:
    parser.error("rows and storage repetitions must be positive")
mode = "optimized" if a.reuse_profile else "baseline"
root = a.output_dir.resolve()
root.mkdir(parents=True, exist_ok=True)
N = a.rows


class Rows:
    def __init__(self):
        self.reads = 0

    def __len__(self):
        return N

    def __iter__(self):
        for i in range(N):
            yield self[i]

    def __getitem__(self, i):
        if not 0 <= i < N:
            raise IndexError(i)
        self.reads += 1
        ids = [i % 31 + 1] * 8
        return {"input_ids": ids, "attention_mask": [1] * 8, "labels": [-100] + ids[1:]}


def collate(rows):
    return {k: torch.tensor([r[k] for r in rows]) for k in rows[0]}


data = Rows()
sampler = SeedableRandomSampler(data, data_seed=42)
args = SimpleNamespace(
    world_size=1,
    n_gpu=0,
    dataloader_num_workers=0,
    dataloader_prefetch_factor=None,
    dataloader_persistent_workers=False,
    dataloader_pin_memory=False,
    dataloader_drop_last=False,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=8,
    data_seed=42,
    seed=42,
)
trainer = SimpleNamespace(
    train_dataset=data,
    args=args,
    accelerator=SimpleNamespace(num_processes=1),
    data_collator=collate,
)
trainer.get_train_dataloader = lambda: torch.utils.data.DataLoader(
    data, batch_size=8, sampler=sampler, collate_fn=collate
)
gc.collect()
tracemalloc.start()
start = time.perf_counter()
profile = profile_tokenized_records(
    data, sequence_length=16, source_framework="huggingface"
)
kw = {} if mode == "baseline" else {"dataset_profile": profile}
plan = plan_native_batches(
    trainer, max_steps=4, framework="huggingface", sequence_length=16, **kw
)
elapsed = time.perf_counter() - start
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
summary = {
    "scope": "Lazy dataset; native SeedableRandomSampler; synthetic deterministic collator; no GPU timing",
    "mode": mode,
    "rows": N,
    "row_reads": data.reads,
    "optimizer_steps": 4,
    "planning_seconds": elapsed,
    "planning_tracemalloc_peak_bytes": peak,
    "plan_sha256": plan["plan_sha256"],
}
(root / (mode + "-plan.json")).write_text(json.dumps(plan, sort_keys=True))
if mode != "baseline":
    payload = {
        "schema_version": "fakegpu.storage_benchmark.v1",
        "repeated_synthetic_updates": plan["updates"] * a.storage_repetitions,
    }
    summary["storage"] = {}
    for suffix in ("json", "json.gz"):
        path = root / ("large-report." + suffix)
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        write_json(path, payload)
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert load_mapping(path) == payload
        summary["storage"][suffix] = {
            "bytes": path.stat().st_size,
            "write_seconds": seconds,
            "write_tracemalloc_peak_bytes": peak,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
(root / (mode + "-summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
