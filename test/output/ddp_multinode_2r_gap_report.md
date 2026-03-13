# Step 10 Torch Distributed Smoke Report

- `nproc_per_node`: 2
- `torchrun_exit_code`: 1
- `overall_status`: gap
- `torchrun_log`: `/home/l1ght/repos/fakeGPU/test/output/ddp_multinode_2r_torchrun.log`
- `coordinator_log`: `/home/l1ght/repos/fakeGPU/test/output/ddp_multinode_2r_coordinator.log`

## Missing Rank Reports

- missing ranks: 0, 1

## Torchrun Log Excerpt

- ImportError: /home/l1ght/anaconda3/envs/fakegpu/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommAbort

## Rank Results

- no per-rank reports were generated

## Gap Summary

- launcher failed before rank reports were written: ImportError: /home/l1ght/anaconda3/envs/fakegpu/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommAbort
