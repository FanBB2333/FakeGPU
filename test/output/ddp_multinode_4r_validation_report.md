# Step 15 DDP Validation Report

- `nproc_per_node`: 4
- `torchrun_exit_code`: 0
- `cluster_report_check_exit_code`: 0
- `overall_status`: success
- `torchrun_log`: `/home/l1ght/repos/fakeGPU/test/output/ddp_multinode_4r_torchrun.log`
- `coordinator_log`: `/home/l1ght/repos/fakeGPU/test/output/ddp_multinode_4r_coordinator.log`
- `cluster_report`: `/home/l1ght/repos/fakeGPU/test/output/ddp_multinode_4r_cluster_report.json`

## Rank Results

- rank 0: status=success epochs=1 steps=2 stage=completed
- rank 1: status=success epochs=1 steps=2 stage=completed
- rank 2: status=success epochs=1 steps=2 stage=completed
- rank 3: status=success epochs=1 steps=2 stage=completed

- framework barrier successes: 8

## Cluster Report Summary

- world_size=4 node_count=2 communicators=1
- all_reduce: calls=5 bytes=6832
- broadcast: calls=5 bytes=3888
- barrier: calls=0 bytes=0

## Log Excerpt

- W0315 01:39:54.126000 4191765 site-packages/torch/distributed/run.py:803] 
- W0315 01:39:54.126000 4191765 site-packages/torch/distributed/run.py:803] *****************************************
- W0315 01:39:54.126000 4191765 site-packages/torch/distributed/run.py:803] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
- W0315 01:39:54.126000 4191765 site-packages/torch/distributed/run.py:803] *****************************************
- {"rank": 3, "world_size": 4, "local_rank": 3, "pid": 4191794, "status": "success", "torch_version": "2.9.1+cu128", "cuda_available": true, "cuda_device_count": 4, "init_process_group": "ok", "device": "cuda:3", "barrier_before_training": "ok", "broadcast_value": 2026, "ddp": "ok", "epochs_completed": 1, "steps_completed": 2, "last_loss": 0.0, "post_train_all_reduce": 10.0, "post_train_all_reduce_expected": 10.0, "barrier_after_training": "ok", "destroy_process_group": "ok"}
- {"rank": 2, "world_size": 4, "local_rank": 2, "pid": 4191792, "status": "success", "torch_version": "2.9.1+cu128", "cuda_available": true, "cuda_device_count": 4, "init_process_group": "ok", "device": "cuda:2", "barrier_before_training": "ok", "broadcast_value": 2026, "ddp": "ok", "epochs_completed": 1, "steps_completed": 2, "last_loss": 0.0, "post_train_all_reduce": 10.0, "post_train_all_reduce_expected": 10.0, "barrier_after_training": "ok", "destroy_process_group": "ok"}
- {"rank": 1, "world_size": 4, "local_rank": 1, "pid": 4191791, "status": "success", "torch_version": "2.9.1+cu128", "cuda_available": true, "cuda_device_count": 4, "init_process_group": "ok", "device": "cuda:1", "barrier_before_training": "ok", "broadcast_value": 2026, "ddp": "ok", "epochs_completed": 1, "steps_completed": 2, "last_loss": 0.0, "post_train_all_reduce": 10.0, "post_train_all_reduce_expected": 10.0, "barrier_after_training": "ok", "destroy_process_group": "ok"}{"rank": 0, "world_size": 4, "local_rank": 0, "pid": 4191790, "status": "success", "torch_version": "2.9.1+cu128", "cuda_available": true, "cuda_device_count": 4, "init_process_group": "ok", "device": "cuda:0", "barrier_before_training": "ok", "broadcast_value": 2026, "ddp": "ok", "epochs_completed": 1, "steps_completed": 2, "last_loss": 0.0, "post_train_all_reduce": 10.0, "post_train_all_reduce_expected": 10.0, "barrier_after_training": "ok", "destroy_process_group": "ok"}
- 
