# Step 19 Hybrid Validation Report

- `nproc_per_node`: 2
- `runner_exit_code`: 0
- `cluster_report_check_exit_code`: 0
- `overall_status`: success
- `runner_log`: `/home/l1ght/repos/fakeGPU/test/output/hybrid_multinode_2r_runner.log`
- `coordinator_log`: `/home/l1ght/repos/fakeGPU/test/output/hybrid_multinode_2r_coordinator.log`
- `cluster_report`: `/home/l1ght/repos/fakeGPU/test/output/hybrid_multinode_2r_cluster_report.json`

## Rank Results

- rank 0: status=success matmul_checksum=1176.0 all_reduce=3.0 broadcast=2048
- rank 0 detail: matmul_max_abs_diff=0.0 comm_init=0 destroy=0
- rank 1: status=success matmul_checksum=1312.0 all_reduce=3.0 broadcast=2048
- rank 1 detail: matmul_max_abs_diff=0.0 comm_init=0 destroy=0

## Cluster Report Summary

- world_size=2 node_count=2 communicators=1
- all_reduce: calls=1 bytes=8
- broadcast: calls=1 bytes=8
- links: 2

## Log Excerpt

- {"rank": 0, "world_size": 2, "device_index": 0, "status": "success", "torch_version": "2.9.1+cu128", "cuda_device": 0, "matmul_checksum": 1176.0, "matmul_reference_checksum": 1176.0, "matmul_max_abs_diff": 0.0, "comm_init_result": 0, "all_reduce_result": 0, "broadcast_result": 0, "all_reduce_value": 3.0, "all_reduce_expected": 3.0, "broadcast_value": 2048, "destroy_result": 0}
- {"rank": 1, "world_size": 2, "device_index": 0, "status": "success", "torch_version": "2.9.1+cu128", "cuda_device": 0, "matmul_checksum": 1312.0, "matmul_reference_checksum": 1312.0, "matmul_max_abs_diff": 0.0, "comm_init_result": 0, "all_reduce_result": 0, "broadcast_result": 0, "all_reduce_value": 3.0, "all_reduce_expected": 3.0, "broadcast_value": 2048, "destroy_result": 0}
