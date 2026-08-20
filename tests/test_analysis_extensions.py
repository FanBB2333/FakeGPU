from __future__ import annotations

from contextlib import nullcontext
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from fakegpu.calibration import (
    BUNDLE_SCHEMA_VERSION,
    CalibrationError,
    SERVING_OBSERVATION_SCHEMA_VERSION,
    SERVING_SAMPLE_SCHEMA_VERSION,
    VERIFICATION_SCHEMA_VERSION,
    build_cuda_serving_sample,
    build_serving_memory_observation,
    build_workload_calibration_bundle,
    collect_serving_memory_observation,
    compare_memory_reports,
    measure_transformers_serving_sample,
    _parse_serving_runner_sample,
    verify_calibration_reports,
)
from fakegpu.kernel_analysis import analyze_ptx, estimate_occupancy
from fakegpu.operator_profiles import (
    evaluate_operator_profile,
    load_operator_profiles,
    match_operator_profile,
)
from fakegpu.repository_analyzer import analyze_repository
from fakegpu.topology import Topology, simulate_collective
from fakegpu.trace_replay import replay_trace
from fakegpu.training_plan import (
    estimate_training_plan,
    normalize_training_config,
)


ROOT = Path(__file__).resolve().parents[1]
MEMORY_EVIDENCE_PATH = (
    ROOT / "tests" / "data" / "memory_validation_evidence.json"
)


def _serving_protocol_plan() -> dict:
    return {
        "schema_version": "fakegpu.llm_serving_plan.v1",
        "method": "test_serving_model",
        "workload_signature": (
            "fakegpu.serving_workload.v1:sha256:" + "e" * 64
        ),
        "inputs": {
            "dtype": "bfloat16",
            "active_sequences": 2,
            "prompt_tokens": 8,
            "generated_tokens": 3,
            "attention_implementation": "sdpa",
            "prefill_chunk_tokens": None,
            "shared_prefix_tokens": 0,
            "kv_cache_strategy": "dynamic",
            "kv_cache_max_tokens": None,
            "kv_cache_window_tokens": None,
            "speculative_decoding": {"enabled": False},
        },
        "target": {
            "profile": {
                "id": "a100",
                "compute_capability": "8.0",
                "memory_bytes": 80 * 2**30,
            }
        },
        "memory_timeline": {
            "phases": [
                {"phase": "prefill", "peak_bytes": 1_000},
                {"phase": "decode", "peak_bytes": 1_200},
            ]
        },
    }


def test_analysis_apis_are_exported_from_package() -> None:
    import fakegpu

    for name in (
        "build_cuda_serving_sample",
        "build_serving_memory_observation",
        "collect_serving_memory_observation",
        "compare_memory_reports",
        "estimate_kv_cache_memory",
        "estimate_training_plan",
        "measure_transformers_serving_sample",
        "analyze_kernel_file",
        "replay_trace",
        "simulate_collective",
    ):
        assert callable(getattr(fakegpu, name))
    assert (
        fakegpu.SERVING_SAMPLE_SCHEMA_VERSION
        == SERVING_SAMPLE_SCHEMA_VERSION
    )


def _topology_config() -> dict:
    return {
        "ecmp": True,
        "switches": [
            {"id": "leaf0"},
            {"id": "leaf1"},
            {"id": "spine0"},
        ],
        "nodes": [
            {
                "id": "node0",
                "rack": "rack0",
                "nics": [
                    {
                        "id": "node0.nic0",
                        "switch": "leaf0",
                        "bandwidth_gbps": 100,
                    }
                ],
            },
            {
                "id": "node1",
                "rack": "rack0",
                "nics": [
                    {
                        "id": "node1.nic0",
                        "switch": "leaf0",
                        "bandwidth_gbps": 100,
                    }
                ],
            },
            {
                "id": "node2",
                "rack": "rack1",
                "nics": [
                    {
                        "id": "node2.nic0",
                        "switch": "leaf1",
                        "bandwidth_gbps": 100,
                    }
                ],
            },
            {
                "id": "node3",
                "rack": "rack1",
                "nics": [
                    {
                        "id": "node3.nic0",
                        "switch": "leaf1",
                        "bandwidth_gbps": 100,
                    }
                ],
            },
        ],
        "links": [
            {
                "src": "leaf0",
                "dst": "spine0",
                "bandwidth_gbps": 200,
                "latency_us": 0.5,
                "scope": "leaf_spine",
            },
            {
                "src": "leaf1",
                "dst": "spine0",
                "bandwidth_gbps": 200,
                "latency_us": 0.5,
                "scope": "leaf_spine",
            },
        ],
        "ranks": [
            {"rank": 0, "node": "node0"},
            {"rank": 1, "node": "node1"},
            {"rank": 2, "node": "node2"},
            {"rank": 3, "node": "node3"},
        ],
    }


def test_calibration_comparison_and_bundle() -> None:
    prediction = {
        "schema_version": "static_memory_estimate.v1",
        "inputs": {"target_profile": "a100"},
        "memory_timeline": {
            "phases": [
                {
                    "phase": "forward",
                    "peak_bytes": 900,
                    "interval_bytes": {
                        "lower": 800,
                        "expected": 900,
                        "upper": 1_100,
                    },
                },
                {"phase": "backward", "peak_bytes": 1_800},
            ]
        },
        "unmodeled_components": ["cuda_context_and_loaded_modules"],
    }
    observation = {
        "schema_version": "measurement.v1",
        "profile": "a100",
        "memory_timeline": {
            "phases": [
                {"phase": "forward", "peak_bytes": 1_000},
                {"phase": "backward", "peak_bytes": 2_000},
            ]
        },
    }
    report = compare_memory_reports(
        prediction,
        observation,
        workload="tiny-sft",
    )

    assert report["summary"]["phase_count"] == 2
    assert report["summary"]["recommended_memory_safety_margin_bytes"] == 200
    assert report["summary"]["recommended_memory_safety_factor"] > 1
    assert report["comparisons"][1]["absolute_percentage_error"] == 0.1
    forward = next(
        item for item in report["comparisons"] if item["phase"] == "forward"
    )
    assert forward[
        "observation_within_prediction_interval"
    ] is True

    bundle = build_workload_calibration_bundle(
        [report],
        labels=["tiny-sft"],
    )
    assert bundle["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert bundle["entries"][0]["id"] == "tiny-sft"

    verification = verify_calibration_reports(
        [report],
        labels=["tiny-sft"],
        max_underestimate_percent=10,
        max_absolute_percentage_error_percent=10,
        min_interval_coverage_percent=100,
        capacity_bytes=2_100,
        min_comparisons=2,
    )
    assert verification["schema_version"] == VERIFICATION_SCHEMA_VERSION
    assert verification["status"] == "passed"
    assert verification["metrics"][
        "maximum_underestimate_percent"
    ] == pytest.approx(10)
    assert verification["metrics"]["false_safe_count"] == 0


def test_serving_observation_aggregates_repeated_peaks_for_calibration() -> None:
    plan = {
        "schema_version": "fakegpu.llm_serving_plan.v1",
        "method": "test_serving_model",
        "workload_signature": (
            "fakegpu.serving_workload.v1:sha256:" + "a" * 64
        ),
        "inputs": {
            "dtype": "bfloat16",
            "prompt_tokens": 4096,
        },
        "target": {
            "profile": {
                "id": "a100",
                "compute_capability": "8.0",
            }
        },
        "memory_timeline": {
            "phases": [
                {"phase": "prefill", "peak_bytes": 1_000},
                {"phase": "decode", "peak_bytes": 1_200},
            ]
        },
    }
    observation = build_serving_memory_observation(
        plan,
        phase_samples={
            "prefill": [1_000, 1_020, 1_010, 1_005, 1_030],
            "decode": [1_180, 1_200, 1_210, 1_190, 1_230],
        },
        source="torch.cuda.max_memory_reserved",
    )

    assert (
        observation["schema_version"]
        == SERVING_OBSERVATION_SCHEMA_VERSION
    )
    assert observation["evidence_status"] == "ready_for_comparison"
    assert observation["profile"] == "a100"
    assert observation["compute_capability"] == "8.0"
    assert observation["measurement"]["phase_sample_counts"] == {
        "prefill": 5,
        "decode": 5,
    }
    assert observation["measurement"]["insufficient_phases"] == []
    assert observation["memory_timeline"]["peak_bytes"] == 1_230
    phases = {
        item["phase"]: item
        for item in observation["memory_timeline"]["phases"]
    }
    assert phases["prefill"]["peak_bytes"] == 1_030
    assert phases["prefill"]["median_bytes"] == 1_010
    assert phases["decode"]["samples_bytes"] == [
        1_180,
        1_200,
        1_210,
        1_190,
        1_230,
    ]

    comparison = compare_memory_reports(
        plan,
        observation,
        workload="speculative-serving",
    )
    verification = verify_calibration_reports(
        [comparison],
        max_underestimate_percent=3,
        max_absolute_percentage_error_percent=3,
        min_comparisons=2,
    )
    assert verification["status"] == "passed"
    assert verification["metrics"]["dimension_mismatch_count"] == 0

    insufficient = build_serving_memory_observation(
        plan,
        phase_samples={
            "prefill": [1_000, 1_010],
            "decode": [1_200, 1_210],
        },
    )
    assert insufficient["evidence_status"] == "insufficient_samples"
    assert insufficient["measurement"]["insufficient_phases"] == [
        "prefill",
        "decode",
    ]

    with pytest.raises(CalibrationError, match="missing phases: decode"):
        build_serving_memory_observation(
            plan,
            phase_samples={"prefill": [1_000]},
        )
    with pytest.raises(CalibrationError, match="must be a positive"):
        build_serving_memory_observation(
            plan,
            phase_samples={
                "prefill": [0],
                "decode": [1_200],
            },
        )
    with pytest.raises(CalibrationError, match="select a target profile"):
        build_serving_memory_observation(
            {**plan, "target": {"profile": None}},
            phase_samples={
                "prefill": [1_000],
                "decode": [1_200],
            },
        )


def test_serving_collector_repeats_runner_and_checks_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_bytes = 80 * 2**30
    observed_memory_bytes = memory_bytes + int(memory_bytes * 0.018)
    signature = "fakegpu.serving_workload.v1:sha256:" + "c" * 64
    plan = {
        "schema_version": "fakegpu.llm_serving_plan.v1",
        "method": "test_serving_model",
        "workload_signature": signature,
        "inputs": {"dtype": "bfloat16", "prompt_tokens": 4096},
        "target": {
            "profile": {
                "id": "a100",
                "compute_capability": "8.0",
                "memory_bytes": memory_bytes,
            }
        },
        "memory_timeline": {
            "phases": [
                {"phase": "prefill", "peak_bytes": 1_000},
                {"phase": "decode", "peak_bytes": 1_200},
            ]
        },
    }

    def sample_payload(
        run_index: int,
        *,
        simulated: bool = False,
        framework_version: str = "1.2.3",
        total_memory_bytes: int = observed_memory_bytes,
    ) -> dict:
        return {
            "schema_version": SERVING_SAMPLE_SCHEMA_VERSION,
            "workload_signature": signature,
            "run_index": run_index,
            "metric": "torch.cuda.max_memory_reserved",
            "phases": {
                "prefill": {"peak_bytes": 1_000 + run_index},
                "decode": {"peak_bytes": 1_200 + run_index},
            },
            "environment": {
                "backend": "cuda",
                "simulated": simulated,
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "gpu_uuid": "GPU-test",
                "compute_capability": [8, 0],
                "total_memory_bytes": total_memory_bytes,
                "software": {
                    "framework": "transformers",
                    "framework_version": framework_version,
                    "cuda_version": "12.8",
                    "torch_version": "2.9.1",
                },
            },
        }

    def completed_run(command: list[str], **kwargs: object) -> object:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        run_index = int(environment["FAKEGPU_SERVING_RUN_INDEX"])
        assert environment["FAKEGPU_SERVING_RUN_COUNT"] == "3"
        assert environment["FAKEGPU_SERVING_WORKLOAD_SIGNATURE"] == signature
        stdout = (
            "framework log\nFAKEGPU_SERVING_SAMPLE="
            + json.dumps(sample_payload(run_index))
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(
        "fakegpu.calibration.subprocess.run",
        completed_run,
    )
    observation = collect_serving_memory_observation(
        plan,
        command=["python", "private-runner.py", "--token=secret"],
        repetitions=3,
        minimum_samples_per_phase=3,
    )

    assert observation["evidence_status"] == "ready_for_comparison"
    assert observation["memory_timeline"]["peak_bytes"] == 1_203
    assert observation["measurement"]["reported_metric"] == (
        "torch.cuda.max_memory_reserved"
    )
    assert observation["measurement"]["environment"]["gpu_uuid"] == (
        "GPU-test"
    )
    assert observation["measurement"]["environment"][
        "target_capacity_difference_percent"
    ] == pytest.approx(1.8)
    assert observation["measurement"]["environment"][
        "target_capacity_tolerance_percent"
    ] == 2.0
    runner = observation["measurement"]["runner"]
    assert runner["successful_runs"] == 3
    assert runner["executable"] == "python"
    assert runner["command_fingerprint"].startswith("sha256:")
    assert "secret" not in json.dumps(observation)
    assert [
        item["phase_peaks_bytes"]["prefill"] for item in runner["runs"]
    ] == [1_001, 1_002, 1_003]

    def simulated_run(command: list[str], **kwargs: object) -> object:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        run_index = int(environment["FAKEGPU_SERVING_RUN_INDEX"])
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(sample_payload(run_index, simulated=True)),
            "",
        )

    monkeypatch.setattr(
        "fakegpu.calibration.subprocess.run",
        simulated_run,
    )
    with pytest.raises(CalibrationError, match="simulated must be false"):
        collect_serving_memory_observation(
            plan,
            command=["python", "simulated-runner.py"],
            repetitions=1,
            minimum_samples_per_phase=1,
        )

    def mismatched_capacity_run(
        command: list[str],
        **kwargs: object,
    ) -> object:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        run_index = int(environment["FAKEGPU_SERVING_RUN_INDEX"])
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                sample_payload(
                    run_index,
                    total_memory_bytes=memory_bytes // 2,
                )
            ),
            "",
        )

    monkeypatch.setattr(
        "fakegpu.calibration.subprocess.run",
        mismatched_capacity_run,
    )
    with pytest.raises(CalibrationError, match="within 2.0%"):
        collect_serving_memory_observation(
            plan,
            command=["python", "mismatched-gpu.py"],
            repetitions=1,
            minimum_samples_per_phase=1,
        )

    def changing_environment_run(
        command: list[str],
        **kwargs: object,
    ) -> object:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        run_index = int(environment["FAKEGPU_SERVING_RUN_INDEX"])
        payload = sample_payload(
            run_index,
            framework_version="1.2.4" if run_index == 2 else "1.2.3",
        )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(payload),
            "",
        )

    monkeypatch.setattr(
        "fakegpu.calibration.subprocess.run",
        changing_environment_run,
    )
    with pytest.raises(CalibrationError, match="changed its GPU or software"):
        collect_serving_memory_observation(
            plan,
            command=["python", "changing-runner.py"],
            repetitions=2,
            minimum_samples_per_phase=2,
        )

    def timed_out_run(command: list[str], **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(
        "fakegpu.calibration.subprocess.run",
        timed_out_run,
    )
    with pytest.raises(CalibrationError, match="timed out on repetition 1"):
        collect_serving_memory_observation(
            plan,
            command=["python", "slow-runner.py"],
            repetitions=1,
            minimum_samples_per_phase=1,
            timeout_seconds=0.01,
        )


def test_serving_sample_parser_does_not_treat_log_json_as_sample() -> None:
    stdout = 'framework log\n{"level": "info", "message": "ready"}\n'

    with pytest.raises(
        CalibrationError,
        match="did not emit a 'fakegpu.serving_peak_sample.v1' JSON object",
    ):
        _parse_serving_runner_sample(stdout, run_index=2)


def test_transformers_runner_measures_phases_and_builds_vllm_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _serving_protocol_plan()
    model_calls: list[dict[str, object]] = []

    class FakeLogits:
        def __getitem__(self, key: object) -> "FakeLogits":
            return self

        def argmax(self, **kwargs: object) -> object:
            assert kwargs == {"dim": -1, "keepdim": True}
            return SimpleNamespace(shape=(2, 1))

    class FakeModel:
        config = SimpleNamespace(
            vocab_size=32,
            bos_token_id=1,
            pad_token_id=None,
            eos_token_id=2,
        )

        def to(self, device: str) -> "FakeModel":
            assert device == "cuda:0"
            return self

        def eval(self) -> None:
            return None

        def __call__(self, **kwargs: object) -> object:
            model_calls.append(dict(kwargs))
            return SimpleNamespace(
                past_key_values=f"cache-{len(model_calls)}",
                logits=FakeLogits(),
            )

    model = FakeModel()

    class FakeFactory:
        load_kwargs: dict[str, object] | None = None

        @classmethod
        def from_pretrained(
            cls,
            model_dir: str,
            **kwargs: object,
        ) -> FakeModel:
            assert model_dir == str(tmp_path)
            cls.load_kwargs = dict(kwargs)
            return model

    class FakeCuda:
        reset_count = 0
        empty_cache_count = 0

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def set_device(device: int) -> None:
            assert device == 0

        @staticmethod
        def synchronize(device: int) -> None:
            assert device == 0

        @classmethod
        def reset_peak_memory_stats(cls, device: int) -> None:
            assert device == 0
            cls.reset_count += 1

        @classmethod
        def empty_cache(cls) -> None:
            cls.empty_cache_count += 1

        @classmethod
        def max_memory_reserved(cls, device: int) -> int:
            assert device == 0
            return 1_000 if cls.reset_count == 1 else 1_200

        @classmethod
        def max_memory_allocated(cls, device: int) -> int:
            assert device == 0
            return 900 if cls.reset_count == 1 else 1_100

        @staticmethod
        def get_device_properties(device: int) -> object:
            assert device == 0
            return SimpleNamespace(
                name="NVIDIA A100-SXM4-80GB",
                total_memory=80 * 2**30,
                uuid="GPU-built-in-runner",
            )

        @staticmethod
        def get_device_capability(device: int) -> tuple[int, int]:
            assert device == 0
            return (8, 0)

        @staticmethod
        def get_device_name(device: int) -> str:
            assert device == 0
            return "NVIDIA A100-SXM4-80GB"

        @staticmethod
        def get_allocator_backend() -> str:
            return "native"

    fake_torch = SimpleNamespace(
        __version__="2.9.1",
        version=SimpleNamespace(cuda="12.8"),
        cuda=FakeCuda,
        bfloat16="torch.bfloat16",
        long="torch.int64",
        full=lambda shape, value, **kwargs: SimpleNamespace(
            shape=shape,
            value=value,
            kwargs=kwargs,
        ),
        inference_mode=nullcontext,
    )
    fake_transformers = SimpleNamespace(
        __version__="4.57.1",
        AutoModelForCausalLM=FakeFactory,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "fakegpu.calibration._transformers_serving_workload",
        lambda supplied_plan, *, model_dir: {
            "model_dir": tmp_path,
            "active_sequences": 2,
            "prompt_tokens": 8,
            "generated_tokens": 3,
            "attention_implementation": "sdpa",
            "dtype": "bfloat16",
        },
    )
    for key in (
        "FAKEGPU_MODE",
        "FAKEGPU_SERVING_RUN_INDEX",
        "FAKEGPU_SERVING_SAMPLE_SCHEMA",
        "FAKEGPU_SERVING_PLAN_SCHEMA",
        "FAKEGPU_SERVING_WORKLOAD_SIGNATURE",
        "FAKEGPU_SERVING_TARGET_PROFILE",
        "FAKEGPU_SERVING_COMPUTE_CAPABILITY",
    ):
        monkeypatch.delenv(key, raising=False)

    sample = measure_transformers_serving_sample(
        plan,
        run_index=4,
    )

    assert sample["run_index"] == 4
    assert sample["metric"] == "torch.cuda.max_memory_reserved"
    assert sample["phases"] == {
        "prefill": {"peak_bytes": 1_000},
        "decode": {"peak_bytes": 1_200},
    }
    assert sample["environment"]["gpu_uuid"] == "GPU-built-in-runner"
    assert sample["environment"]["software"] == {
        "framework": "transformers",
        "framework_version": "4.57.1",
        "cuda_version": "12.8",
        "torch_version": "2.9.1",
        "attention_implementation": "sdpa",
        "cache_implementation": "dynamic",
        "dtype": "bfloat16",
        "runner": "fakegpu.calibrate.sample-transformers",
    }
    assert len(model_calls) == 3
    assert "past_key_values" not in model_calls[0]
    assert model_calls[1]["past_key_values"] == "cache-1"
    assert model_calls[2]["past_key_values"] == "cache-2"
    assert FakeCuda.empty_cache_count == 1
    assert FakeFactory.load_kwargs is not None
    assert FakeFactory.load_kwargs["local_files_only"] is True
    assert FakeFactory.load_kwargs["use_safetensors"] is True

    vllm_sample = build_cuda_serving_sample(
        plan,
        phase_peaks={"prefill": 2_000, "decode": 2_400},
        metric="nvml.process_family_peak_bytes",
        framework="vllm",
        framework_version="0.10.2",
        run_index=5,
    )
    assert vllm_sample["phases"]["decode"]["peak_bytes"] == 2_400
    assert vllm_sample["environment"]["software"]["framework"] == "vllm"

    monkeypatch.setenv(
        "FAKEGPU_SERVING_WORKLOAD_SIGNATURE",
        "wrong-signature",
    )
    with pytest.raises(CalibrationError, match="does not match"):
        build_cuda_serving_sample(
            plan,
            phase_peaks={"prefill": 2_000, "decode": 2_400},
            metric="nvml.process_family_peak_bytes",
            framework="vllm",
            framework_version="0.10.2",
            run_index=5,
        )


def test_transformers_runner_rejects_unmodeled_serving_semantics() -> None:
    paged_plan = _serving_protocol_plan()
    paged_plan["inputs"]["kv_cache_strategy"] = "paged"
    with pytest.raises(CalibrationError, match="requires.*dynamic"):
        measure_transformers_serving_sample(paged_plan)

    request_set_plan = _serving_protocol_plan()
    request_set_plan["schema_version"] = (
        "fakegpu.llm_serving_request_set_plan.v1"
    )
    with pytest.raises(CalibrationError, match="homogeneous"):
        measure_transformers_serving_sample(request_set_plan)


def test_calibration_verification_rejects_oom_risk_and_signature_drift() -> None:
    report = compare_memory_reports(
        {
            "schema_version": "prediction.v1",
            "inputs": {"batch_size": 1, "dtype": "bfloat16"},
            "memory_timeline": {
                "phases": [{"phase": "peak", "peak_bytes": 900}]
            },
        },
        {
            "schema_version": "observation.v1",
            "inputs": {"batch_size": 2, "dtype": "bfloat16"},
            "memory_timeline": {
                "phases": [{"phase": "peak", "peak_bytes": 1_000}]
            },
        },
    )

    verification = verify_calibration_reports(
        [report],
        max_underestimate_percent=5,
        max_absolute_percentage_error_percent=5,
        min_interval_coverage_percent=90,
        capacity_bytes=950,
    )

    assert verification["status"] == "failed"
    assert verification["metrics"]["false_safe_count"] == 1
    assert verification["metrics"]["dimension_mismatch_count"] == 1
    assert {failure["gate"] for failure in verification["failures"]} == {
        "matching_workload_dimensions",
        "maximum_absolute_percentage_error_percent",
        "maximum_false_safe_count",
        "maximum_underestimate_percent",
        "minimum_prediction_interval_coverage_percent",
    }

    zero_report = compare_memory_reports(
        {
            "schema_version": "prediction.v1",
            "memory_timeline": {
                "phases": [{"phase": "peak", "peak_bytes": 0}]
            },
        },
        {
            "schema_version": "observation.v1",
            "memory_timeline": {
                "phases": [{"phase": "peak", "peak_bytes": 0}]
            },
        },
    )
    zero_verification = verify_calibration_reports([zero_report])
    assert zero_verification["status"] == "failed"
    assert zero_verification["failures"] == [
        {
            "gate": "positive_observation_count",
            "actual": 0,
            "minimum": 1,
        }
    ]


def test_published_memory_evidence_matches_calibration_math() -> None:
    evidence = json.loads(MEMORY_EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert (
        evidence["schema_version"]
        == "fakegpu.memory_validation_evidence.v1"
    )
    revision = evidence["source_revision"]
    assert len(revision) == 40
    int(revision, 16)

    groups = {
        group["id"]: group for group in evidence["evidence_groups"]
    }
    assert set(groups) == {
        "controlled_aten",
        "qwen3_8b_inference",
        "qwen_sft",
        "qwen_qlora",
    }
    for group in groups.values():
        assert group["source_path"].startswith("docs/")
        assert group["source_anchor"]

    controlled = groups["controlled_aten"]
    assert controlled["workload_count"] == 13
    assert controlled["observation_count"] == 26
    assert (
        controlled[
            "published_maximum_absolute_percentage_error_percent"
        ]
        == 0.08
    )
    assert controlled["published_maximum_underestimate_percent"] == 0.08

    inference = groups["qwen3_8b_inference"]
    prediction = {
        "schema_version": "readme_evidence_prediction.v1",
        "memory_timeline": {
            "phases": [
                {
                    "phase": item["id"],
                    "peak_bytes": item["predicted_bytes"],
                }
                for item in inference["measurements"]
            ]
        },
    }
    observation = {
        "schema_version": "readme_evidence_observation.v1",
        "memory_timeline": {
            "phases": [
                {
                    "phase": item["id"],
                    "peak_bytes": item["observed_bytes"],
                }
                for item in inference["measurements"]
            ]
        },
    }
    comparison = compare_memory_reports(
        prediction,
        observation,
        workload=inference["id"],
    )
    comparisons = {
        item["phase"]: item for item in comparison["comparisons"]
    }
    for measurement in inference["measurements"]:
        actual_percent = (
            comparisons[measurement["id"]][
                "absolute_percentage_error"
            ]
            * 100
        )
        assert actual_percent == pytest.approx(
            measurement[
                "published_absolute_percentage_error_percent"
            ],
            abs=0.0000005,
        )
    assert comparison["summary"]["underprediction_phase_count"] == 2
    assert comparison["summary"]["recommended_memory_safety_margin_bytes"] > 0
    assert comparison["summary"]["recommended_memory_safety_factor"] > 1

    sft_errors = [
        item["published_absolute_percentage_error_percent"]
        for item in groups["qwen_sft"]["measurements"]
    ]
    assert len(sft_errors) == 10
    assert (min(sft_errors), max(sft_errors)) == (0.102, 1.921)

    qlora_errors = [
        item["published_absolute_percentage_error_percent"]
        for item in groups["qwen_qlora"]["measurements"]
    ]
    assert len(qlora_errors) == 10
    assert (min(qlora_errors), max(qlora_errors)) == (0.628, 1.732)


def test_training_plan_normalizes_deepspeed_zero3_offload() -> None:
    normalized = normalize_training_config(
        {
            "world_size": 4,
            "gradient_accumulation_steps": 8,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "overlap_comm": True,
                "reduce_bucket_size": 200,
                "allgather_bucket_size": 300,
            },
            "activation_checkpointing": {"partition_activations": True},
        }
    )
    report = estimate_training_plan(
        normalized,
        parameter_bytes=4_000,
        activation_bytes=2_000,
        optimizer="adamw",
        parameter_element_bytes=2,
    )

    assert normalized["source_framework"] == "deepspeed"
    assert report["sharding"]["parameters_sharded"] is True
    assert report["rank_local_storage"]["parameter_bytes"] == 1_000
    assert report["rank_local_storage"]["optimizer_state_bytes"] == 0
    assert report["cpu_offload"]["optimizer_and_master_parameter_bytes"] == 4_000
    assert report["communication"]["gradient_accumulation_steps"] == 8
    assert len(report["memory_timeline"]["phases"]) == 4


def test_training_plan_handles_accelerate_fsdp_and_auto_deepspeed_values() -> None:
    fsdp = normalize_training_config(
        {
            "distributed_type": "FSDP",
            "num_processes": 2,
            "mixed_precision": "bf16",
            "fsdp_config": {
                "fsdp_sharding_strategy": "SHARD_GRAD_OP",
                "fsdp_activation_checkpointing": "false",
            },
        }
    )
    fsdp_report = estimate_training_plan(
        fsdp,
        parameter_bytes=2_000,
        activation_bytes=500,
    )
    assert fsdp_report["sharding"]["parameters_sharded"] is False
    assert fsdp_report["sharding"]["gradients_sharded"] is True
    assert fsdp_report["rank_local_storage"]["gradient_bytes"] == 1_000
    assert fsdp["activation_checkpointing"] is False

    deepspeed = normalize_training_config(
        {
            "distributed_type": "DEEPSPEED",
            "num_processes": 4,
            "mixed_precision": "bf16",
            "deepspeed_config": {
                "zero_stage": 3,
                "offload_optimizer_device": "cpu",
                "gradient_accumulation_steps": "auto",
                "reduce_bucket_size": "auto",
            },
        }
    )
    assert deepspeed["zero_stage"] == 3
    assert deepspeed["optimizer_offload_device"] == "cpu"
    assert deepspeed["reduce_bucket_bytes"] == 0


def test_hierarchical_topology_routes_across_racks_and_reports_links() -> None:
    topology = Topology.from_mapping(_topology_config())
    report = simulate_collective(
        topology,
        collective="all_reduce",
        payload_bytes_per_rank=1_000_000,
        algorithm="hierarchical",
    )

    assert report["topology"]["rack_count"] == 2
    assert report["summary"]["round_count"] >= 3
    assert report["summary"]["estimated_time_us"] > 0
    assert len(report["rank_summary"]) == 4
    assert any(
        "leaf_spine" in transfer["path_scopes"]
        for transfer in report["transfers"]
    )
    assert report["summary"]["critical_links"]


def test_trace_replay_reports_overlap_pairs_memory_and_recovery() -> None:
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "name": "fused_attention",
                "cat": "compute",
                "pid": 0,
                "ts": 0,
                "dur": 100,
                "args": {
                    "fusion_id": "f0",
                    "shape": [2, 8],
                    "dtype": "bf16",
                    "allocated_bytes": 1_000,
                    "reserved_bytes": 1_200,
                },
            },
            {
                "ph": "X",
                "name": "softmax",
                "cat": "compute",
                "pid": 0,
                "ts": 10,
                "dur": 20,
                "args": {"fusion_id": "f0"},
            },
            {
                "ph": "X",
                "name": "ncclAllReduce",
                "cat": "communication",
                "pid": 0,
                "ts": 50,
                "dur": 100,
                "args": {
                    "bytes": 1_000,
                    "src_rank": 0,
                    "dst_rank": 2,
                    "links": ["leaf0->spine0"],
                },
            },
            {
                "ph": "X",
                "name": "worker_failure",
                "cat": "wait",
                "pid": 0,
                "ts": 160,
                "dur": 10,
                "args": {"generation": 0},
            },
            {
                "ph": "X",
                "name": "elastic_restart",
                "cat": "wait",
                "pid": 0,
                "ts": 180,
                "dur": 10,
                "args": {"generation": 1},
            },
            {
                "ph": "X",
                "name": "matmul",
                "cat": "compute",
                "pid": 2,
                "ts": 0,
                "dur": 120,
                "args": {},
            },
        ]
    }
    profiles = [
        {
            "id": "fused-attention",
            "match": {"operator": "fused_attention", "dtype": "bf16"},
            "model": {"workspace_bytes": 256, "flops_per_element": 4},
            "fused_operators": ["softmax"],
        }
    ]
    report = replay_trace(trace, operator_profiles=profiles)

    rank0 = next(item for item in report["rank_summary"] if item["rank"] == 0)
    assert rank0["compute_communication_overlap_us"] == 50
    assert rank0["communication_wait_us"] == 50
    assert report["rank_pairs"][0]["total_bytes"] == 1_000
    assert report["memory_timeline"]["peak_reserved_bytes"] == 1_200
    assert report["operator_profile_summary"]["matched_event_count"] == 1
    assert report["resilience"]["elastic_restart_observed"] is True
    assert len(report["events"]) == 5


def test_trace_replay_accepts_native_cluster_timeline_shape() -> None:
    report = replay_trace(
        {
            "schema_version": "cluster_report.v1",
            "operation_timeline": {
                "retained_entries": 1,
                "dropped_entries": 0,
                "entries": [
                    {
                        "index": 1,
                        "kind": "collective",
                        "operation": "all_reduce",
                        "data_type": "float32",
                        "ranks": [0, 1],
                        "logical_payload_bytes": 4_096,
                        "rendezvous_wait_us": 5,
                        "execution_time_us": 20,
                        "coordinator_duration_us": 25,
                        "modeled_time_us": 18,
                    }
                ],
            },
        }
    )
    assert report["summary"]["rank_count"] == 2
    assert report["summary"]["event_count"] == 4
    assert report["summary"]["communication_bytes"] == 8_192
    assert all(item["explicit_wait_us"] == 5 for item in report["rank_summary"])


def test_operator_profile_catalog_matches_exact_shape(tmp_path: Path) -> None:
    path = tmp_path / "operators.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "fakegpu.operator_profiles.v1",
                "profiles": [
                    {
                        "id": "custom-op",
                        "priority": 10,
                        "match": {
                            "operator_regex": "custom::.*",
                            "dtype": "fp16",
                            "shape": [2, 4],
                        },
                        "model": {
                            "workspace_bytes": 64,
                            "flops_per_element": 3,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    profiles = load_operator_profiles([path])
    profile = match_operator_profile(
        profiles,
        operator="custom::foo",
        dtype="fp16",
        shape=[2, 4],
    )
    assert profile is not None
    evaluated = evaluate_operator_profile(profile, numel=8)
    assert evaluated["flops"] == 24
    assert evaluated["workspace_bytes"] == 64


def test_ptx_static_analysis_and_profile_occupancy() -> None:
    ptx = """
.version 8.0
.target sm_89
.address_size 64
.visible .entry kernel() {
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .shared .align 4 .b8 scratch[1024];
    ld.global.f32 %f1, [%rd1];
    fma.rn.f32 %f2, %f1, %f1, %f1;
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {%r0}, {%r1}, {%r2}, {%r3};
    st.global.f32 [%rd2], %f2;
    ret;
}
"""
    analysis = analyze_ptx(ptx)
    occupancy = estimate_occupancy(
        analysis,
        profile_id="rtx4090",
        threads_per_block=128,
    )

    assert analysis["entry_points"] == ["kernel"]
    assert analysis["declared_register_count"] == 10
    assert analysis["static_shared_memory_bytes"] == 1_024
    assert analysis["instruction_classes"]["tensor_core"] == 1
    assert analysis["recognized_flops_per_static_issue"] >= 4_096
    assert 0 < occupancy["occupancy_upper_bound"] <= 1


def test_repository_analyzer_detects_aliases_decorators_and_build_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    (repository / "train.py").write_text(
        """
import triton as tr
from torch.utils.cpp_extension import load_inline as build_cuda

@tr.jit
def kernel(x):
    return x

PTX = ".version 8.0\\n.target sm_89\\n.entry generated() {}"
build_cuda(name="x", cpp_sources="", cuda_sources="__global__ void x() {}")

if __name__ == "__main__":
    kernel
""".strip(),
        encoding="utf-8",
    )
    (repository / "CMakeLists.txt").write_text(
        "project(sample LANGUAGES CXX CUDA)\nfind_package(CUDAToolkit)\n",
        encoding="utf-8",
    )
    (repository / "kernel.ptx").write_text(
        ".version 8.0\n.target sm_89\n.entry kernel() { ret; }\n",
        encoding="utf-8",
    )

    report = analyze_repository(repository)
    markers = report["python"]["call_markers"]
    codes = {item["code"] for item in report["findings"]}

    assert markers["triton.jit"] >= 1
    assert markers["torch.utils.cpp_extension.load_inline"] == 1
    assert markers["embedded_cuda_or_ptx_source"] >= 1
    assert report["build_system"]["markers"]["cmake_cuda_language"] == 1
    assert report["kernel_static_analysis"][0]["status"] == "analyzed"
    assert {
        "generated_kernel_path",
        "runtime_cuda_extension_build",
    } <= codes
