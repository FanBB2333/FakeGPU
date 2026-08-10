from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fakegpu import (
    estimate_diffusion_generation,
    estimate_kv_cache_memory,
    estimate_serving_kv_pool,
    estimate_serving_request_kv_pool,
    estimate_training_plan,
)
from fakegpu.__main__ import BUILTIN_COMMANDS
from fakegpu.profile_catalog import (
    architecture_for_compute_capability,
    catalog_summary,
    load_profiles,
    official_compute_capabilities,
    validate_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
README_PATHS = (
    "README.md",
    "README.zh-CN.md",
    "README.zh-TW.md",
)
MEMORY_EVIDENCE_PATH = (
    ROOT / "tests" / "data" / "memory_validation_evidence.json"
)


def _anchored_readme_section(readme: str, anchor: str) -> str:
    match = re.search(
        rf'<a id="{re.escape(anchor)}"></a>\n\n'
        rf"### [^\n]+\n\n"
        rf"(?P<body>.*?)"
        rf'(?=\n(?:<a id="[^"]+"></a>\n\n)?### )',
        readme,
        flags=re.DOTALL,
    )
    assert match is not None, anchor
    return match.group("body")


def _run_fakegpu(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    env.setdefault("XONSH_HISTORY_BACKEND", "dummy")
    return subprocess.run(
        [sys.executable, "-m", "fakegpu", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )


@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [
        (5, 2, "maxwell"),
        (6, 0, "pascal"),
        (6, 1, "pascal"),
        (7, 0, "volta"),
        (7, 5, "turing"),
        (8, 0, "ampere"),
        (8, 6, "ampere"),
        (8, 7, "ampere"),
        (8, 9, "ada"),
        (9, 0, "hopper"),
        (10, 0, "blackwell"),
        (10, 3, "blackwell"),
        (11, 0, "blackwell"),
        (12, 0, "blackwell"),
        (12, 1, "blackwell"),
        (8, 8, "unknown"),
        (10, 1, "unknown"),
        (13, 0, "unknown"),
    ],
)
def test_compute_capability_architecture_mapping(
    major: int,
    minor: int,
    expected: str,
) -> None:
    assert architecture_for_compute_capability(major, minor) == expected


def test_profile_catalog_matches_nvidia_snapshot() -> None:
    profiles = load_profiles()
    validation = validate_catalog(profiles)
    summary = catalog_summary(profiles)

    assert validation.errors == ()
    assert summary["profile_count"] == 82
    assert set(summary["architectures"]) == {
        "maxwell",
        "pascal",
        "volta",
        "turing",
        "ampere",
        "ada",
        "hopper",
        "blackwell",
    }
    assert summary["segments"] == {
        "consumer": 40,
        "datacenter": 16,
        "embedded": 2,
        "test": 1,
        "workstation": 23,
    }
    assert summary["compute_capabilities"] == [
        "5.2",
        "6.0",
        "6.1",
        "7.0",
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0",
        "10.0",
        "10.3",
        "11.0",
        "12.0",
        "12.1",
    ]
    for profile in profiles.values():
        path_parts = Path(profile.profile_path).parts
        assert path_parts == (
            profile.architecture,
            profile.segment,
            f"{profile.id}.yaml",
        )

    expected_profile_capabilities = {
        "gtx1080ti": "6.1",
        "rtx2080": "7.5",
        "rtx2080-super": "7.5",
        "rtx2080ti": "7.5",
        "rtx3070": "8.6",
        "rtx4090": "8.9",
        "rtx5090": "12.0",
        "quadro-rtx8000": "7.5",
        "rtx-a6000": "8.6",
        "rtx-6000-ada": "8.9",
        "rtx-pro-4000-blackwell": "12.0",
    }
    for profile_id, capability in expected_profile_capabilities.items():
        assert profiles[profile_id].compute_capability_text == capability

    official = official_compute_capabilities()
    expected_models = {
        "Tesla P4": "6.1",
        "NVIDIA A30": "8.0",
        "NVIDIA A10": "8.6",
        "Jetson AGX Orin": "8.7",
        "NVIDIA L4": "8.9",
        "NVIDIA H200": "9.0",
        "NVIDIA B200": "10.0",
        "NVIDIA B300": "10.3",
        "Jetson T5000": "11.0",
        "NVIDIA RTX PRO 5000 Blackwell": "12.0",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": "12.0",
        "NVIDIA GB10 (DGX Spark)": "12.1",
    }
    for model, capability in expected_models.items():
        assert official[model] == capability


def test_torch_patch_registries_are_generated_from_yaml() -> None:
    from fakegpu import torch_patch

    profiles = load_profiles()
    assert set(torch_patch._PROFILE_CC) == set(profiles)
    for profile_id, profile in profiles.items():
        assert torch_patch._PROFILE_CC[profile_id] == profile.compute_capability
        assert torch_patch._PROFILE_NAMES[profile_id] == profile.torch_name
        assert torch_patch._PROFILE_TOTAL_MEMORY[profile_id] == profile.memory_bytes
        assert (
            torch_patch._PROFILE_SUPPORTED_TYPES[profile_id]
            == profile.supported_types
        )
        assert (
            torch_patch._arch_name(profile.compute_major, profile.compute_minor)
            == profile.architecture.title()
        )


def test_doctor_reports_selected_blackwell_profile_as_json() -> None:
    result = _run_fakegpu("doctor", "--profile", "b300", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["profile_summary"]["profile_count"] == 82
    assert payload["selected_profile"]["id"] == "b300"
    assert payload["selected_profile"]["architecture"] == "blackwell"
    assert payload["selected_profile"]["segment"] == "datacenter"
    assert (
        payload["selected_profile"]["profile_path"]
        == "blackwell/datacenter/b300.yaml"
    )
    assert payload["selected_profile"]["compute_capability"] == "10.3"
    assert payload["selected_profile"]["compiler_target"] == "sm_103"


def test_doctor_rejects_unknown_profile() -> None:
    result = _run_fakegpu("doctor", "--profile", "does-not-exist", "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any(
        item["status"] == "fail" and "unknown GPU profile" in item["detail"]
        for item in payload["checks"]
    )


def test_demo_runs_tiny_training_with_ada_profile() -> None:
    result = _run_fakegpu("demo", "--profile", "l4", "--steps", "1", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["profile_id"] == "l4"
    assert payload["architecture"] == "ada"
    assert payload["compute_capability"] == "8.9"
    assert payload["compiler_target"] == "sm_89"
    assert payload["tensor_device"] == "cuda:0"
    assert payload["tensor_is_cuda"] is True
    assert payload["steps"] == 1


def test_fakecuda_profile_matrix() -> None:
    profiles = load_profiles()
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT)
        if not env.get("PYTHONPATH")
        else str(ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    result = subprocess.run(
        [sys.executable, "tests/support/fakecuda_profile_matrix.py"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert (
        f"validated {len(profiles)} profiles across 15 compute capabilities"
        in result.stdout
    )


def test_calibrate_compare_cli_writes_error_and_safety_data(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "prediction.json"
    observation_path = tmp_path / "observation.json"
    output_path = tmp_path / "comparison.json"
    prediction_path.write_text(
        json.dumps(
            {
                "schema_version": "test_prediction.v1",
                "memory_timeline": {
                    "phases": [
                        {"phase": "peak", "peak_bytes": 900},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    observation_path.write_text(
        json.dumps(
            {
                "schema_version": "test_observation.v1",
                "memory_timeline": {
                    "phases": [
                        {"phase": "peak", "peak_bytes": 1_000},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    result = _run_fakegpu(
        "calibrate",
        "compare",
        str(prediction_path),
        str(observation_path),
        "--json",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    comparison = json.loads(output_path.read_text(encoding="utf-8"))
    assert (
        comparison["schema_version"]
        == "fakegpu.calibration_comparison.v1"
    )
    assert comparison["comparisons"][0]["absolute_percentage_error"] == 0.1
    summary = comparison["summary"]
    assert summary["underprediction_phase_count"] == 1
    assert summary["recommended_memory_safety_margin_bytes"] == 100
    assert summary["recommended_memory_safety_factor"] == pytest.approx(
        10 / 9
    )

    verification_path = tmp_path / "verification.json"
    result = _run_fakegpu(
        "calibrate",
        "verify",
        str(output_path),
        "--max-underestimate-percent",
        "5",
        "--capacity-bytes",
        "950",
        "--json",
        str(verification_path),
    )
    assert result.returncode == 1, result.stderr
    verification = json.loads(
        verification_path.read_text(encoding="utf-8")
    )
    assert verification["status"] == "failed"
    assert verification["metrics"]["false_safe_count"] == 1
    assert {
        failure["gate"] for failure in verification["failures"]
    } == {
        "maximum_false_safe_count",
        "maximum_underestimate_percent",
    }


def test_calibrate_observe_serving_cli_tracks_sample_sufficiency(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "serving-plan.json"
    observation_path = tmp_path / "serving-observation.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "fakegpu.llm_serving_plan.v1",
                "method": "test_serving_model",
                "workload_signature": (
                    "fakegpu.serving_workload.v1:sha256:" + "b" * 64
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
        ),
        encoding="utf-8",
    )

    result = _run_fakegpu(
        "calibrate",
        "observe-serving",
        str(plan_path),
        "--prefill-peak-bytes",
        "1_000,1_010,1_020",
        "--prefill-peak-bytes",
        "1_030,1_040",
        "--decode-peak-bytes",
        "1_200,1_210,1_220,1_230,1_240",
        "--source",
        "torch.cuda.max_memory_reserved",
        "--strict",
        "--json",
        str(observation_path),
    )
    assert result.returncode == 0, result.stderr
    assert "Serving observation:" in result.stdout
    observation = json.loads(
        observation_path.read_text(encoding="utf-8")
    )
    assert observation["evidence_status"] == "ready_for_comparison"
    assert observation["measurement"]["phase_sample_counts"] == {
        "prefill": 5,
        "decode": 5,
    }
    assert observation["memory_timeline"]["peak_bytes"] == 1_240

    insufficient_path = tmp_path / "insufficient-observation.json"
    result = _run_fakegpu(
        "calibrate",
        "observe-serving",
        str(plan_path),
        "--prefill-peak-bytes",
        "1000",
        "--decode-peak-bytes",
        "1200",
        "--strict",
        "--json",
        str(insufficient_path),
    )
    assert result.returncode == 1, result.stderr
    insufficient = json.loads(
        insufficient_path.read_text(encoding="utf-8")
    )
    assert insufficient["evidence_status"] == "insufficient_samples"
    assert insufficient["measurement"]["insufficient_phases"] == [
        "prefill",
        "decode",
    ]

    result = _run_fakegpu(
        "calibrate",
        "observe-serving",
        str(plan_path),
        "--prefill-peak-bytes",
        "invalid",
        "--decode-peak-bytes",
        "1200",
    )
    assert result.returncode == 2
    assert "prefill peak sample must be an integer" in result.stderr


def test_calibrate_collect_serving_cli_runs_json_sample_protocol(
    tmp_path: Path,
) -> None:
    memory_bytes = 80 * 2**30
    signature = "fakegpu.serving_workload.v1:sha256:" + "d" * 64
    plan_path = tmp_path / "serving-plan.json"
    observation_path = tmp_path / "collected-observation.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "fakegpu.llm_serving_plan.v1",
                "method": "test_serving_model",
                "workload_signature": signature,
                "inputs": {
                    "dtype": "bfloat16",
                    "prompt_tokens": 4096,
                },
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
        ),
        encoding="utf-8",
    )
    runner_code = "\n".join(
        (
            "import json, os",
            "index = int(os.environ['FAKEGPU_SERVING_RUN_INDEX'])",
            "payload = {",
            "  'schema_version': os.environ['FAKEGPU_SERVING_SAMPLE_SCHEMA'],",
            "  'workload_signature': os.environ['FAKEGPU_SERVING_WORKLOAD_SIGNATURE'],",
            "  'run_index': index,",
            "  'metric': 'torch.cuda.max_memory_reserved',",
            "  'phases': {",
            "    'prefill': {'peak_bytes': 1000 + index},",
            "    'decode': {'peak_bytes': 1200 + index},",
            "  },",
            "  'environment': {",
            "    'backend': 'cuda', 'simulated': False,",
            "    'gpu_name': 'NVIDIA A100-SXM4-80GB',",
            "    'gpu_uuid': 'GPU-cli-test',",
            "    'compute_capability': [8, 0],",
            f"    'total_memory_bytes': {memory_bytes},",
            "    'software': {",
            "      'framework': 'vllm',",
            "      'framework_version': '0.10.0',",
            "      'cuda_version': '12.8',",
            "    },",
            "  },",
            "}",
            "print('runner log')",
            "print('FAKEGPU_SERVING_SAMPLE=' + json.dumps(payload))",
        )
    )

    result = _run_fakegpu(
        "calibrate",
        "collect-serving",
        str(plan_path),
        "--repetitions",
        "3",
        "--minimum-samples-per-phase",
        "3",
        "--timeout-seconds",
        "10",
        "--strict",
        "--json",
        str(observation_path),
        "--",
        sys.executable,
        "-c",
        runner_code,
    )
    assert result.returncode == 0, result.stderr
    observation = json.loads(
        observation_path.read_text(encoding="utf-8")
    )
    assert observation["evidence_status"] == "ready_for_comparison"
    assert observation["memory_timeline"]["peak_bytes"] == 1_203
    assert observation["measurement"]["phase_sample_counts"] == {
        "prefill": 3,
        "decode": 3,
    }
    assert observation["measurement"]["environment"]["software"][
        "framework"
    ] == "vllm"
    assert observation["measurement"]["runner"]["successful_runs"] == 3
    assert runner_code not in json.dumps(observation)

    failed = _run_fakegpu(
        "calibrate",
        "collect-serving",
        str(plan_path),
        "--repetitions",
        "1",
        "--minimum-samples-per-phase",
        "1",
        "--",
        sys.executable,
        "-c",
        "raise SystemExit(7)",
    )
    assert failed.returncode == 2
    assert "failed on repetition 1 with status 7" in failed.stderr


def test_calibrate_framework_sample_commands_reject_invalid_inputs(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "serving-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "fakegpu.llm_serving_plan.v1",
                "method": "test_serving_model",
                "workload_signature": (
                    "fakegpu.serving_workload.v1:sha256:" + "f" * 64
                ),
                "inputs": {
                    "dtype": "bfloat16",
                    "active_sequences": 1,
                    "prompt_tokens": 8,
                    "generated_tokens": 2,
                    "kv_cache_strategy": "paged",
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
        ),
        encoding="utf-8",
    )

    result = _run_fakegpu(
        "calibrate",
        "sample-transformers",
        str(plan_path),
    )
    assert result.returncode == 2
    assert "requires --kv-cache-strategy dynamic" in result.stderr

    runtime_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runtime_plan["inputs"]["kv_cache_strategy"] = "dynamic"
    runtime_plan["inputs"]["runtime"] = "vllm"
    plan_path.write_text(json.dumps(runtime_plan), encoding="utf-8")
    result = _run_fakegpu(
        "calibrate",
        "sample-transformers",
        str(plan_path),
    )
    assert result.returncode == 2
    assert "requires a generic serving runtime plan" in result.stderr

    result = _run_fakegpu(
        "calibrate",
        "emit-serving-sample",
        str(plan_path),
        "--prefill-peak-bytes",
        "-1",
        "--decode-peak-bytes",
        "1200",
        "--metric",
        "nvml.process_family_peak_bytes",
        "--framework",
        "vllm",
        "--framework-version",
        "0.10.2",
    )
    assert result.returncode == 2
    assert "prefill peak must be a positive integer" in result.stderr


def test_top_level_help_names_builtin_commands() -> None:
    result = _run_fakegpu("--help")
    assert result.returncode == 0
    for command in BUILTIN_COMMANDS:
        assert f"fakegpu {command}" in result.stdout
        command_result = _run_fakegpu(command, "--help")
        assert command_result.returncode == 0, (
            command,
            command_result.stderr,
        )

    for readme_path in README_PATHS:
        readme = (ROOT / readme_path).read_text(encoding="utf-8")
        documented_commands = set(
            re.findall(
                r"^\| `fakegpu ([a-z0-9-]+)` \|",
                readme,
                flags=re.MULTILINE,
            )
        )
        assert documented_commands == set(BUILTIN_COMMANDS), readme_path


def test_readmes_document_supported_use_cases() -> None:
    use_case_commands = {
        "estimate-llm",
        "estimate-diffusion",
        "preflight",
        "demo",
        "validate",
        "metrics",
        "plan-serving",
        "plan-training",
        "analyze-repo",
        "analyze-kernel",
        "capabilities",
        "simulate-topology",
        "replay-trace",
        "bandwidth",
        "calibrate",
    }
    assert use_case_commands <= set(BUILTIN_COMMANDS)

    for readme_path in README_PATHS:
        readme = (ROOT / readme_path).read_text(encoding="utf-8")
        for language_path in README_PATHS:
            assert f"({language_path})" in readme, (
                readme_path,
                language_path,
            )

        section = _anchored_readme_section(readme, "use-cases")
        table_rows = [
            line for line in section.splitlines() if line.startswith("|")
        ]
        assert len(table_rows) == 11, readme_path
        for command in use_case_commands:
            assert re.search(
                rf"`{re.escape(command)}(?:`| )",
                section,
            ), (readme_path, command)


def test_readmes_match_memory_validation_evidence() -> None:
    evidence = json.loads(MEMORY_EVIDENCE_PATH.read_text(encoding="utf-8"))
    revision = evidence["source_revision"]
    groups = {
        group["id"]: group for group in evidence["evidence_groups"]
    }

    controlled = groups["controlled_aten"]
    controlled_error = float(
        controlled[
            "published_maximum_absolute_percentage_error_percent"
        ]
    )
    inference = groups["qwen3_8b_inference"]["measurements"]
    inference_errors = [
        abs(item["predicted_bytes"] - item["observed_bytes"])
        / item["observed_bytes"]
        * 100
        for item in inference
    ]
    sft_errors = [
        item["published_absolute_percentage_error_percent"]
        for item in groups["qwen_sft"]["measurements"]
    ]
    qlora_errors = [
        item["published_absolute_percentage_error_percent"]
        for item in groups["qwen_qlora"]["measurements"]
    ]
    expected_claims = {
        str(controlled["workload_count"]),
        str(controlled["observation_count"]),
        f"{controlled_error:.2f}%",
        f"{100 - controlled_error:.2f}%",
        *(f"{error:.4f}%" for error in inference_errors),
        *(f"{100 - error:.4f}%" for error in inference_errors),
        f"{min(sft_errors):.3f}%–{max(sft_errors):.3f}%",
        f"{100 - max(sft_errors):.3f}%–{100 - min(sft_errors):.3f}%",
        f"{min(qlora_errors):.3f}%–{max(qlora_errors):.3f}%",
        (
            f"{100 - max(qlora_errors):.3f}%–"
            f"{100 - min(qlora_errors):.3f}%"
        ),
    }

    for readme_path in README_PATHS:
        readme = (ROOT / readme_path).read_text(encoding="utf-8")
        section = _anchored_readme_section(
            readme,
            "memory-estimation-evidence",
        )
        evidence_rows = [
            line for line in section.splitlines() if line.startswith("|")
        ]
        assert len(evidence_rows) == 6, readme_path
        for claim in expected_claims:
            assert claim in section, (readme_path, claim)
        assert "tests/data/memory_validation_evidence.json" in section

        for group in groups.values():
            source_url = (
                "https://github.com/FanBB2333/FakeGPU/blob/"
                f"{revision}/{group['source_path']}"
                f"#{group['source_anchor']}"
            )
            assert source_url in readme, (readme_path, group["id"])


def test_readmes_report_research_reliability_scope() -> None:
    profiles = load_profiles()
    compute_capability_count = len(
        {profile.compute_capability for profile in profiles.values()}
    )
    capabilities = json.loads(
        (
            ROOT
            / "fakegpu"
            / "data"
            / "native_api_capabilities.json"
        ).read_text(encoding="utf-8")
    )
    policy_enforced_count = sum(
        bool(item.get("policy_enforced"))
        for item in capabilities["apis"]
    )
    expected_terms = {
        "scripts/test.sh",
        "223",
        str(len(profiles)),
        str(compute_capability_count),
        str(len(capabilities["groups"])),
        str(len(capabilities["apis"])),
        str(policy_enforced_count),
        "GPU-validated",
        "CPU-validated",
        "Modeled",
        "Planned",
        "Qwen3-8B",
        "LoRA",
        "QLoRA",
        "FSDP/FSDP2",
        "ZeRO",
        "MoE",
        "continuous batching",
        "chunked prefill",
        "prefix caching",
        "speculative decoding",
        "--draft-model-dir",
        "--speculative-tokens",
        "--speculative-acceptance-rate",
        "workload_signature",
        "observe-serving",
        "collect-serving",
        "2.94",
        "false-safe",
        "5%",
        "calibrate verify",
        "research_validation.yaml",
        "--kv-cache-strategy",
        "--runtime",
        "--vllm-kv-cache-memory-bytes",
        "--vllm-non-kv-cache-memory-bytes",
        "--model-dir",
        "PixArt-Sigma",
        "Flux",
        "uncalibrated",
        "https://huggingface.co/docs/transformers/kv_cache",
        "https://huggingface.co/docs/transformers/assisted_decoding",
        "https://arxiv.org/abs/2211.17192",
        "https://docs.vllm.ai/en/stable/",
    }

    assert len(profiles) == 82
    assert compute_capability_count == 15
    assert len(capabilities["groups"]) == 5
    assert len(capabilities["apis"]) == 26
    assert policy_enforced_count == 24
    smi_terms = {
        "FAKEGPU_SMI_STATE_DIR",
        "--query-gpu",
        "--query-compute-apps",
        "--query-runtime",
        "native.kernel_launches",
        "fakegpu metrics",
        "/metrics",
        "/healthz",
        "/api/v1/history",
        "--max-process-series",
    }

    for readme_path in README_PATHS:
        readme = (ROOT / readme_path).read_text(encoding="utf-8")
        section = _anchored_readme_section(readme, "llm-reliability")
        for term in expected_terms:
            assert term in section, (readme_path, term)
        for term in smi_terms:
            assert term in readme, (readme_path, term)


def test_readmes_report_reproducible_research_scenario_effects() -> None:
    kv_short = estimate_kv_cache_memory(
        num_hidden_layers=32,
        num_key_value_heads=8,
        head_dim=128,
        batch_size=1,
        prompt_tokens=4096,
        element_bytes=2,
    )
    kv_long = estimate_kv_cache_memory(
        num_hidden_layers=32,
        num_key_value_heads=8,
        head_dim=128,
        batch_size=1,
        prompt_tokens=32768,
        element_bytes=2,
    )
    attention_cache_architectures = [
        estimate_kv_cache_memory(
            num_hidden_layers=32,
            num_key_value_heads=kv_heads,
            head_dim=128,
            batch_size=8,
            prompt_tokens=4096,
            element_bytes=2,
            strategy="paged",
            block_tokens=16,
            elements_per_token_per_layer=elements_per_layer,
            cache_layout=(
                "compressed_latent_and_rope"
                if elements_per_layer is not None
                else "separate_key_value"
            ),
        )
        for kv_heads, elements_per_layer in (
            (32, None),
            (8, None),
            (1, None),
            (1, 576),
        )
    ]
    serving_unshared = estimate_serving_kv_pool(
        num_hidden_layers=32,
        num_key_value_heads=8,
        head_dim=128,
        active_sequences=8,
        prompt_tokens=4096,
        generated_tokens=257,
        element_bytes=2,
        strategy="paged",
        block_tokens=16,
    )
    serving_shared = estimate_serving_kv_pool(
        num_hidden_layers=32,
        num_key_value_heads=8,
        head_dim=128,
        active_sequences=8,
        prompt_tokens=4096,
        generated_tokens=257,
        element_bytes=2,
        strategy="paged",
        shared_prefix_tokens=1024,
        block_tokens=16,
    )
    mixed_requests = [
        {
            "id": "chat-a",
            "prompt_tokens": 4096,
            "generated_tokens": 257,
        },
        {
            "id": "chat-b",
            "prompt_tokens": 8192,
            "generated_tokens": 513,
        },
        {
            "id": "completion",
            "prompt_tokens": 2048,
            "generated_tokens": 1025,
        },
    ]
    mixed_unshared = estimate_serving_request_kv_pool(
        mixed_requests,
        num_hidden_layers=32,
        num_key_value_heads=8,
        head_dim=128,
        element_bytes=2,
        strategy="paged",
        block_tokens=16,
    )
    mixed_shared = estimate_serving_request_kv_pool(
        [
            {
                **request,
                **(
                    {
                        "prefix_group": "system",
                        "shared_prefix_tokens": 1024,
                    }
                    if request["id"].startswith("chat-")
                    else {}
                ),
            }
            for request in mixed_requests
        ],
        num_hidden_layers=32,
        num_key_value_heads=8,
        head_dim=128,
        element_bytes=2,
        strategy="paged",
        block_tokens=16,
    )
    replicated = estimate_training_plan(
        {
            "sharding_strategy": "replicated",
            "world_size": 4,
            "mixed_precision": "bf16",
            "activation_checkpointing": True,
        },
        parameter_bytes=16_000_000_000,
        activation_bytes=8_000_000_000,
    )
    full_shard = estimate_training_plan(
        {
            "sharding_strategy": "full_shard",
            "world_size": 4,
            "mixed_precision": "bf16",
            "activation_checkpointing": True,
        },
        parameter_bytes=16_000_000_000,
        activation_bytes=8_000_000_000,
    )
    sd15_resident = estimate_diffusion_generation(
        "stable-diffusion-v1-5"
    )
    sd15_offload = estimate_diffusion_generation(
        "stable-diffusion-v1-5",
        offload="model",
    )
    sdxl_batch = estimate_diffusion_generation(
        "stable-diffusion-xl-base-1.0",
        batch_size=4,
    )
    sdxl_sliced = estimate_diffusion_generation(
        "stable-diffusion-xl-base-1.0",
        batch_size=4,
        vae_slicing=True,
    )
    sdxl_large = estimate_diffusion_generation(
        "stable-diffusion-xl-base-1.0",
        height=2048,
        width=2048,
    )
    sdxl_tiled = estimate_diffusion_generation(
        "stable-diffusion-xl-base-1.0",
        height=2048,
        width=2048,
        vae_tiling=True,
    )
    pixart_eager = estimate_diffusion_generation(
        "pixart-sigma-xl-2-1024-ms",
        attention_backend="eager",
        offload="model",
    )
    pixart_sdpa = estimate_diffusion_generation(
        "pixart-sigma-xl-2-1024-ms",
        attention_backend="sdpa",
        offload="model",
    )

    def phase_peak(report: dict, phase_name: str) -> int:
        return next(
            int(phase["peak_bytes"])
            for phase in report["memory_timeline"]["phases"]
            if phase["phase"] == phase_name
        )

    def gib(value: int) -> str:
        return f"{value / 2**30:.2f}"

    expected_effects = {
        (
            f"{gib(kv_short['prefill']['allocated_bytes'])} → "
            f"{gib(kv_long['prefill']['allocated_bytes'])} GiB"
        ),
        (
            f"{gib(attention_cache_architectures[0]['prefill']['allocated_bytes'])} / "
            f"{gib(attention_cache_architectures[1]['prefill']['allocated_bytes'])} / "
            f"{gib(attention_cache_architectures[2]['prefill']['allocated_bytes'])} / "
            f"{attention_cache_architectures[3]['prefill']['allocated_bytes'] / 2**30:.3f} GiB"
        ),
        (
            f"{gib(serving_unshared['decode']['allocated_bytes'])} → "
            f"{gib(serving_shared['decode']['allocated_bytes'])} GiB"
        ),
        (
            f"{gib(mixed_unshared['decode']['allocated_bytes'])} → "
            f"{gib(mixed_shared['decode']['allocated_bytes'])} GiB"
        ),
        (
            f"{gib(replicated['estimated_rank_peak_bytes'])} → "
            f"{gib(full_shard['estimated_rank_peak_bytes'])} GiB"
        ),
        (
            f"{gib(sd15_resident['memory_timeline']['peak_bytes'])} → "
            f"{gib(sd15_offload['memory_timeline']['peak_bytes'])} GiB"
        ),
        (
            f"{gib(sdxl_batch['memory_timeline']['peak_bytes'])} → "
            f"{gib(sdxl_sliced['memory_timeline']['peak_bytes'])} GiB"
        ),
        (
            f"{gib(sdxl_large['memory_timeline']['peak_bytes'])} → "
            f"{gib(sdxl_tiled['memory_timeline']['peak_bytes'])} GiB"
        ),
        (
            f"{gib(phase_peak(pixart_eager, 'denoise'))} → "
            f"{gib(phase_peak(pixart_sdpa, 'denoise'))} GiB"
        ),
    }
    expected_terms = {
        "estimate-diffusion",
        "plan-serving",
        "fakegpu.serving_requests.v1",
        "--requests",
        "--prefill-concurrency",
        "stable-diffusion-v1-5",
        "stable-diffusion-xl-base-1.0",
        "pixart-sigma-xl-2-1024-ms",
        "research_validation.yaml",
        "CPU-validated",
        "Modeled",
        *expected_effects,
    }
    for readme_path in README_PATHS:
        readme = (ROOT / readme_path).read_text(encoding="utf-8")
        for term in expected_terms:
            assert term in readme, (readme_path, term)
