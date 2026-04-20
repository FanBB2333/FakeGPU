from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent


def _run_snippet(
    code: str,
    *,
    extra_env: dict[str, str] | None = None,
    timeout: float = 45.0,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(TEST_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["FAKEGPU_TERMINAL_REPORT"] = "0"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def test_memory_stats_keys() -> None:
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=1)

        import torch

        stats = torch.cuda.memory_stats()
        required = {
            "active_bytes.all.current",
            "active_bytes.all.peak",
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
        }
        assert required.issubset(stats)
        assert all(isinstance(stats[key], int) for key in required)
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_torchtune_style_single_device_finetune_smoke() -> None:
    code = textwrap.dedent(
        """
        from hf_test_utils import patch_fakegpu

        patch_fakegpu(profile="a100", device_count=1)

        import torch
        from transformers import AutoModelForCausalLM

        config = AutoModelForCausalLM.from_config.__globals__["AutoConfig"].for_model(
            "qwen2",
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            use_cache=False,
        )
        model = AutoModelForCausalLM.from_config(config).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, config.vocab_size, (2, 16), device="cuda")
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        stats = torch.cuda.memory_stats()
        assert next(model.parameters()).device.type == "cuda"
        assert stats["allocated_bytes.all.peak"] >= stats["allocated_bytes.all.current"] >= 0
        assert isinstance(loss.item(), float)
        print("ok")
        """
    )
    proc = _run_snippet(code)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
