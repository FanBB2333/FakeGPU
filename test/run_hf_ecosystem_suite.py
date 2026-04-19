from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "test-results" / "hf-ecosystem"

SUITES = [
    {
        "name": "hf_trainer_baseline",
        "command": [sys.executable, "-m", "pytest", "-q", "test/test_hf_trainer.py"],
    },
    {
        "name": "hf_trainer_cuda",
        "command": [sys.executable, "-m", "pytest", "-q", "test/test_hf_trainer_cuda.py"],
    },
    {
        "name": "autocast_dtype",
        "command": [sys.executable, "-m", "pytest", "-q", "test/test_error_dtype_autocast.py"],
    },
    {
        "name": "torch_patch_proof_regressions",
        "command": [sys.executable, "-m", "pytest", "-q", "test/test_torch_patch_proof_regressions.py"],
    },
    {
        "name": "peft_lora",
        "command": [sys.executable, "-m", "pytest", "-q", "test/test_peft_lora.py"],
    },
    {
        "name": "trl_sft",
        "command": [sys.executable, "-m", "pytest", "-q", "test/test_trl_sft.py"],
    },
    {
        "name": "trl_dpo",
        "command": [sys.executable, "-m", "pytest", "-q", "test/test_trl_dpo.py"],
    },
]


def _status_for_output(proc: subprocess.CompletedProcess[str], combined_output: str) -> str:
    if proc.returncode != 0:
        return "failed"
    lowered = combined_output.lower()
    if " failed" in lowered or "\nfailed" in lowered:
        return "failed"
    if " skipped" in lowered or "\nskipped" in lowered or "\nsss" in lowered:
        return "skipped"
    return "passed"


def _summary_line(output: str) -> str:
    preferred_markers = (" passed", " failed", " skipped", " error", " errors", " warnings")
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        lowered = stripped.lower()
        if stripped and any(marker in lowered for marker in preferred_markers):
            return stripped
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["XONSH_HISTORY_BACKEND"] = "dummy"
    env.setdefault("FAKEGPU_TERMINAL_REPORT", "0")

    summary: dict[str, object] = {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "python": sys.executable,
        "suites": [],
    }

    overall_exit = 0

    for suite in SUITES:
        proc = subprocess.run(
            suite["command"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        combined = proc.stdout
        if proc.stderr:
            combined = f"{combined}\n{proc.stderr}" if combined else proc.stderr

        log_path = OUTPUT_DIR / f"{suite['name']}.log"
        log_path.write_text(combined, encoding="utf-8")

        status = _status_for_output(proc, combined)
        if status == "failed":
            overall_exit = 1

        summary["suites"].append(
            {
                "name": suite["name"],
                "command": suite["command"],
                "exit_code": proc.returncode,
                "status": status,
                "summary_line": _summary_line(combined),
                "log_path": str(log_path.relative_to(REPO_ROOT)),
            }
        )

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return overall_exit


if __name__ == "__main__":
    raise SystemExit(main())
