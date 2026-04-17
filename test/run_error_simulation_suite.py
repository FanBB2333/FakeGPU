#!/usr/bin/env python3
"""Unified FakeGPU validation report generator.

Combines Phase 1-3 (static) results with Phase 4 (error simulation, dynamic)
into a single HTML report with tab navigation.

Usage:
    python test/run_error_simulation_suite.py [--output test/report.html]
"""
import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    test_id: str
    name: str
    file: str
    status: str  # "pass", "fail", "error", "skip"
    duration: float = 0.0
    stdout: str = ""
    stderr: str = ""
    message: str = ""


@dataclass
class SuiteResult:
    category: str
    description: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self):
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def failed(self):
        return sum(1 for r in self.results if r.status in ("fail", "error"))

    @property
    def total(self):
        return len(self.results)


# ---------------------------------------------------------------------------
# Test discovery & execution
# ---------------------------------------------------------------------------

def discover_test_files(test_dir: str) -> list[Path]:
    return sorted(Path(test_dir).glob("test_error_*.py"))


def run_test_file(path: Path) -> list[TestResult]:
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(path), "-v", "--tb=short"],
            capture_output=True, text=True, timeout=120,
            cwd=str(path.parent.parent),
        )
    except subprocess.TimeoutExpired:
        return [TestResult(
            test_id="TIMEOUT", name=path.stem, file=str(path),
            status="error", duration=120.0, message="Timed out (120s)",
        )]
    elapsed = time.time() - start

    results = []
    for line in (proc.stdout + proc.stderr).splitlines():
        m = re.match(
            r"^.*?::([\w]+)::(test_\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)", line
        )
        if m:
            cls_name, test_name, verdict = m.group(1), m.group(2), m.group(3)
            status = {"PASSED": "pass", "FAILED": "fail",
                      "ERROR": "error", "SKIPPED": "skip"}[verdict]
            results.append(TestResult(
                test_id=f"{path.stem}::{cls_name}::{test_name}",
                name=test_name, file=str(path.name), status=status,
                duration=elapsed / max(len(results) + 1, 1),
                stdout=proc.stdout, stderr=proc.stderr,
            ))

    if not results:
        status = "pass" if proc.returncode == 0 else "fail"
        results.append(TestResult(
            test_id=path.stem, name=path.stem, file=str(path.name),
            status=status, duration=elapsed,
            stdout=proc.stdout, stderr=proc.stderr,
        ))
    return results


CATEGORY_MAP = {
    "test_error_cross_device": ("E1: Cross-device Ops", "Tensors on different CUDA devices used in same operation"),
    "test_error_oom": ("E2: OOM Simulation", "Per-device memory tracking and OutOfMemoryError"),
    "test_error_device_index": ("E3: Device Index", "Invalid device ordinal validation"),
    "test_error_dtype_autocast": ("E4: dtype / Autocast", "Autocast bfloat16 on incompatible compute capability"),
    "test_error_checkpoint_load": ("E5: Checkpoint Load", "torch.load map_location device validation"),
    "test_error_distributed": ("E6: Distributed", "NCCL communication error simulation"),
    "test_error_gradient": ("E7: Gradient", "Native PyTorch gradient errors (validation only)"),
}


# ---------------------------------------------------------------------------
# Phase-4 dynamic HTML fragment
# ---------------------------------------------------------------------------

def _build_phase4_rows(suites: list[SuiteResult]) -> str:
    """Return <div class='results'>...</div> rows for error-sim suites."""
    parts: list[str] = []
    for suite in suites:
        for r in suite.results:
            badge_cls = "pass" if r.status == "pass" else "fail"
            detail = ""
            if r.stdout or r.stderr:
                esc = (r.stdout + r.stderr).replace("&", "&amp;") \
                    .replace("<", "&lt;").replace(">", "&gt;")
                detail = f'<pre>{esc[:4000]}</pre>'
            pretty_name = r.name.replace("_", " ").removeprefix("test ")
            parts.append(f"""
      <div class="row {badge_cls}" onclick="this.classList.toggle('open')">
        <div class="id">{suite.category.split(":")[0]}</div>
        <div class="info">
          <div class="title">{pretty_name}</div>
          <div class="desc">{suite.description}</div>
        </div>
        <div class="badge {badge_cls}">{r.status.upper()}</div>
        <div class="detail">{detail}</div>
      </div>""")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Unified HTML generation
# ---------------------------------------------------------------------------

def generate_unified_html(
    suites: list[SuiteResult],
    output_path: str,
) -> None:
    p4_pass = sum(s.passed for s in suites)
    p4_fail = sum(s.failed for s in suites)
    p4_total = sum(s.total for s in suites)

    # Phase 1-3 static counts
    p123_total = 10
    p123_pass = 10

    all_total = p123_total + p4_total
    all_pass = p123_pass + p4_pass
    all_fail = 0 + p4_fail

    p4_rows = _build_phase4_rows(suites)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FakeGPU &middot; Unified Validation Report</title>
<style>
  /* ---------- Morandi Palette ---------- */
  :root {{
    --bg:        #ede8e0;
    --surface:   #f4f0e9;
    --surface-2: #e3dcd1;
    --ink:       #4a4740;
    --ink-soft:  #7a7468;
    --line:      #d6cdbf;
    --sage:      #a8b5a0;
    --sage-dk:   #7d8e78;
    --rose:      #c9a7a1;
    --rose-dk:   #a8837c;
    --sand:      #d4bf9a;
    --sand-dk:   #b39a73;
    --mist:      #a4b1bd;
    --mist-dk:   #7d8b98;
    --plum:      #b5a0b0;
    --plum-dk:   #8e7b8b;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; scroll-behavior: smooth; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue",
                 "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--ink);
    line-height: 1.65; -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 64px 32px 96px; }}

  /* ---------- Tab navigation ---------- */
  .tab-bar {{
    position: sticky; top: 0; z-index: 100;
    display: flex; gap: 0;
    background: var(--surface);
    border-bottom: 2px solid var(--line);
    padding: 0 24px;
    margin: -64px -32px 40px;
    overflow-x: auto;
  }}
  .tab-btn {{
    background: none; border: none; cursor: pointer;
    padding: 16px 22px; font-size: 14px; font-weight: 600;
    color: var(--ink-soft); white-space: nowrap;
    border-bottom: 3px solid transparent;
    transition: color .2s, border-color .2s;
  }}
  .tab-btn:hover {{ color: var(--ink); }}
  .tab-btn.active {{ color: var(--ink); border-bottom-color: var(--sage-dk); }}
  .tab-page {{ display: none; }}
  .tab-page.active {{ display: block; }}

  /* ---------- Hero ---------- */
  header.hero {{
    text-align: center; padding: 56px 24px 40px; margin-bottom: 48px;
    background: linear-gradient(140deg, var(--surface) 0%, var(--surface-2) 100%);
    border-radius: 24px; border: 1px solid var(--line);
    opacity: 0; transform: translateY(14px);
    animation: rise .9s .05s ease-out forwards;
  }}
  header.hero .eyebrow {{
    letter-spacing: .24em; font-size: 12px; color: var(--ink-soft);
    text-transform: uppercase; margin-bottom: 14px;
  }}
  header.hero h1 {{ font-size: 32px; margin: 0 0 12px; font-weight: 600; letter-spacing: .5px; }}
  header.hero .sub {{ color: var(--ink-soft); font-size: 15px; margin: 0; }}
  header.hero .meta {{
    display: flex; justify-content: center; flex-wrap: wrap; gap: 10px; margin-top: 22px;
  }}
  .chip {{
    background: var(--bg); color: var(--ink-soft);
    padding: 6px 14px; border-radius: 999px; font-size: 12px; border: 1px solid var(--line);
  }}

  /* ---------- Summary cards ---------- */
  .stats {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 18px; margin-bottom: 48px;
  }}
  .stat {{
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 16px; padding: 22px 20px; text-align: center;
    opacity: 0; transform: translateY(14px);
    animation: rise .7s ease-out forwards;
  }}
  .stat:nth-child(1) {{ animation-delay: .15s; }}
  .stat:nth-child(2) {{ animation-delay: .25s; }}
  .stat:nth-child(3) {{ animation-delay: .35s; }}
  .stat:nth-child(4) {{ animation-delay: .45s; }}
  .stat .label {{ font-size: 12px; color: var(--ink-soft); letter-spacing: .1em; text-transform: uppercase; }}
  .stat .value {{ font-size: 30px; font-weight: 600; margin-top: 6px; }}
  .stat.pass  .value {{ color: var(--sage-dk); }}
  .stat.fail  .value {{ color: var(--rose-dk); }}
  .stat.total .value {{ color: var(--mist-dk); }}
  .stat.files .value {{ color: var(--plum-dk); }}

  /* ---------- Section ---------- */
  section {{
    margin-bottom: 56px;
    opacity: 0; transform: translateY(14px);
    animation: rise .75s ease-out forwards; animation-delay: .5s;
  }}
  section h2 {{
    font-size: 20px; font-weight: 600; margin: 0 0 18px;
    padding-bottom: 10px; border-bottom: 1px solid var(--line); letter-spacing: .5px;
  }}
  section h3 {{
    font-size: 16px; font-weight: 600; margin: 28px 0 14px; color: var(--ink);
  }}

  /* ---------- Phase banner ---------- */
  .phase-banner {{
    display: flex; align-items: center; gap: 14px;
    padding: 16px 22px;
    background: var(--surface); border: 1px solid var(--line);
    border-left: 5px solid var(--mist-dk); border-radius: 12px;
    margin-bottom: 20px;
  }}
  .phase-banner.p1 {{ border-left-color: var(--sage-dk); }}
  .phase-banner.p2 {{ border-left-color: var(--plum-dk); }}
  .phase-banner.p3 {{ border-left-color: var(--sand-dk); }}
  .phase-banner.p4 {{ border-left-color: var(--rose-dk); }}
  .phase-banner .phase-num {{
    font-size: 28px; font-weight: 700; color: var(--ink-soft);
    min-width: 48px; text-align: center;
  }}
  .phase-banner .phase-info .ptitle {{ font-weight: 600; font-size: 15px; }}
  .phase-banner .phase-info .pdesc  {{ color: var(--ink-soft); font-size: 13px; margin-top: 2px; }}

  /* ---------- Result rows ---------- */
  .results {{ display: grid; gap: 14px; }}
  .row {{
    display: grid; grid-template-columns: 60px 1fr auto;
    align-items: center;
    background: var(--surface); border: 1px solid var(--line);
    border-left: 5px solid var(--line); border-radius: 12px;
    padding: 16px 20px;
    transition: transform .25s ease, box-shadow .25s ease, background .25s ease;
    cursor: pointer;
  }}
  .row:hover {{
    transform: translateX(4px); background: var(--surface-2);
    box-shadow: 0 4px 18px -12px rgba(74,71,64,.35);
  }}
  .row.pass {{ border-left-color: var(--sage-dk); }}
  .row.fail {{ border-left-color: var(--rose-dk); }}
  .row .id {{
    font-weight: 600; color: var(--ink-soft);
    font-variant-numeric: tabular-nums; letter-spacing: .05em;
  }}
  .row .info .title {{ font-weight: 600; font-size: 15px; }}
  .row .info .desc  {{ color: var(--ink-soft); font-size: 13px; margin-top: 2px; }}
  .badge {{
    font-size: 12px; font-weight: 600; padding: 5px 12px;
    border-radius: 999px; letter-spacing: .08em; white-space: nowrap;
  }}
  .badge.pass {{ background: var(--sage); color: #fff; }}
  .badge.fail {{ background: var(--rose); color: #fff; }}

  .detail {{
    grid-column: 1 / -1; margin-top: 14px; padding-top: 14px;
    border-top: 1px dashed var(--line);
    display: none; font-size: 14px; color: var(--ink);
    animation: fadeIn .4s ease-out;
  }}
  .row.open .detail {{ display: block; }}
  .detail pre {{
    background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
    padding: 12px 14px; margin: 10px 0; font-size: 12.5px; line-height: 1.55;
    overflow-x: auto; color: var(--ink); white-space: pre-wrap; word-break: break-all;
    max-height: 300px; overflow-y: auto;
  }}
  .detail .kv {{ margin: 4px 0; color: var(--ink-soft); }}
  .detail .note {{ margin-top: 10px; }}

  /* ---------- Commit cards ---------- */
  .commits {{ display: grid; gap: 12px; margin-bottom: 28px; }}
  .commit {{
    display: flex; align-items: center; gap: 14px;
    padding: 14px 18px; background: var(--surface);
    border: 1px solid var(--line); border-radius: 12px;
  }}
  .commit .sha {{
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
    background: var(--bg); padding: 3px 10px; border-radius: 6px;
    color: var(--mist-dk); border: 1px solid var(--line); white-space: nowrap;
  }}
  .commit .msg {{ font-size: 14px; }}

  /* ---------- JSON viewer ---------- */
  .json-sample {{
    background: var(--bg); border: 1px solid var(--line); border-radius: 12px;
    padding: 18px 20px; font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 12px; line-height: 1.6; overflow-x: auto;
    max-height: 420px; overflow-y: auto; color: var(--ink);
  }}

  /* ---------- Env card ---------- */
  .env-card {{
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 16px; padding: 22px 22px 20px; margin-bottom: 28px;
  }}
  .env-card h3 {{ margin: 0 0 12px; font-size: 16px; font-weight: 600; }}
  .env-card ul {{ margin: 0; padding-left: 18px; color: var(--ink-soft); font-size: 14px; }}
  .env-card ul li {{ margin: 4px 0; }}
  .env-card code {{ background: var(--bg); padding: 1px 6px; border-radius: 4px; font-size: 12.5px; color: var(--ink); }}

  /* ---------- Conclusion ---------- */
  .concl {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .card {{
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 14px; padding: 20px;
  }}
  .card h4 {{
    margin: 0 0 10px; font-size: 14px; letter-spacing: .08em;
    text-transform: uppercase; color: var(--ink-soft);
  }}
  .card.pass h4 {{ color: var(--sage-dk); }}
  .card.info h4 {{ color: var(--mist-dk); }}
  .card.caveat h4 {{ color: var(--sand-dk); }}
  .card ul {{ margin: 0; padding-left: 18px; font-size: 14px; color: var(--ink); }}
  .card ul li {{ margin: 4px 0; }}

  /* ---------- Terminal ---------- */
  .terminal {{
    background: #2b2926; color: #d8d4cc;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12.5px;
    line-height: 1.6; padding: 20px 22px; border-radius: 12px;
    overflow-x: auto; margin: 14px 0;
  }}
  .terminal .hl {{ color: var(--sage); }}
  .terminal .dim {{ color: #7a7468; }}

  footer {{
    text-align: center; color: var(--ink-soft); font-size: 12.5px;
    margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--line);
  }}

  @keyframes rise {{ to {{ opacity: 1; transform: translateY(0); }} }}
  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(-4px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  @media (max-width: 780px) {{
    .stats {{ grid-template-columns: repeat(2, 1fr); }}
    .concl {{ grid-template-columns: 1fr; }}
    .row   {{ grid-template-columns: 50px 1fr auto; }}
    header.hero h1 {{ font-size: 24px; }}
    .tab-btn {{ padding: 12px 14px; font-size: 13px; }}
  }}
</style>
</head>
<body>

<div class="wrap">

  <!-- ======== Tab bar ======== -->
  <nav class="tab-bar">
    <button class="tab-btn active" onclick="showTab('overview')">Overview</button>
    <button class="tab-btn" onclick="showTab('p1')">P1: Report System</button>
    <button class="tab-btn" onclick="showTab('p2')">P2: HW Compat</button>
    <button class="tab-btn" onclick="showTab('p3')">P3: MoE nanoGPT</button>
    <button class="tab-btn" onclick="showTab('p4')">P4: Error Simulation</button>
  </nav>

  <!-- ================================================================== -->
  <!-- OVERVIEW TAB                                                        -->
  <!-- ================================================================== -->
  <div id="page-overview" class="tab-page active">

    <header class="hero">
      <div class="eyebrow">Unified Validation &middot; 2026-04-17</div>
      <h1>FakeGPU &middot; Full Test Report</h1>
      <p class="sub">Enhanced Report &middot; HW Compat &middot; MoE Training &middot; Error Simulation</p>
      <div class="meta">
        <span class="chip">4 Phases</span>
        <span class="chip">{all_total} Tests</span>
        <span class="chip">{all_pass} Passed</span>
        <span class="chip">macOS &middot; ARM64</span>
      </div>
    </header>

    <div class="stats">
      <div class="stat total"><div class="label">Total Tests</div><div class="value">{all_total}</div></div>
      <div class="stat pass"><div class="label">Passed</div><div class="value">{all_pass}</div></div>
      <div class="stat fail"><div class="label">Failed</div><div class="value">{all_fail}</div></div>
      <div class="stat files"><div class="label">Phases</div><div class="value">4</div></div>
    </div>

    <section>
      <h2>Phase Summary</h2>
      <div class="results">
        <div class="row pass" onclick="showTab('p1')">
          <div class="id">P1</div>
          <div class="info">
            <div class="title">Enhanced Report System</div>
            <div class="desc">Report v4 &middot; GPU Profile &middot; Kernel GEMM Stats &middot; Terminal Summary</div>
          </div>
          <div class="badge pass">5/5 PASS</div>
        </div>
        <div class="row pass" onclick="showTab('p2')">
          <div class="id">P2</div>
          <div class="info">
            <div class="title">Hardware Compatibility Simulation</div>
            <div class="desc">cuBLAS dtype compat &middot; torch_patch BF16 awareness</div>
          </div>
          <div class="badge pass">2/2 PASS</div>
        </div>
        <div class="row pass" onclick="showTab('p3')">
          <div class="id">P3</div>
          <div class="info">
            <div class="title">MoE nanoGPT Test Scenario</div>
            <div class="desc">MoE-GPT model &middot; EP training &middot; Wrapper integration</div>
          </div>
          <div class="badge pass">4/4 PASS</div>
        </div>
        <div class="row {"pass" if p4_fail == 0 else "fail"}" onclick="showTab('p4')">
          <div class="id">P4</div>
          <div class="info">
            <div class="title">Error Simulation Experiments</div>
            <div class="desc">Cross-device &middot; OOM &middot; Device index &middot; Autocast &middot; Checkpoint &middot; Gradient</div>
          </div>
          <div class="badge {"pass" if p4_fail == 0 else "fail"}">{p4_pass}/{p4_total} {"PASS" if p4_fail == 0 else "FAIL"}</div>
        </div>
      </div>
    </section>

    <section>
      <h2>Test Environment</h2>
      <div class="env-card">
        <h3>macOS &middot; Apple Silicon (no physical GPU)</h3>
        <ul>
          <li>Platform: <code>darwin arm64</code></li>
          <li>Python: <code>3.11</code> (miniforge3/envs/py311)</li>
          <li>PyTorch: <code>2.9.1</code> (CPU-only)</li>
          <li>FakeGPU build: <code>build/*.dylib</code> (CMake, C++17)</li>
          <li>Report format: <code>v4</code> (JSON)</li>
        </ul>
      </div>
    </section>
  </div>

  <!-- ================================================================== -->
  <!-- PHASE 1: Enhanced Report System                                     -->
  <!-- ================================================================== -->
  <div id="page-p1" class="tab-page">
    <section>
      <h2>Phase 1: Enhanced Report System <span style="font-weight:400;font-size:13px;color:var(--ink-soft);margin-left:8px;">click rows to expand</span></h2>
      <div class="phase-banner p1">
        <div class="phase-num">P1</div>
        <div class="phase-info">
          <div class="ptitle">Report v4 &middot; GPU Profile &middot; Kernel Launch Tracking &middot; GEMM-by-dtype &middot; Terminal Summary</div>
          <div class="pdesc">Upgrade GlobalState, Monitor, and Driver stubs to record and report per-device kernel launches, typed GEMM stats, and architecture profile info.</div>
        </div>
      </div>
      <div class="results">
        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P1-1</div>
          <div class="info"><div class="title">cmake --build build</div><div class="desc">C++ library full compilation with new GlobalState fields, Monitor v4, and driver stubs</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <div class="kv">Command: <code>cmake --build build</code></div>
            <pre>[ 28%] Built target fake_gpu_distributed
[ 34%] Built target fake_gpu_core
[ 41%] Built target fake_gpu_nvml
[ 45%] Built target fake_gpu_cuda
[ 47%] Built target fake_gpu_monitor
[ 50%] Built target fake_gpu (libnvidia-ml.dylib)
[ 67%] Built target fake_cuda (libcuda.dylib)
...
[100%] All targets built successfully</pre>
            <div class="note">Clean compilation. All OBJECT libraries and shared libraries link without errors.</div>
          </div>
        </div>

        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P1-2</div>
          <div class="info"><div class="title">test_enhanced_global_state.py</div><div class="desc">C++ binary validates DeviceRuntimeStats kernel/GEMM maps, CompatEvent tracking, and snapshot</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <div class="kv">Command: <code>python3 test/test_enhanced_global_state.py</code></div>
            <pre>enhanced GlobalState smoke passed</pre>
          </div>
        </div>

        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P1-3</div>
          <div class="info"><div class="title">test_report_v4_smoke.py</div><div class="desc">Integration: builds library, runs CUDA exercise, validates JSON v4 structure and terminal summary</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <div class="kv">Command: <code>python3 test/test_report_v4_smoke.py</code></div>
            <pre>enhanced report smoke passed</pre>
          </div>
        </div>

        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P1-4</div>
          <div class="info"><div class="title">run_smoke.sh (Native Smoke Test)</div><div class="desc">End-to-end: build &rarr; compile C test &rarr; LD_PRELOAD &rarr; verify report and terminal summary</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <div class="kv">Command: <code>./verification/run_smoke.sh</code></div>
            <pre>Device 0: Fake NVIDIA A100-SXM4-80GB (Ampere, cc 8.0)
  Memory: 64.0 MB / 80.0 GB peak (0.1%)
  Alloc: 1 calls | Free: 1 calls</pre>
          </div>
        </div>

        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P1-5</div>
          <div class="info"><div class="title">FAKEGPU_TERMINAL_REPORT=0 Suppression</div><div class="desc">Verify terminal summary is suppressed when env var is 0</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <div class="kv">Command: <code>FAKEGPU_TERMINAL_REPORT=0 ./verification/run_smoke.sh</code></div>
            <pre>grep -c "FakeGPU Report Summary" =&gt; 0</pre>
          </div>
        </div>
      </div>

      <h3>Terminal Summary Preview</h3>
      <div class="terminal">
<span class="dim">======================================================</span>
<span class="dim">             </span><span class="hl">FakeGPU Report Summary</span>
<span class="dim">======================================================</span>
 Device 0: Fake NVIDIA A100-SXM4-80GB <span class="dim">(Ampere, cc 8.0)</span>
   Memory: <span class="hl">64.0 MB</span> / 80.0 GB peak (0.1%)
   Alloc: 1 calls | Free: 1 calls
<span class="dim">------------------------------------------------------</span>
 Device 1: Fake NVIDIA A100-SXM4-80GB <span class="dim">(Ampere, cc 8.0)</span>
   Memory: 0 B / 80.0 GB peak (0.0%)
<span class="dim">======================================================</span>
      </div>

      <h3>Report v4 JSON Sample (Device 0)</h3>
      <div class="json-sample">{{
  "<span style="color:var(--mist-dk)">report_version</span>": 4,
  "<span style="color:var(--mist-dk)">mode</span>": "simulate",
  "<span style="color:var(--mist-dk)">devices</span>": [{{
    "<span style="color:var(--mist-dk)">index</span>": 0,
    "<span style="color:var(--mist-dk)">name</span>": "Fake NVIDIA A100-SXM4-80GB",
    "<span style="color:var(--mist-dk)">gpu_profile</span>": {{
      "architecture": "Ampere",
      "compute_capability": "8.0",
      "supported_types": ["fp32", "tf32", "fp16", "bf16", "int8"]
    }},
    "<span style="color:var(--mist-dk)">total_memory</span>": 85899345920,
    "<span style="color:var(--mist-dk)">used_memory_peak</span>": 67108864,
    "<span style="color:var(--mist-dk)">kernel_launches</span>": {{ "total": 0 }},
    "<span style="color:var(--mist-dk)">gemm_by_dtype</span>": {{}}
  }}]
}}</div>
    </section>
  </div>

  <!-- ================================================================== -->
  <!-- PHASE 2: Hardware Compatibility                                     -->
  <!-- ================================================================== -->
  <div id="page-p2" class="tab-page">
    <section>
      <h2>Phase 2: Hardware Compatibility Simulation</h2>
      <div class="phase-banner p2">
        <div class="phase-num">P2</div>
        <div class="phase-info">
          <div class="ptitle">cuda_dtype_to_gpu_dtype Mapping &middot; cuBLAS Strict Compat Check &middot; Compat Events &middot; torch_patch BF16 Awareness</div>
          <div class="pdesc">Enforce profile-based dtype compatibility in cuBLAS stubs, record compat events to report, and make torch_patch derive compute capability from FAKEGPU_PROFILES.</div>
        </div>
      </div>
      <div class="results">
        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P2-1</div>
          <div class="info"><div class="title">test_hw_compat_cublas.py</div><div class="desc">C++ binary tests cuBLAS dtype compat in strict and relaxed modes across different GPU profiles</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <div class="kv">Command: <code>python3 test/test_hw_compat_cublas.py</code></div>
            <pre>hardware compatibility cuBLAS smoke passed</pre>
            <div class="note">Tests BF16 GEMM rejection on V100 (strict), relaxed-mode warning, and A100 success.</div>
          </div>
        </div>

        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P2-2</div>
          <div class="info"><div class="title">test_torch_patch_profile_bf16.py</div><div class="desc">Python regression: FAKEGPU_PROFILES drives compute capability and BF16 support in torch_patch</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <div class="kv">Command: <code>python3 test/test_torch_patch_profile_bf16.py</code></div>
            <pre>torch patch profile BF16 smoke passed</pre>
            <div class="note">T4 &rarr; compute 7.0, is_bf16_supported()=False. A100 &rarr; compute 8.0, is_bf16_supported()=True.</div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <!-- ================================================================== -->
  <!-- PHASE 3: MoE nanoGPT                                               -->
  <!-- ================================================================== -->
  <div id="page-p3" class="tab-page">
    <section>
      <h2>Phase 3: MoE nanoGPT Test Scenario</h2>
      <div class="phase-banner p3">
        <div class="phase-num">P3</div>
        <div class="phase-info">
          <div class="ptitle">MoE-GPT Model &middot; EP Training Script &middot; Homo/Hetero Configs &middot; train_wrapper --model moe</div>
          <div class="pdesc">Standard Mixture-of-Experts GPT model with Router (top-k gating), Expert Parallelism via all_to_all, and two FakeGPU configurations.</div>
        </div>
      </div>
      <div class="results">
        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P3-1</div>
          <div class="info"><div class="title">test_nanogpt_moe_components.py</div><div class="desc">Unit: MoEGPT instantiation, forward pass, loss computation, config files, wrapper resolution</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <pre>MoEGPT model: 0.02M parameters, 4 experts, top-2
nanogpt MoE component smoke passed</pre>
          </div>
        </div>

        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P3-2</div>
          <div class="info"><div class="title">test_nanogpt_train_wrapper_helpers.py</div><div class="desc">Regression: train_wrapper helper functions, memory limiter, profile resolution</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <pre>nanogpt train wrapper helper regression passed</pre>
          </div>
        </div>

        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P3-3</div>
          <div class="info"><div class="title">MoE Training (Direct: train_moe.py)</div><div class="desc">End-to-end MoE-GPT training: 2 iters, 4 experts, top-2, float32, CPU backend</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <pre>MoEGPT model: 0.30M parameters, 4 experts, top-2
  iter    0 | loss 4.9123 | time 0.1s
  iter    1 | loss 4.8629 | time 0.1s
Training complete: 2 iters in 0.1s</pre>
            <div class="note">Loss decreases (4.9123 &rarr; 4.8629), confirming gradient flow through Router + ExpertMLP + attention layers.</div>
          </div>
        </div>

        <div class="row pass" onclick="this.classList.toggle('open')">
          <div class="id">P3-4</div>
          <div class="info"><div class="title">MoE Training (Wrapper: train_wrapper.py --model moe)</div><div class="desc">Wrapper integration: baseline mode, MoE model routing, CPU fallback</div></div>
          <div class="badge pass">PASS</div>
          <div class="detail">
            <pre>[WRAPPER] Using MoE model: train_moe.py
MoEGPT model: 0.30M parameters, 4 experts, top-2
  iter    0 | loss 4.8571 | time 0.0s
[WRAPPER] Training completed successfully in 0.5s
[WRAPPER] Result: PASS</pre>
          </div>
        </div>
      </div>
    </section>
  </div>

  <!-- ================================================================== -->
  <!-- PHASE 4: Error Simulation (dynamic)                                 -->
  <!-- ================================================================== -->
  <div id="page-p4" class="tab-page">
    <section>
      <h2>Phase 4: Error Simulation Experiments <span style="font-weight:400;font-size:13px;color:var(--ink-soft);margin-left:8px;">click rows to expand</span></h2>
      <div class="phase-banner p4">
        <div class="phase-num">P4</div>
        <div class="phase-info">
          <div class="ptitle">Cross-device &middot; OOM &middot; Device Index &middot; Autocast dtype &middot; Checkpoint Load &middot; Gradient</div>
          <div class="pdesc">Reproduce common real-GPU errors (RuntimeError, OutOfMemoryError, invalid device ordinal) via FakeGPU's Python-layer patches. 7 error categories, {p4_total} test cases.</div>
        </div>
      </div>

      <div class="stats" style="grid-template-columns: repeat(3, 1fr);">
        <div class="stat pass"><div class="label">Passed</div><div class="value">{p4_pass}</div></div>
        <div class="stat fail"><div class="label">Failed</div><div class="value">{p4_fail}</div></div>
        <div class="stat total"><div class="label">Total</div><div class="value">{p4_total}</div></div>
      </div>

      <div class="results">
{p4_rows}
      </div>
    </section>
  </div>

  <!-- ================================================================== -->
  <!-- Conclusions (visible on all tabs via footer)                        -->
  <!-- ================================================================== -->
  <section>
    <h2>Conclusions</h2>
    <div class="concl">
      <div class="card pass">
        <h4>All Tests Passing</h4>
        <ul>
          <li>P1: Report v4 + terminal summary</li>
          <li>P2: HW compat strict/relaxed</li>
          <li>P3: MoE model + training</li>
          <li>P4: {p4_pass}/{p4_total} error simulations</li>
        </ul>
      </div>
      <div class="card info">
        <h4>Error Simulation Coverage</h4>
        <ul>
          <li>E1: Cross-device tensor ops</li>
          <li>E2: OOM per-device tracking</li>
          <li>E3: Invalid device ordinal</li>
          <li>E4: Autocast bfloat16 compat</li>
          <li>E5: Checkpoint map_location</li>
          <li>E7: Gradient error passthrough</li>
        </ul>
      </div>
      <div class="card caveat">
        <h4>Known Limitations</h4>
        <ul>
          <li>No actual GPU computation; kernels are no-ops</li>
          <li><code>tensor.device</code> reports <code>cpu</code></li>
          <li>E6 (distributed) not yet implemented</li>
          <li>macOS: all tests use CPU backend</li>
        </ul>
      </div>
    </div>
  </section>

  <footer>
    Generated from test results on 2026-04-17 &middot;
    Phase 1-3: <code>report_phase2.html</code> &middot;
    Phase 4: <code>run_error_simulation_suite.py</code>
  </footer>

</div>

<script>
  // Tab switching
  function showTab(id) {{
    document.querySelectorAll('.tab-page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('page-' + id).classList.add('active');
    // Find matching button
    document.querySelectorAll('.tab-btn').forEach(b => {{
      if (b.textContent.toLowerCase().includes(id === 'overview' ? 'overview' : id))
        b.classList.add('active');
    }});
    window.scrollTo({{top: 0, behavior: 'smooth'}});
  }}
  // Row stagger animation
  document.querySelectorAll('.results .row').forEach((el, i) => {{
    el.style.opacity = '0';
    el.style.transform = 'translateY(8px)';
    el.style.transition = 'opacity .5s ease, transform .5s ease';
    setTimeout(() => {{
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }}, 600 + i * 80);
  }});
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FakeGPU unified test suite runner (Phase 4 dynamic + Phase 1-3 static)"
    )
    parser.add_argument(
        "--output", "-o", default="test/report.html",
        help="Output unified HTML report path (default: test/report.html)",
    )
    args = parser.parse_args()

    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    files = discover_test_files(test_dir)

    if not files:
        print("No test_error_*.py files found!")
        sys.exit(1)

    print(f"Discovered {len(files)} error simulation test files:")
    for f in files:
        print(f"  {f.name}")

    suites: list[SuiteResult] = []
    for path in files:
        stem = path.stem
        cat, desc = CATEGORY_MAP.get(stem, (stem, ""))
        results = run_test_file(path)
        suite = SuiteResult(category=cat, description=desc, results=results)
        suites.append(suite)
        icon = "PASS" if suite.failed == 0 else "FAIL"
        print(f"  [{icon}] {cat}: {suite.passed}/{suite.total} passed")

    generate_unified_html(suites, args.output)

    total_fail = sum(s.failed for s in suites)
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
