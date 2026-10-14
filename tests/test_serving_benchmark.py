import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
import time
import subprocess
import sys

from fakegpu.structured_io import load_mapping

import pytest

from fakegpu.serving_benchmark import _events, benchmark_online_serving


def manifest():
    return {
        "schema_version": "fakegpu.online_serving_requests.v1",
        "requests": [
            {
                "request_id": "later",
                "prompt_token_ids": [7, 8, 9],
                "max_tokens": 2,
                "arrival_offset_seconds": 0.01,
            },
            {
                "request_id": "first",
                "prompt_token_ids": [7],
                "max_tokens": 2,
                "arrival_offset_seconds": 0,
            },
        ],
    }


@pytest.fixture
def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            if self.path.startswith("/http-error/"):
                self.send_response(429)
                self.end_headers()
                self.wfile.write(b'{"error":"busy"}')
                return
            if self.path.startswith("/slow-error/"):
                self.send_response(429)
                self.end_headers()
                time.sleep(0.15)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            usage = {
                "prompt_tokens": len(payload["prompt"]),
                "completion_tokens": payload["max_tokens"],
            }
            if self.path.startswith("/bad-usage/"):
                usage["completion_tokens"] += 1
            values = [{"choices": [{"text": "测试"}]}, {"choices": [], "usage": usage}]
            for value in values:
                self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.flush()
            if not self.path.startswith("/missing-done/"):
                self.wfile.write(b"data: [DONE]\n\n")

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", seen
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_mixed_streaming_schedule_preserves_raw_evidence_and_usage(server):
    base, seen = server
    source = manifest()
    before = copy.deepcopy(source)
    report = benchmark_online_serving(
        source, base_url=base + "/v1", model="local", concurrency=2, repetitions=2
    )
    assert report["status"] == "complete"
    assert len(seen) == 4
    assert all(
        row["ignore_eos"] and row["stream_options"]["include_usage"] for row in seen
    )
    for round_ in report["rounds"]:
        assert round_["succeeded"] == 2 and round_["failed"] == 0
        assert round_["completion_tokens"] == 4
        assert round_["http_seconds"]["count"] == 2
        for row in round_["requests"]:
            assert row["output_text"] == "测试" and len(row["events"]) == 2
            assert 0 <= row["first_nonempty_text_seconds"] <= row["http_seconds"]
            assert row["end_to_end_seconds"] == pytest.approx(
                row["http_seconds"] + row["client_queue_seconds"]
            )
    assert source == before
    source["requests"][0]["prompt_token_ids"].append(10)
    assert report["request_manifest"] == before


@pytest.mark.parametrize(
    "path,error",
    [
        ("bad-usage", "usage.completion_tokens"),
        ("missing-done", "without [DONE]"),
        ("http-error", "429"),
    ],
)
def test_failed_requests_are_retained_and_never_retried(server, path, error):
    base, seen = server
    report = benchmark_online_serving(
        manifest(), base_url=base + "/" + path, model="local", concurrency=1
    )
    assert report["status"] == "failed"
    assert report["rounds"][0]["failed"] == 2
    assert len(seen) == 2
    assert report["rounds"][0]["completion_tokens"] == 0
    assert all(error in row["error"] for row in report["rounds"][0]["requests"])


def test_sse_multiline_keepalive_and_truncation():
    raw = b': keepalive\nevent: completion\nid: 1\ndata: {"choices":\ndata: []}\n\ndata: [DONE]\n\n'
    assert list(_events(BytesIO(raw))) == ['{"choices":\n[]}', "[DONE]"]
    with pytest.raises(ValueError, match="unterminated"):
        list(_events(BytesIO(b'data: {"choices": []}')))


def test_http_error_body_timeout_still_produces_failure_evidence(server):
    base, seen = server
    report = benchmark_online_serving(
        manifest(), base_url=base + "/slow-error", model="local", timeout_seconds=0.05
    )
    assert report["status"] == "failed" and len(seen) == 2
    assert all(
        row["http_status"] == 429 and "error_body_read_error" in row
        for row in report["rounds"][0]["requests"]
    )


@pytest.mark.parametrize("path,status", [("v1", 0), ("bad-usage", 1)])
def test_stdlib_only_cli_preserves_compressed_success_and_failure(
    server, tmp_path, path, status
):
    base, _ = server
    inputs, output = tmp_path / "requests.json", tmp_path / "report.json.gz"
    inputs.write_text(json.dumps(manifest()))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "fakegpu",
            "benchmark-serving",
            "--requests",
            str(inputs),
            "--base-url",
            base + "/" + path,
            "--model",
            "local",
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == status, result.stderr
    assert load_mapping(output)["status"] == ("complete" if status == 0 else "failed")


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "boolean-token",
        "negative-arrival",
        "infinite-arrival",
        "boolean-count",
        "unknown-field",
    ],
)
def test_invalid_workloads_fail_before_network(change, monkeypatch):
    source = manifest()
    if change == "duplicate":
        source["requests"][1]["request_id"] = source["requests"][0]["request_id"]
    elif change == "boolean-token":
        source["requests"][0]["prompt_token_ids"] = [True]
    elif change == "negative-arrival":
        source["requests"][0]["arrival_offset_seconds"] = -1
    elif change == "infinite-arrival":
        source["requests"][0]["arrival_offset_seconds"] = float("inf")
    elif change == "boolean-count":
        source["requests"][0]["max_tokens"] = True
    else:
        source["requests"][0]["guess"] = 1
    monkeypatch.setattr(
        "fakegpu.serving_benchmark.urlopen",
        lambda *a, **k: pytest.fail("invalid workload reached network"),
    )
    with pytest.raises(ValueError):
        benchmark_online_serving(
            source, base_url="http://127.0.0.1:1/v1", model="local"
        )
