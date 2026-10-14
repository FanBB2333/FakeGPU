"""Measure mixed, scheduled streaming requests against a completion server.

This stdlib client records observed HTTP/text-chunk latency and server-reported
usage. It does not infer CUDA phase peaks, individual token times, or an unseen
server's scheduling policy. Requests are never retried.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ._cli import add_json_path_argument, command_prog, usage_error
from .structured_io import emit_json, load_mapping

REQUEST_SCHEMA = "fakegpu.online_serving_requests.v1"
SCHEMA_VERSION = "fakegpu.online_serving_benchmark.v1"


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seconds(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return float(value)


def _requests(manifest):
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"schema_version", "requests"}
        or manifest["schema_version"] != REQUEST_SCHEMA
    ):
        raise ValueError("unsupported online serving request manifest")
    rows = manifest["requests"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("requests must be a nonempty list")
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "request_id",
            "prompt_token_ids",
            "max_tokens",
            "arrival_offset_seconds",
        }:
            raise ValueError(
                "each request requires request_id, prompt_token_ids, max_tokens and arrival_offset_seconds"
            )
        name = row["request_id"]
        tokens = row["prompt_token_ids"]
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("request IDs must be unique nonempty strings")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in tokens
            )
        ):
            raise ValueError(
                "prompt_token_ids must be nonempty nonnegative integer lists"
            )
        seen.add(name)
        result.append(
            {
                "request_id": name,
                "prompt_token_ids": tokens.copy(),
                "max_tokens": _integer(row["max_tokens"], "max_tokens"),
                "arrival_offset_seconds": _seconds(
                    row["arrival_offset_seconds"], "arrival_offset_seconds"
                ),
            }
        )
    return sorted(result, key=lambda row: row["arrival_offset_seconds"])


def _events(response):
    """Read SSE data fields, including multiline events and keepalive comments."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
        elif not line.startswith((":", "id:", "event:", "retry:")):
            raise ValueError("unexpected non-SSE response line")
    if data:
        raise ValueError("unterminated SSE event")


def _request(base_url, model, row, origin, timeout):
    start = time.perf_counter()
    text = []
    events = []
    usage = None
    first = None
    done = False
    result = {
        "request_id": row["request_id"],
        "scheduled_arrival_seconds": row["arrival_offset_seconds"],
        "client_queue_seconds": max(
            0.0, start - origin - row["arrival_offset_seconds"]
        ),
        "prompt_tokens_requested": len(row["prompt_token_ids"]),
        "completion_tokens_requested": row["max_tokens"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        payload = {
            "model": model,
            "prompt": row["prompt_token_ids"],
            "max_tokens": row["max_tokens"],
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = Request(
            base_url + "/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with urlopen(request, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise ValueError("completion server must return text/event-stream")
            for data in _events(response):
                now = time.perf_counter()
                if now - start > timeout:
                    raise TimeoutError("stream exceeded request timeout")
                if data == "[DONE]":
                    done = True
                    break
                value = json.loads(data)
                if not isinstance(value, dict) or "error" in value:
                    raise ValueError(f"completion stream error: {value}")
                choices = value.get("choices")
                if not isinstance(choices, list):
                    raise ValueError("SSE completion requires choices")
                if len(choices) > 1:
                    raise ValueError("benchmark expects one completion per request")
                chunk = ""
                if choices:
                    choice = choices[0]
                    if not isinstance(choice, dict) or not isinstance(
                        choice.get("text"), str
                    ):
                        raise ValueError("completion choice requires text")
                    chunk = choice["text"]
                    text.append(chunk)
                    if chunk and first is None:
                        first = now - start
                if value.get("usage") is not None:
                    if usage is not None:
                        raise ValueError("completion stream repeated final usage")
                    usage = value["usage"]
                events.append({"elapsed_seconds": now - start, "response": value})
        if not done:
            raise ValueError("completion stream ended without [DONE]")
        if not isinstance(usage, dict):
            raise ValueError("completion stream requires final usage")
        for field, expected in (
            ("prompt_tokens", len(row["prompt_token_ids"])),
            ("completion_tokens", row["max_tokens"]),
        ):
            if _integer(usage.get(field), field) != expected:
                raise ValueError(
                    f"server usage.{field} differs from requested workload"
                )
        result.update(
            status="succeeded", usage=usage, first_nonempty_text_seconds=first
        )
    except Exception as exc:
        result.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        if isinstance(exc, HTTPError):
            result["http_status"] = exc.code
            try:
                with exc:
                    result["error_body"] = exc.read(4096).decode(
                        "utf-8", errors="replace"
                    )
            except OSError as body_error:
                result["error_body_read_error"] = str(body_error)
    end = time.perf_counter()
    output = "".join(text)
    result.update(
        http_seconds=end - start,
        end_to_end_seconds=end - origin - row["arrival_offset_seconds"],
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        output_text=output,
        output_sha256=hashlib.sha256(output.encode()).hexdigest(),
        events=events,
    )
    return result


def _distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "count": len(values),
        "min": values[0],
        "mean": sum(values) / len(values),
        "p50": values[math.ceil(0.5 * len(values)) - 1],
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "max": values[-1],
    }


def benchmark_online_serving(
    manifest, *, base_url, model, concurrency=4, repetitions=1, timeout_seconds=120
):
    """Execute an explicit mixed arrival schedule and retain failures and SSE events."""
    rows = _requests(manifest)
    _integer(concurrency, "concurrency")
    _integer(repetitions, "repetitions")
    timeout = _seconds(timeout_seconds, "timeout_seconds", positive=True)
    url = urlsplit(base_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "base_url must be an HTTP(S) API root without a query or fragment"
        )
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    report = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url.rstrip("/"),
        "model": model,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "request_manifest": json.loads(canonical),
        "concurrency": concurrency,
        "timeout_seconds": timeout,
        "repetitions": repetitions,
        "automatic_retries": False,
        "rounds": [],
        "limitations": [
            "First nonempty text measures HTTP delivery, not the first device token; one chunk may contain multiple tokens.",
            "Usage comes from the server; no tokenizer or device memory measurement is performed by this client.",
            "Client queue time includes the configured concurrency limit; percentiles use nearest rank.",
        ],
    }
    for index in range(repetitions):
        origin = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for row in rows:
                wait = origin + row["arrival_offset_seconds"] - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                futures.append(
                    executor.submit(
                        _request, report["base_url"], model, row, origin, timeout
                    )
                )
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.perf_counter() - origin
        results.sort(key=lambda row: row["request_id"])
        valid = [row for row in results if row["status"] == "succeeded"]
        tokens = sum(row["usage"]["completion_tokens"] for row in valid)
        report["rounds"].append(
            {
                "round_index": index + 1,
                "wall_seconds": elapsed,
                "succeeded": len(valid),
                "failed": len(results) - len(valid),
                "completion_tokens": tokens,
                "successful_completion_tokens_per_wall_second": tokens / elapsed,
                "http_seconds": _distribution([row["http_seconds"] for row in valid]),
                "end_to_end_seconds": _distribution(
                    [row["end_to_end_seconds"] for row in valid]
                ),
                "first_nonempty_text_seconds": _distribution(
                    [
                        row["first_nonempty_text_seconds"]
                        for row in valid
                        if row["first_nonempty_text_seconds"] is not None
                    ]
                ),
                "requests": results,
            }
        )
    report["status"] = (
        "complete" if all(row["failed"] == 0 for row in report["rounds"]) else "failed"
    )
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog=command_prog(__name__), description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    add_json_path_argument(parser)
    args = parser.parse_args(argv)
    try:
        report = benchmark_online_serving(
            load_mapping(args.requests),
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
            repetitions=args.repetitions,
            timeout_seconds=args.timeout_seconds,
        )
        emit_json(args.json_path or "-", report)
    except (OSError, ValueError, TypeError) as exc:
        usage_error(parser, exc)
    return 0 if report["status"] == "complete" else 1
