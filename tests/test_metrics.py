from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from fakegpu.metrics import (
    HISTORY_SCHEMA_VERSION,
    MetricsApplication,
    MetricsCollector,
    MetricsHistory,
    MetricsHttpServer,
    SNAPSHOT_SCHEMA_VERSION,
    build_metrics_snapshot,
    main,
    render_prometheus,
)
from fakegpu.smi import SmiStatePublisher, build_inventory


def _published_state(
    path: Path,
    *,
    pid: int,
    process_name: str,
    used_bytes: int,
) -> dict:
    state = SmiStatePublisher(
        path,
        lambda: {
            "runtime_backend": "upstream",
            "stage": "decode",
            "devices": [
                {
                    "index": 0,
                    "name": "Fake NVIDIA A100-SXM4-80GB",
                    "profile_id": "a100",
                    "total_memory": 80 * 2**30,
                    "current_memory": used_bytes,
                    "peak_memory": used_bytes + 128 * 2**20,
                }
            ],
        },
    ).publish_once(running=True)
    state["pid"] = pid
    state["process_name"] = process_name
    state["timestamp_ns"] = time.time_ns()
    state["devices"][0]["native_activity"] = {
        "io_calls": pid,
        "io_bytes": used_bytes,
        "kernel_launches": pid + 1,
        "gemm_calls": 2,
        "gemm_flops": 4096,
        "compatibility_events": 1,
        "unsupported_api_calls": 0,
    }
    path.write_text(
        json.dumps(state, sort_keys=True),
        encoding="utf-8",
    )
    return state


def test_metrics_snapshot_bounds_process_series_and_renders_prometheus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "FAKEGPU_MIG_LAYOUT",
        "0:1g.10gb:10240:2",
    )
    monkeypatch.setenv(
        "FAKEGPU_FAULT_EVENTS",
        "0:XID_79:critical",
    )
    low = _published_state(
        tmp_path / "low.json",
        pid=11,
        process_name="low-worker",
        used_bytes=1 * 2**30,
    )
    high = _published_state(
        tmp_path / "high.json",
        pid=22,
        process_name='high"\nworker',
        used_bytes=3 * 2**30,
    )
    inventory = build_inventory(
        [low, high],
        now_ns=time.time_ns(),
    )

    snapshot = build_metrics_snapshot(
        inventory,
        max_process_series=1,
        source_state_count=2,
    )

    assert snapshot["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert snapshot["scrape_success"] is True
    assert snapshot["source_state_count"] == 2
    assert snapshot["device_count"] == 1
    assert snapshot["process_series_total"] == 2
    assert snapshot["process_series_retained"] == 1
    assert snapshot["process_series_dropped"] == 1
    assert snapshot["processes"][0]["pid"] == 22
    assert snapshot["devices"][0]["memory"]["used_bytes"] == (
        4 * 2**30
    )
    assert snapshot["devices"][0]["health"]["max_severity"] == (
        "critical"
    )
    assert snapshot["devices"][0]["mig"]["instance_count"] == 2

    rendered = render_prometheus(
        snapshot,
        history_sample_count=3,
    )
    assert "# TYPE fakegpu_device_memory_bytes gauge" in rendered
    assert (
        'fakegpu_exporter_process_series{kind="dropped"} 1'
        in rendered
    )
    assert "fakegpu_device_health_severity" in rendered
    assert "fakegpu_device_mig_instances" in rendered
    assert "fakegpu_device_native_activity" in rendered
    assert 'process_name="high\\"\\nworker"' in rendered
    assert "fakegpu_exporter_history_samples 3" in rendered


def test_metrics_history_is_bounded_and_returns_copies() -> None:
    history = MetricsHistory(max_samples=2)
    for timestamp in (1, 2, 3):
        history.append(
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "generated_at_ns": timestamp,
            }
        )

    samples = history.snapshots()
    assert [item["generated_at_ns"] for item in samples] == [2, 3]
    samples[0]["generated_at_ns"] = 99
    assert history.snapshots()[0]["generated_at_ns"] == 2


def test_metrics_collector_and_cli_export_state_directory(
    tmp_path: Path,
    capsys,
) -> None:
    _published_state(
        tmp_path / "worker-1.json",
        pid=31,
        process_name="worker-1",
        used_bytes=1 * 2**30,
    )
    _published_state(
        tmp_path / "worker-2.json",
        pid=32,
        process_name="worker-2",
        used_bytes=2 * 2**30,
    )

    assert (
        main(
            [
                "--state-dir",
                str(tmp_path),
                "--max-process-series",
                "1",
            ]
        )
        == 0
    )
    prometheus = capsys.readouterr().out
    assert "fakegpu_exporter_scrape_success 1" in prometheus
    assert "fakegpu_exporter_source_states 2" in prometheus
    assert (
        'fakegpu_exporter_process_series{kind="retained"} 1'
        in prometheus
    )

    assert (
        main(
            [
                "--state-dir",
                str(tmp_path),
                "--json",
                "--max-process-series",
                "1",
            ]
        )
        == 0
    )
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["source_state_count"] == 2
    assert snapshot["process_series_retained"] == 1
    assert snapshot["process_series_dropped"] == 1

    (tmp_path / "invalid.json").write_text(
        "{",
        encoding="utf-8",
    )
    collector = MetricsCollector(
        state_dir=tmp_path,
        history_size=2,
    )
    degraded = collector.collect_once()
    assert degraded["scrape_success"] is False
    assert degraded["source_state_count"] == 2
    assert degraded["error_count"] == 1
    assert len(collector.history) == 1
    degraded["error_count"] = 99
    assert collector.history.snapshots()[0]["error_count"] == 1


def test_metrics_http_endpoints_expose_latest_and_bounded_history(
    tmp_path: Path,
) -> None:
    _published_state(
        tmp_path / "worker.json",
        pid=41,
        process_name="http-worker",
        used_bytes=2 * 2**30,
    )
    collector = MetricsCollector(
        state_dir=tmp_path,
        history_size=2,
    )
    application = MetricsApplication(
        collector,
        interval_seconds=60,
    )
    server = MetricsHttpServer(
        ("127.0.0.1", 0),
        application,
    )
    application.start()
    refreshed = application.refresh()
    refreshed["device_count"] = 99
    assert application.snapshot()["device_count"] == 1
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"
    try:
        with urlopen(f"{base_url}/metrics", timeout=5) as response:
            body = response.read().decode("utf-8")
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert "fakegpu_exporter_history_samples 2" in body

        with urlopen(f"{base_url}/healthz", timeout=5) as response:
            health = json.loads(response.read())
            assert health["status"] == "ok"
            assert health["source_state_count"] == 1
            assert health["history_sample_count"] == 2

        with urlopen(
            f"{base_url}/api/v1/history",
            timeout=5,
        ) as response:
            history = json.loads(response.read())
            assert history["schema_version"] == (
                HISTORY_SCHEMA_VERSION
            )
            assert history["max_samples"] == 2
            assert history["sample_count"] == 2

        with pytest.raises(HTTPError) as missing:
            urlopen(f"{base_url}/missing", timeout=5)
        assert missing.value.code == 404
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        application.stop()


def test_metrics_cli_rejects_missing_source_and_invalid_limits() -> None:
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["--state", "state.json", "--history-size", "0"])
    with pytest.raises(SystemExit):
        main(["--state", "state.json", "--max-process-series", "257"])
    with pytest.raises(SystemExit):
        main(["--state", "state.json", "--serve", "--json"])
