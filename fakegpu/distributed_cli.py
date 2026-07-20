from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from ._api import library_dir


_SIZE_PATTERN = re.compile(r"^([0-9]+)([kmgt]?i?b)?$", re.IGNORECASE)
_SIZE_MULTIPLIERS = {
    "": 1,
    "b": 1,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
    "tb": 1_000_000_000_000,
    "kib": 1 << 10,
    "mib": 1 << 20,
    "gib": 1 << 30,
    "tib": 1 << 40,
}


def parse_size(value: str) -> int:
    match = _SIZE_PATTERN.fullmatch(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            "expected a byte count such as 65536, 64KiB, or 4MiB"
        )
    amount = int(match.group(1))
    suffix = (match.group(2) or "").lower()
    size = amount * _SIZE_MULTIPLIERS[suffix]
    if size <= 0:
        raise argparse.ArgumentTypeError("size must be greater than zero")
    if size % 4 != 0:
        raise argparse.ArgumentTypeError(
            "size must be a multiple of 4 for the float32 benchmark"
        )
    return size


def parse_ranks(value: str) -> list[int]:
    ranks: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise argparse.ArgumentTypeError("rank list contains an empty item")
        if "-" in part:
            start_text, separator, end_text = part.partition("-")
            if (
                separator != "-"
                or not start_text.isdigit()
                or not end_text.isdigit()
            ):
                raise argparse.ArgumentTypeError(
                    "expected ranks such as 0,2,3 or 0-3"
                )
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError("rank ranges must be ascending")
            ranks.extend(range(start, end + 1))
        elif part.isdigit():
            ranks.append(int(part))
        else:
            raise argparse.ArgumentTypeError(
                "expected ranks such as 0,2,3 or 0-3"
            )
    if len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError("rank list contains duplicates")
    return ranks


def parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    host, separator, port_text = endpoint.rpartition(":")
    if separator != ":" or not host or not port_text.isdigit():
        raise argparse.ArgumentTypeError("expected HOST:PORT")
    port = int(port_text)
    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError("port must be within 1..65535")
    return host, port


def _nccl_library_path(native_dir: Path) -> Path:
    names = (
        ("libnccl.dylib", "libnccl.2.dylib")
        if sys.platform == "darwin"
        else ("libnccl.so.2", "libnccl.so")
    )
    for name in names:
        candidate = native_dir / name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"fake NCCL library was not found under {native_dir}")


def _coordinator_binary_path(native_dir: Path) -> Path:
    candidate = native_dir / "fakegpu-coordinator"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"fakegpu-coordinator was not found under {native_dir}; "
        "build the coordinator target or reinstall the package"
    )


def _clean_native_injection(env: dict[str, str]) -> None:
    env.pop("LD_PRELOAD", None)
    env.pop("DYLD_INSERT_LIBRARIES", None)


def _request_tcp(endpoint: str, request: str, timeout: float = 2.0) -> str:
    host, port = parse_tcp_endpoint(endpoint)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall((request + "\n").encode("utf-8"))
        response = bytearray()
        while not response.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
    return response.decode("utf-8", errors="replace").strip()


def _connect_endpoint_for_listen(endpoint: str) -> str:
    host, port = parse_tcp_endpoint(endpoint)
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    return f"{host}:{port}"


def _wait_for_coordinator(
    endpoint: str,
    process: subprocess.Popen[str],
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"coordinator exited with code {process.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            response = _request_tcp(endpoint, "PING", timeout=0.5)
            if (
                response.startswith("OK ")
                and "status=ready" in response
                and "transport=tcp" in response
            ):
                return
        except OSError as exc:
            last_error = exc
        time.sleep(0.05)
    raise TimeoutError(
        f"coordinator did not become ready at {endpoint}: {last_error}"
    )


def _write_local_cluster_config(
    path: Path,
    *,
    nodes: int,
    ranks_per_node: int,
    profile: str,
    interconnect_bandwidth_gbps: float,
    interconnect_latency_us: float,
) -> None:
    lines = [
        "version: 1",
        "cluster:",
        "  name: local-tcp-simulation",
        "  default_backend: nccl",
        "",
        "nodes:",
    ]
    next_rank = 0
    for node in range(nodes):
        ranks = list(range(next_rank, next_rank + ranks_per_node))
        next_rank += ranks_per_node
        lines.extend(
            [
                f"  - id: node{node}",
                f"    host: logical-node-{node}",
                f"    ranks: [{', '.join(str(rank) for rank in ranks)}]",
                "    gpus:",
            ]
        )
        lines.extend(f"      - profile: {profile}" for _ in ranks)
        lines.append("")
    lines.extend(
        [
            "fabric:",
            "  intra_node:",
            "    type: nvlink",
            "    bandwidth_gbps: 300",
            "    latency_us: 3",
            "",
            "  inter_node:",
            "    type: tcp",
            f"    bandwidth_gbps: {interconnect_bandwidth_gbps:g}",
            f"    latency_us: {interconnect_latency_us:g}",
            "    oversubscription: 1",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _coordinator_env(
    *,
    endpoint: str,
    cluster_config: Path | None,
    cluster_report: Path | None,
) -> dict[str, str]:
    env = dict(os.environ)
    _clean_native_injection(env)
    env.update(
        {
            "FAKEGPU_MODE": "simulate",
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_COORDINATOR_TRANSPORT": "tcp",
            "FAKEGPU_COORDINATOR_ADDR": endpoint,
        }
    )
    if cluster_config is not None:
        env["FAKEGPU_CLUSTER_CONFIG"] = str(cluster_config.resolve())
    else:
        env.pop("FAKEGPU_CLUSTER_CONFIG", None)
    if cluster_report is not None:
        cluster_report.parent.mkdir(parents=True, exist_ok=True)
        env["FAKEGPU_CLUSTER_REPORT_PATH"] = str(cluster_report.resolve())
    else:
        env.pop("FAKEGPU_CLUSTER_REPORT_PATH", None)
    return env


def coordinator_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu coordinator",
        description="Serve the FakeGPU distributed coordinator on a TCP endpoint.",
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--listen",
        "--address",
        dest="listen",
        help="TCP endpoint to bind, for example 0.0.0.0:29591.",
    )
    actions.add_argument(
        "--ping",
        metavar="HOST:PORT",
        help="Check whether a TCP coordinator is ready.",
    )
    actions.add_argument(
        "--shutdown",
        metavar="HOST:PORT",
        help="Ask a TCP coordinator to write its report and stop.",
    )
    parser.add_argument("--cluster-config", type=Path)
    parser.add_argument("--report", type=Path, help="Write the cluster report on shutdown.")
    parser.add_argument("--build-dir")
    parser.add_argument("--lib-dir")
    args = parser.parse_args(argv)

    try:
        if args.ping or args.shutdown:
            if args.cluster_config is not None or args.report is not None:
                parser.error("--cluster-config/--report are only valid with --listen")
            endpoint = args.ping or args.shutdown
            assert endpoint is not None
            parse_tcp_endpoint(endpoint)
            request = "PING" if args.ping else "SHUTDOWN"
            response = _request_tcp(endpoint, request)
            if not response.startswith("OK "):
                raise RuntimeError(f"coordinator returned: {response}")
            print(response)
            return 0

        assert args.listen is not None
        parse_tcp_endpoint(args.listen)
        if args.cluster_config is not None and not args.cluster_config.is_file():
            parser.error(f"cluster config does not exist: {args.cluster_config}")
        native_dir = library_dir(build_dir=args.build_dir, lib_dir=args.lib_dir)
        coordinator = _coordinator_binary_path(native_dir)
        env = _coordinator_env(
            endpoint=args.listen,
            cluster_config=args.cluster_config,
            cluster_report=args.report,
        )
        os.execve(
            str(coordinator),
            [
                str(coordinator),
                "--transport",
                "tcp",
                "--address",
                args.listen,
            ],
            env,
        )
    except (
        argparse.ArgumentTypeError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        parser.exit(1, f"fakegpu coordinator: {exc}\n")
    return 0


def _worker_env(
    *,
    endpoint: str,
    timeout_seconds: float,
    cluster_config: Path | None,
) -> dict[str, str]:
    env = dict(os.environ)
    _clean_native_injection(env)
    env.update(
        {
            "FAKEGPU_MODE": "simulate",
            "FAKEGPU_DIST_MODE": "simulate",
            "FAKEGPU_COORDINATOR_TRANSPORT": "tcp",
            "FAKEGPU_COORDINATOR_ADDR": endpoint,
            "FAKEGPU_COORDINATOR_TIMEOUT_MS": str(
                max(1, round(timeout_seconds * 1000))
            ),
            "FAKEGPU_STAGING_FORCE_SOCKET": "1",
        }
    )
    if cluster_config is not None:
        env["FAKEGPU_CLUSTER_CONFIG"] = str(cluster_config.resolve())
    else:
        env.pop("FAKEGPU_CLUSTER_CONFIG", None)
    return env


def _run_workers(
    *,
    endpoint: str,
    ranks: list[int],
    world_size: int,
    session: str,
    payload_bytes: int,
    warmup: int,
    iterations: int,
    timeout_seconds: float,
    nccl_lib: Path,
    report_dir: Path,
    cluster_config: Path | None,
) -> list[dict[str, Any]]:
    processes: list[tuple[int, Path, subprocess.Popen[str]]] = []
    worker_env = _worker_env(
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        cluster_config=cluster_config,
    )

    for rank in ranks:
        report_path = report_dir / f"rank_{rank}.json"
        command = [
            sys.executable,
            "-m",
            "fakegpu._bandwidth_worker",
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--session",
            session,
            "--payload-bytes",
            str(payload_bytes),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--nccl-lib",
            str(nccl_lib),
            "--report",
            str(report_path),
        ]
        process = subprocess.Popen(
            command,
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((rank, report_path, process))

    process_timeout = max(
        30.0,
        timeout_seconds * float(warmup + iterations + 2),
    )
    failures: list[str] = []
    reports: list[dict[str, Any]] = []
    try:
        for rank, report_path, process in processes:
            try:
                stdout, stderr = process.communicate(timeout=process_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
                failures.append(
                    f"rank {rank} exceeded {process_timeout:.1f}s\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
                continue
            if report_path.is_file():
                reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            if process.returncode != 0:
                failures.append(
                    f"rank {rank} exited with code {process.returncode}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
        if failures:
            details = "\n\n".join(failures)
            reported_errors = [
                f"rank {item.get('rank')}: {item.get('exception_message')}"
                for item in reports
                if item.get("status") != "success"
            ]
            if reported_errors:
                details += "\n\n" + "\n".join(reported_errors)
            raise RuntimeError(details)
        return sorted(reports, key=lambda item: int(item["rank"]))
    finally:
        for _, _, process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def _aggregate_report(
    *,
    endpoint: str,
    listen_endpoint: str | None,
    ranks: list[int],
    world_size: int,
    nodes: int | None,
    session: str,
    payload_bytes: int,
    warmup: int,
    iterations: int,
    rank_reports: list[dict[str, Any]],
    cluster_report: dict[str, Any] | None,
) -> dict[str, Any]:
    successful = [
        report for report in rank_reports if report.get("status") == "success"
    ]
    complete_world = len(successful) == world_size and len(ranks) == world_size
    slowest_total = max(
        (float(report["total_seconds"]) for report in successful),
        default=0.0,
    )

    aggregate: dict[str, Any] = {
        "schema_version": "fakegpu.tcp_bandwidth.v1",
        "status": "success" if len(successful) == len(ranks) else "error",
        "endpoint": endpoint,
        "listen_endpoint": listen_endpoint,
        "transport": "tcp_socket_payload",
        "session": session,
        "world_size": world_size,
        "logical_node_count": nodes,
        "local_ranks": ranks,
        "complete_world_measured_locally": complete_world,
        "payload_bytes_per_rank": payload_bytes,
        "warmup_iterations": warmup,
        "iterations": iterations,
        "rank_reports": rank_reports,
        "measurement_scope": (
            "local_complete_world" if complete_world else "local_rank_subset"
        ),
        "measurement_note": (
            "End-to-end FakeGPU collective throughput includes TCP transfer, "
            "coordinator reduction, memory copies, and process scheduling."
        ),
    }
    if complete_world and slowest_total > 0.0:
        payload_bits = float(payload_bytes * 8)
        aggregate.update(
            {
                "timed_seconds": slowest_total,
                "algorithmic_bandwidth_gbps": (
                    payload_bits * iterations / slowest_total / 1e9
                ),
                "aggregate_socket_payload_throughput_gbps": (
                    payload_bits
                    * 2.0
                    * world_size
                    * iterations
                    / slowest_total
                    / 1e9
                ),
            }
        )
    if cluster_report is not None:
        aggregate["cluster_report"] = cluster_report
    return aggregate


def _format_bytes(size: int) -> str:
    for suffix, divisor in (
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
    ):
        if size >= divisor and size % divisor == 0:
            return f"{size // divisor} {suffix}"
    return f"{size} B"


def _print_bandwidth_summary(report: dict[str, Any]) -> None:
    print("FakeGPU TCP bandwidth benchmark: PASS")
    print(f"  endpoint: {report['endpoint']}")
    if report.get("listen_endpoint"):
        print(f"  listen: {report['listen_endpoint']}")
    print(
        f"  ranks: {report['local_ranks']} / world_size={report['world_size']}"
    )
    if report.get("logical_node_count") is not None:
        print(f"  logical nodes: {report['logical_node_count']}")
    print(
        f"  payload: {_format_bytes(int(report['payload_bytes_per_rank']))} per rank"
    )
    if "algorithmic_bandwidth_gbps" in report:
        print(
            "  end-to-end algorithmic bandwidth: "
            f"{report['algorithmic_bandwidth_gbps']:.3f} Gbit/s"
        )
        print(
            "  aggregate TCP payload throughput: "
            f"{report['aggregate_socket_payload_throughput_gbps']:.3f} Gbit/s"
        )
    else:
        for rank_report in report["rank_reports"]:
            print(
                f"  rank {rank_report['rank']} algorithmic bandwidth: "
                f"{rank_report['algorithmic_bandwidth_gbps']:.3f} Gbit/s"
            )
    print("  note: value includes coordinator reduction and memory-copy overhead")


def bandwidth_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fakegpu bandwidth",
        description=(
            "Measure end-to-end FakeGPU collective throughput over TCP. "
            "Use --listen for a self-contained local multi-node simulation, "
            "or --connect on each physical host."
        ),
    )
    endpoints = parser.add_mutually_exclusive_group(required=True)
    endpoints.add_argument(
        "--listen",
        help="Start a local coordinator on HOST:PORT and launch all logical ranks.",
    )
    endpoints.add_argument(
        "--connect",
        help="Connect local rank workers to an existing coordinator at HOST:PORT.",
    )
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--ranks-per-node", type=int, default=1)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--ranks", type=parse_ranks)
    parser.add_argument(
        "--session",
        help="Shared session name. Required for --connect on multiple hosts.",
    )
    parser.add_argument(
        "--size",
        "--bytes",
        dest="payload_bytes",
        type=parse_size,
        default=parse_size("4MiB"),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--profile", default="a100")
    parser.add_argument("--cluster-config", type=Path)
    parser.add_argument("--cluster-report", type=Path)
    parser.add_argument(
        "--interconnect-bandwidth-gbps",
        type=float,
        default=25.0,
        help="Modeled bandwidth written to an auto-generated local topology.",
    )
    parser.add_argument(
        "--interconnect-latency-us",
        type=float,
        default=50.0,
        help="Modeled latency written to an auto-generated local topology.",
    )
    parser.add_argument("--json", type=Path, help="Write the benchmark report as JSON.")
    parser.add_argument("--build-dir")
    parser.add_argument("--lib-dir")
    args = parser.parse_args(argv)

    try:
        if args.nodes <= 0:
            parser.error("--nodes must be greater than zero")
        if args.ranks_per_node <= 0:
            parser.error("--ranks-per-node must be greater than zero")
        if args.warmup < 0:
            parser.error("--warmup must be non-negative")
        if args.iterations <= 0:
            parser.error("--iterations must be greater than zero")
        if args.timeout <= 0:
            parser.error("--timeout must be greater than zero")
        if args.interconnect_bandwidth_gbps <= 0:
            parser.error("--interconnect-bandwidth-gbps must be greater than zero")
        if args.interconnect_latency_us < 0:
            parser.error("--interconnect-latency-us must be non-negative")
        if args.cluster_config is not None and not args.cluster_config.is_file():
            parser.error(f"cluster config does not exist: {args.cluster_config}")

        native_dir = library_dir(build_dir=args.build_dir, lib_dir=args.lib_dir)
        nccl_lib = _nccl_library_path(native_dir)

        if args.listen:
            parse_tcp_endpoint(args.listen)
            if args.world_size is not None or args.ranks is not None:
                parser.error(
                    "--world-size/--ranks are only used with --connect; "
                    "use --nodes and --ranks-per-node with --listen"
                )
            world_size = args.nodes * args.ranks_per_node
            ranks = list(range(world_size))
            session = args.session or f"local-{secrets.token_hex(8)}"
            connect_endpoint = _connect_endpoint_for_listen(args.listen)

            with tempfile.TemporaryDirectory(
                prefix="fakegpu-tcp-bandwidth-"
            ) as temp_dir_text:
                temp_dir = Path(temp_dir_text)
                cluster_config = args.cluster_config
                if cluster_config is None:
                    cluster_config = temp_dir / "cluster.yaml"
                    _write_local_cluster_config(
                        cluster_config,
                        nodes=args.nodes,
                        ranks_per_node=args.ranks_per_node,
                        profile=args.profile,
                        interconnect_bandwidth_gbps=(
                            args.interconnect_bandwidth_gbps
                        ),
                        interconnect_latency_us=args.interconnect_latency_us,
                    )
                cluster_report_path = (
                    args.cluster_report
                    if args.cluster_report is not None
                    else temp_dir / "cluster_report.json"
                )
                coordinator = _coordinator_binary_path(native_dir)
                coordinator_process = subprocess.Popen(
                    [
                        str(coordinator),
                        "--transport",
                        "tcp",
                        "--address",
                        args.listen,
                    ],
                    env=_coordinator_env(
                        endpoint=args.listen,
                        cluster_config=cluster_config,
                        cluster_report=cluster_report_path,
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    _wait_for_coordinator(
                        connect_endpoint,
                        coordinator_process,
                    )
                    rank_reports = _run_workers(
                        endpoint=connect_endpoint,
                        ranks=ranks,
                        world_size=world_size,
                        session=session,
                        payload_bytes=args.payload_bytes,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        timeout_seconds=args.timeout,
                        nccl_lib=nccl_lib,
                        report_dir=temp_dir / "ranks",
                        cluster_config=cluster_config,
                    )
                    response = _request_tcp(
                        connect_endpoint,
                        "SHUTDOWN",
                        timeout=2.0,
                    )
                    if not response.startswith("OK "):
                        raise RuntimeError(
                            f"coordinator rejected shutdown: {response}"
                        )
                    coordinator_process.wait(timeout=5)
                    if coordinator_process.returncode != 0:
                        stdout, stderr = coordinator_process.communicate()
                        raise RuntimeError(
                            "coordinator failed during shutdown\n"
                            f"stdout:\n{stdout}\nstderr:\n{stderr}"
                        )
                    cluster_report = (
                        json.loads(
                            cluster_report_path.read_text(encoding="utf-8")
                        )
                        if cluster_report_path.is_file()
                        else None
                    )
                    report = _aggregate_report(
                        endpoint=connect_endpoint,
                        listen_endpoint=args.listen,
                        ranks=ranks,
                        world_size=world_size,
                        nodes=args.nodes,
                        session=session,
                        payload_bytes=args.payload_bytes,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        rank_reports=rank_reports,
                        cluster_report=cluster_report,
                    )
                finally:
                    if coordinator_process.poll() is None:
                        try:
                            _request_tcp(
                                connect_endpoint,
                                "SHUTDOWN",
                                timeout=1.0,
                            )
                            coordinator_process.wait(timeout=3)
                        except Exception:
                            coordinator_process.kill()
                            coordinator_process.wait(timeout=5)
        else:
            assert args.connect is not None
            parse_tcp_endpoint(args.connect)
            if args.world_size is None or args.world_size <= 0:
                parser.error(
                    "--world-size is required and must be positive with --connect"
                )
            if args.ranks is None:
                parser.error("--ranks is required with --connect")
            if not args.session:
                parser.error(
                    "--session is required with --connect so every host uses "
                    "the same communicator ID"
                )
            if any(rank < 0 or rank >= args.world_size for rank in args.ranks):
                parser.error("every rank must be within [0, world-size)")
            world_size = args.world_size
            ranks = args.ranks
            session = args.session
            with tempfile.TemporaryDirectory(
                prefix="fakegpu-tcp-bandwidth-client-"
            ) as temp_dir_text:
                rank_reports = _run_workers(
                    endpoint=args.connect,
                    ranks=ranks,
                    world_size=world_size,
                    session=session,
                    payload_bytes=args.payload_bytes,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    timeout_seconds=args.timeout,
                    nccl_lib=nccl_lib,
                    report_dir=Path(temp_dir_text) / "ranks",
                    cluster_config=args.cluster_config,
                )
                report = _aggregate_report(
                    endpoint=args.connect,
                    listen_endpoint=None,
                    ranks=ranks,
                    world_size=world_size,
                    nodes=None,
                    session=session,
                    payload_bytes=args.payload_bytes,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    rank_reports=rank_reports,
                    cluster_report=None,
                )

        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _print_bandwidth_summary(report)
        return 0
    except (
        argparse.ArgumentTypeError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.exit(1, f"fakegpu bandwidth: {exc}\n")
    return 1
