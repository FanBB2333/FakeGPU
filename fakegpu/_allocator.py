"""FakeCUDA device-memory accounting and the modeled caching allocator.

The allocator part ports the block model of PyTorch's
``c10/cuda/CUDACachingAllocator.cpp`` (checked against v2.9.1 and v2.14.1):
rounding, the small and large pools with per-stream free lists, splitting,
coalescing, oversize blocks, garbage collection under a memory fraction,
``record_stream`` events, expandable segments, the OOM retry chain and the
allocation history behind ``torch.cuda.memory._snapshot()``.  The
``PYTORCH_CUDA_ALLOC_CONF`` / ``PYTORCH_ALLOC_CONF`` parser follows
``CUDAAllocatorConfig`` (v2.9) and ``AcceleratorAllocatorConfig`` (v2.10+).

Where the installed torch versions differ, the tracker follows the installed
version; ``torch_version`` selects another one for tests.  Device memory is
synthetic: segment addresses are made up and only the byte accounting is
modeled.  The tracker also keeps FakeGPU's own counters (requested bytes per
category and stage, largest allocations) for the FakeGPU reports.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from bisect import bisect_left, insort
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable


_dispatch_tracking_lock = threading.RLock()


_dispatch_tracking_stats: dict[str, Any] = {}


def _reset_dispatch_tracking_stats(*, enabled: bool) -> None:
    with _dispatch_tracking_lock:
        _dispatch_tracking_stats.clear()
        _dispatch_tracking_stats.update(
            {
                "enabled": bool(enabled),
                "operator_calls": 0,
                "output_tensors": 0,
                "new_allocations": 0,
                "alias_outputs": 0,
                "inaccessible_outputs": 0,
                "operators": {},
            }
        )


_reset_dispatch_tracking_stats(enabled=False)


def _record_dispatch_tracking(
    operator: str,
    *,
    output_tensors: int,
    new_allocations: int,
    alias_outputs: int,
    inaccessible_outputs: int,
) -> None:
    with _dispatch_tracking_lock:
        _dispatch_tracking_stats["operator_calls"] += 1
        _dispatch_tracking_stats["output_tensors"] += int(output_tensors)
        _dispatch_tracking_stats["new_allocations"] += int(new_allocations)
        _dispatch_tracking_stats["alias_outputs"] += int(alias_outputs)
        _dispatch_tracking_stats["inaccessible_outputs"] += int(
            inaccessible_outputs
        )
        operators = _dispatch_tracking_stats["operators"]
        record = operators.setdefault(
            operator,
            {
                "calls": 0,
                "output_tensors": 0,
                "new_allocations": 0,
                "alias_outputs": 0,
                "inaccessible_outputs": 0,
            },
        )
        record["calls"] += 1
        record["output_tensors"] += int(output_tensors)
        record["new_allocations"] += int(new_allocations)
        record["alias_outputs"] += int(alias_outputs)
        record["inaccessible_outputs"] += int(inaccessible_outputs)


def _dispatch_tracking_snapshot() -> dict[str, Any]:
    with _dispatch_tracking_lock:
        operators = {
            name: dict(values)
            for name, values in sorted(
                _dispatch_tracking_stats["operators"].items()
            )
        }
        return {
            **{
                key: value
                for key, value in _dispatch_tracking_stats.items()
                if key != "operators"
            },
            "operators": operators,
        }


# ---------------------------------------------------------------------------
# Constants (c10/core/AllocatorConfig.h, CUDACachingAllocator.cpp)
# ---------------------------------------------------------------------------

_MB = 1024 * 1024
_K_MIN_BLOCK_SIZE = 512
_K_SMALL_SIZE = _MB
_K_SMALL_BUFFER = 2 * _MB
_K_MIN_LARGE_ALLOC = 10 * _MB
_K_ROUND_LARGE = 2 * _MB
_K_LARGE_BUFFER = 20 * _MB
_ROUNDUP_INTERVALS = 16
_SIZE_MAX = 2**64 - 1
_PINNED_MAX_REGISTER_THREADS = 128

# Names kept for older imports.
_ALLOCATOR_ALIGNMENT_BYTES = _K_MIN_BLOCK_SIZE
_ALLOCATOR_SMALL_SEGMENT_BYTES = _K_SMALL_BUFFER
_ALLOCATOR_MEDIUM_SEGMENT_BYTES = _K_LARGE_BUFFER
_ALLOCATOR_LARGE_ALIGNMENT_BYTES = _K_ROUND_LARGE
_ALLOCATOR_SMALL_REQUEST_LIMIT = _K_SMALL_SIZE
_ALLOCATOR_MEDIUM_REQUEST_LIMIT = _K_MIN_LARGE_ALLOC

# First synthetic segment address reported by memory_snapshot(); each device
# gets its own range, and segments never reuse an address.
_SNAPSHOT_ADDRESS_BASE = 0x7F0000000000
_DEVICE_ADDRESS_STRIDE = 1 << 44

# Modeled default for the memory the CUDA context and library handles hold
# outside the caching allocator.  Measured once on an Ampere GPU (sm_80) with
# driver 580 and torch 2.9.1+cu128, after the first cuBLAS matmul: total minus
# free minus reserved was 601.5 MiB.  It is one device's number, not a
# property of the architecture; FAKEGPU_CUDA_CONTEXT_BYTES overrides it.
_DEFAULT_CUDA_CONTEXT_BYTES = 630_718_464
_CONTEXT_BYTES_ENV = "FAKEGPU_CUDA_CONTEXT_BYTES"

# The newest torch release whose allocator sources this module follows.
_LATEST_MODELED_VERSION = (2, 14)


def _detect_torch_version() -> tuple[int, int]:
    try:
        import torch

        text = str(torch.__version__)
    except Exception:
        return _LATEST_MODELED_VERSION
    match = re.match(r"(\d+)\.(\d+)", text)
    if match is None:
        return _LATEST_MODELED_VERSION
    return int(match.group(1)), int(match.group(2))


def _configured_context_bytes() -> int:
    value = os.environ.get(_CONTEXT_BYTES_ENV)
    if value is None or not value.strip():
        return _DEFAULT_CUDA_CONTEXT_BYTES
    try:
        return max(0, int(value.strip()))
    except ValueError:
        return _DEFAULT_CUDA_CONTEXT_BYTES


def _format_size(size: int) -> str:
    """Mirror ``c10::CachingAllocator::format_size`` used in OOM messages."""
    if size <= 1024:
        return f"{size} bytes"
    if size <= 1024**2:
        return f"{size / 1024:.2f} KiB"
    if size <= 1024**3:
        return f"{size / 1024**2:.2f} MiB"
    return f"{size / 1024**3:.2f} GiB"


def _is_power_of_2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _log2_floor(value: int) -> int:
    return value.bit_length() - 1


def _int64(value: int) -> int:
    """Cast a size_t to int64_t the way the stats bindings do."""
    value &= _SIZE_MAX
    return value - 2**64 if value >= 2**63 else value


def _public_allocation_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not str(key).startswith("_")}


# ---------------------------------------------------------------------------
# PYTORCH_CUDA_ALLOC_CONF / PYTORCH_ALLOC_CONF
# ---------------------------------------------------------------------------


@dataclass
class _AllocatorConfig:
    """The settings ``CUDAAllocatorConfig`` and ``AcceleratorAllocatorConfig`` keep."""

    max_split_size: int = _SIZE_MAX
    max_non_split_rounding_size: int = _K_LARGE_BUFFER
    large_segment_size: int = _K_LARGE_BUFFER
    garbage_collection_threshold: float = 0.0
    roundup_power2_divisions: list[int] = field(
        default_factory=lambda: [0] * _ROUNDUP_INTERVALS
    )
    expandable_segments: bool = False
    backend: str = "native"
    release_lock_on_cudamalloc: bool = False
    pinned_use_cuda_host_register: bool = False
    pinned_num_register_threads: int = 1
    pinned_use_background_threads: bool = False
    pinned_reserve_segment_size_mb: int = 0
    graph_capture_record_stream_reuse: bool = False
    per_process_memory_fraction: float = 1.0
    pinned_free_catch_all: bool = False
    throw_on_cudamalloc_oom: bool = False
    pinned_max_round_threshold: int = _SIZE_MAX
    pinned_max_cached_size: int = _SIZE_MAX
    last_allocator_settings: str = ""

    def divisions_for(self, size: int) -> int:
        # roundup_power2_divisions(size): intervals start at 1 MiB (2^20).
        index = max(0, _log2_floor(size) - 20)
        index = min(index, _ROUNDUP_INTERVALS - 1)
        return self.roundup_power2_divisions[index]


_INTEGER_PREFIX = re.compile(r"[ \t\n\r\f\v]*([+-]?)(\d+)")
_FLOAT_PREFIX = re.compile(
    r"[ \t\n\r\f\v]*[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)",
    re.IGNORECASE,
)


def _c_integer(token: str, function: str) -> int:
    """``std::stoi`` / ``std::stoul`` / ``std::stoull`` on a token.

    The parse stops at the first non-digit; a token without a leading number
    raises ``std::invalid_argument``, which torch reports as a RuntimeError
    carrying the function name.  The unsigned forms wrap negative values.
    """
    match = _INTEGER_PREFIX.match(token)
    if match is None:
        raise RuntimeError(function)
    magnitude = int(match.group(2))
    negative = match.group(1) == "-"
    if function == "stoi":
        value = -magnitude if negative else magnitude
        if not -(2**31) <= value < 2**31:
            raise RuntimeError(function)
        return value
    if magnitude > _SIZE_MAX:
        raise RuntimeError(function)
    return (-magnitude) & _SIZE_MAX if negative else magnitude


def _c_double(token: str) -> float:
    match = _FLOAT_PREFIX.match(token)
    if match is None:
        raise RuntimeError("stod")
    return float(match.group(0).strip())


def _size_t(value: int) -> int:
    return value & _SIZE_MAX


class _LegacyConfigParser:
    """``CUDAAllocatorConfig::parseArgs`` up to PyTorch 2.9.

    Errors are ``TORCH_CHECK`` failures, so they surface as RuntimeError.
    """

    def __init__(self, env: str, config: _AllocatorConfig, *, runtime: bool):
        self.config = config
        self.runtime = runtime
        tokens: list[str] = []
        buffer = ""
        for ch in env:
            if ch in ",:[]":
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(ch)
            elif ch != " ":
                buffer += ch
        if buffer:
            tokens.append(buffer)
        self.tokens = tokens
        self.env = env

    def consume(self, i: int, ch: str) -> None:
        if not (i < len(self.tokens) and self.tokens[i] == ch):
            raise RuntimeError(f"Error parsing CachingAllocator settings, expected {ch}")

    def has(self, i: int) -> bool:
        return i < len(self.tokens)

    def parse(self) -> None:
        cfg = self.config
        cfg.max_split_size = _SIZE_MAX
        cfg.roundup_power2_divisions = [0] * _ROUNDUP_INTERVALS
        cfg.garbage_collection_threshold = 0.0
        cfg.last_allocator_settings = self.env
        tokens = self.tokens
        i = 0
        while i < len(tokens):
            key = tokens[i]
            if key in ("max_split_size_mb", "max_non_split_rounding_mb"):
                i = self.parse_mb_size(i, key)
            elif key == "garbage_collection_threshold":
                self.consume(i + 1, ":")
                i += 2
                if not self.has(i):
                    raise RuntimeError(
                        "Error, expecting garbage_collection_threshold value"
                    )
                value = _c_double(tokens[i])
                if not value > 0:
                    raise RuntimeError(
                        "garbage_collect_threshold too small, set it 0.0~1.0"
                    )
                if not value < 1.0:
                    raise RuntimeError(
                        "garbage_collect_threshold too big, set it 0.0~1.0"
                    )
                cfg.garbage_collection_threshold = value
            elif key == "roundup_power2_divisions":
                i = self.parse_roundup(i)
            elif key == "backend":
                self.consume(i + 1, ":")
                i += 2
                if not self.has(i):
                    raise RuntimeError("Error parsing backend value")
                value = tokens[i]
                if value not in ("native", "cudaMallocAsync"):
                    raise RuntimeError(
                        "Unknown allocator backend, options are native and "
                        "cudaMallocAsync"
                    )
                if self.runtime and value != cfg.backend:
                    raise RuntimeError(
                        "Allocator backend parsed at runtime != allocator "
                        "backend parsed at load time"
                    )
                cfg.backend = value
            elif key in (
                "expandable_segments",
                "release_lock_on_hipmalloc",
                "release_lock_on_cudamalloc",
            ):
                self.consume(i + 1, ":")
                i += 2
                option = (
                    "expandable_segments"
                    if key == "expandable_segments"
                    else "release_lock_on_cudamalloc"
                )
                if not (self.has(i) and tokens[i] in ("True", "False")):
                    raise RuntimeError(
                        f"Expected a single True/False argument for {option}"
                    )
                if key == "expandable_segments":
                    cfg.expandable_segments = tokens[i] == "True"
                else:
                    cfg.release_lock_on_cudamalloc = tokens[i] == "True"
            elif key in (
                "pinned_use_hip_host_register",
                "pinned_use_cuda_host_register",
                "pinned_use_background_threads",
                "graph_capture_record_stream_reuse",
            ):
                option = (
                    "pinned_use_cuda_host_register"
                    if key == "pinned_use_hip_host_register"
                    else key
                )
                self.consume(i + 1, ":")
                i += 2
                if not self.has(i):
                    raise RuntimeError(f"Error, expecting {option} value")
                if tokens[i] not in ("True", "False"):
                    raise RuntimeError(
                        f"Expected a single True/False argument for {option}"
                    )
                setattr(cfg, option, tokens[i] == "True")
            elif key == "pinned_num_register_threads":
                self.consume(i + 1, ":")
                i += 2
                if not self.has(i):
                    raise RuntimeError(
                        "Error, expecting pinned_num_register_threads value"
                    )
                value = _size_t(_c_integer(tokens[i], "stoi"))
                if not _is_power_of_2(value):
                    raise RuntimeError(
                        "Number of register threads has to be power of 2 "
                    )
                if value > _PINNED_MAX_REGISTER_THREADS:
                    raise RuntimeError(
                        "Number of register threads should be less than or "
                        f"equal to {_PINNED_MAX_REGISTER_THREADS}"
                    )
                cfg.pinned_num_register_threads = value
            else:
                raise RuntimeError(f"Unrecognized CachingAllocator option: {key}")
            if i + 1 < len(tokens):
                self.consume(i + 1, ",")
                i += 1
            i += 1

    def parse_mb_size(self, i: int, key: str) -> int:
        self.consume(i + 1, ":")
        i += 2
        if not self.has(i):
            raise RuntimeError(f"Error, expecting {key} value")
        value = _size_t(_c_integer(self.tokens[i], "stoi"))
        minimum = _K_LARGE_BUFFER // _MB
        if not value > minimum:
            raise RuntimeError(
                f"CachingAllocator option {key} too small, must be > {minimum}"
            )
        value = min(value, _SIZE_MAX // _MB)
        if key == "max_split_size_mb":
            self.config.max_split_size = value * _MB
        else:
            self.config.max_non_split_rounding_size = value * _MB
        return i

    def parse_roundup(self, i: int) -> int:
        tokens = self.tokens
        divisions = self.config.roundup_power2_divisions
        self.consume(i + 1, ":")
        i += 2
        if not self.has(i):
            raise RuntimeError("Error, expecting roundup_power2_divisions value")
        first_value = True
        if tokens[i] == "[":
            last_index = 0
            i += 1
            while i < len(tokens) and tokens[i] != "]":
                boundary = tokens[i]
                self.consume(i + 1, ":")
                i += 2
                if not self.has(i):
                    raise RuntimeError("Error parsing roundup_power2_divisions value")
                value = _size_t(_c_integer(tokens[i], "stoi"))
                if not (value == 0 or _is_power_of_2(value)):
                    raise RuntimeError(
                        "For roundups, the divisions has to be power of 2 or 0 "
                        "to disable roundup "
                    )
                if boundary == ">":
                    for index in range(last_index, _ROUNDUP_INTERVALS):
                        divisions[index] = value
                else:
                    interval = _c_integer(boundary, "stoul")
                    if not _is_power_of_2(interval):
                        raise RuntimeError(
                            "For roundups, the intervals have to be power of 2 "
                        )
                    index = min(_log2_floor(interval), _ROUNDUP_INTERVALS - 1)
                    if first_value:
                        for fill in range(index):
                            divisions[fill] = value
                        first_value = False
                    divisions[index] = value
                    last_index = index
                if i + 1 >= len(tokens):
                    # config[i + 1] past the end is undefined in C++; treat
                    # the list as unterminated.
                    raise RuntimeError(
                        "Error parsing CachingAllocator settings, expected ,"
                    )
                if tokens[i + 1] != "]":
                    self.consume(i + 1, ",")
                    i += 1
                i += 1
        else:
            value = _size_t(_c_integer(tokens[i], "stoi"))
            if not _is_power_of_2(value):
                raise RuntimeError("For roundups, the divisions has to be power of 2 ")
            for index in range(_ROUNDUP_INTERVALS):
                divisions[index] = value
        return i


class _ConfigTokenizer:
    """``c10::CachingAllocator::ConfigTokenizer`` (PyTorch 2.10+)."""

    def __init__(self, env: str):
        tokens: list[str] = []
        buffer = ""
        for ch in env:
            if ch in ",:[]":
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(ch)
            elif not ch.isspace():
                buffer += ch
        if buffer:
            tokens.append(buffer)
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def check_index(self, i: int) -> None:
        if i >= len(self.tokens):
            # TORCH_INTERNAL_ASSERT; torch prefixes the assert location.
            raise RuntimeError("Index out of bounds in ConfigTokenizer")

    def __getitem__(self, i: int) -> str:
        self.check_index(i)
        return self.tokens[i]

    def check_token(self, i: int, token: str) -> bool:
        # The parsers ignore the result; only the index check can fail.
        self.check_index(i)
        return self.tokens[i] == token

    def to_size_t(self, i: int) -> int:
        self.check_index(i)
        return _c_integer(self.tokens[i], "stoull")

    def to_double(self, i: int) -> float:
        self.check_index(i)
        return _c_double(self.tokens[i])

    def to_bool(self, i: int) -> bool:
        self.check_index(i)
        token = self.tokens[i]
        if token == "True":
            return True
        if token == "False":
            return False
        raise ValueError(
            f"Expected 'True' or 'False' at index {i} in ConfigTokenizer but "
            f"got '{token}'"
        )

    def skip_key(self, i: int) -> int:
        i += 1
        self.check_token(i, ":")
        i += 1
        self.check_index(i)
        if self.tokens[i] != "[":
            return i
        i += 1
        while i < len(self.tokens) and self.tokens[i] != "]":
            i += 1
        if i >= len(self.tokens):
            raise RuntimeError(
                "Expected closing bracket ']' in ConfigTokenizer but reached "
                "end of config"
            )
        return i


def _accelerator_keys(version: tuple[int, int]) -> set[str]:
    keys = {
        "max_split_size_mb",
        "max_non_split_rounding_mb",
        "garbage_collection_threshold",
        "roundup_power2_divisions",
        "expandable_segments",
        "pinned_use_background_threads",
    }
    if version >= (2, 11):
        keys.add("large_segment_size_mb")
    if version >= (2, 13):
        keys.update({"pinned_max_round_threshold_mb", "pinned_max_cached_size_mb"})
    return keys


def _cuda_keys(version: tuple[int, int]) -> set[str]:
    keys = {
        "backend",
        "release_lock_on_cudamalloc",
        "pinned_use_cuda_host_register",
        "release_lock_on_hipmalloc",
        "pinned_use_hip_host_register",
        "graph_capture_record_stream_reuse",
        "pinned_reserve_segment_size_mb",
        "pinned_num_register_threads",
        "per_process_memory_fraction",
    }
    if version >= (2, 12):
        keys.update({"pinned_free_catch_all", "throw_on_cudamalloc_oom"})
    return keys


def _parse_cuda_config(
    env: str,
    config: _AllocatorConfig,
    version: tuple[int, int],
    *,
    runtime: bool,
) -> None:
    """``CUDAAllocatorConfig::parseArgs`` (PyTorch 2.10+)."""
    tok = _ConfigTokenizer(env)
    known = _accelerator_keys(version) | _cuda_keys(version)
    late_keys = {"pinned_free_catch_all", "throw_on_cudamalloc_oom"}
    i = 0
    while i < len(tok):
        key = tok[i]
        if key == "backend":
            tok.check_token(i + 1, ":")
            i += 2
            value = tok[i]
            if value not in ("native", "cudaMallocAsync"):
                raise ValueError(
                    "Unknown allocator backend, options are native and "
                    "cudaMallocAsync"
                )
            if runtime and value != config.backend:
                raise RuntimeError(
                    "Allocator backend parsed at runtime != allocator backend "
                    f"parsed at load time, {value} != {config.backend}"
                )
            config.backend = value
        elif key in ("release_lock_on_hipmalloc", "release_lock_on_cudamalloc"):
            tok.check_token(i + 1, ":")
            i += 2
            config.release_lock_on_cudamalloc = tok.to_bool(i)
        elif key in ("pinned_use_hip_host_register", "pinned_use_cuda_host_register"):
            tok.check_token(i + 1, ":")
            i += 2
            config.pinned_use_cuda_host_register = tok.to_bool(i)
        elif key == "pinned_num_register_threads":
            tok.check_token(i + 1, ":")
            i += 2
            value = tok.to_size_t(i)
            if not _is_power_of_2(value):
                raise ValueError(
                    f"Number of register threads has to be power of 2, got {value}"
                )
            if value > _PINNED_MAX_REGISTER_THREADS:
                raise ValueError(
                    "Number of register threads should be less than or equal "
                    f"to {_PINNED_MAX_REGISTER_THREADS}, got {value}"
                )
            config.pinned_num_register_threads = value
        elif key == "pinned_reserve_segment_size_mb":
            tok.check_token(i + 1, ":")
            i += 2
            value = tok.to_size_t(i)
            if not value > 0:
                raise ValueError(
                    "Pinned reserve segment size has to be greater than 0"
                )
            config.pinned_reserve_segment_size_mb = value
        elif key == "graph_capture_record_stream_reuse":
            tok.check_token(i + 1, ":")
            i += 2
            config.graph_capture_record_stream_reuse = tok.to_bool(i)
        elif key == "per_process_memory_fraction":
            tok.check_token(i + 1, ":")
            i += 2
            value = tok.to_double(i)
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    "per_process_memory_fraction is invalid, set it in [0.0, 1.0]"
                )
            config.per_process_memory_fraction = value
        elif key in late_keys and version >= (2, 12):
            tok.check_token(i + 1, ":")
            i += 2
            setattr(config, key, tok.to_bool(i))
        else:
            if key not in known:
                raise ValueError(f"Unrecognized key '{key}' in CUDA allocator config.")
            i = tok.skip_key(i)
        if i + 1 < len(tok):
            tok.check_token(i + 1, ",")
            i += 1
        i += 1


def _parse_accelerator_config(
    env: str,
    config: _AllocatorConfig,
    version: tuple[int, int],
) -> None:
    """``AcceleratorAllocatorConfig::parseArgs`` (PyTorch 2.10+)."""
    config.max_split_size = _SIZE_MAX
    config.roundup_power2_divisions = [0] * _ROUNDUP_INTERVALS
    config.garbage_collection_threshold = 0.0
    config.last_allocator_settings = env
    tok = _ConfigTokenizer(env)
    known = _accelerator_keys(version) | _cuda_keys(version)

    if version >= (2, 11):
        # large_segment_size_mb is read first: the split limits depend on it.
        i = 0
        while i < len(tok):
            if tok[i] == "large_segment_size_mb":
                tok.check_token(i + 1, ":")
                i += 2
                value = tok.to_size_t(i)
                minimum = _K_MIN_LARGE_ALLOC // _MB
                if not value > minimum:
                    raise ValueError(
                        "CachingAllocator option large_segment_size_mb must be > "
                        f"{minimum} MB"
                    )
                config.large_segment_size = min(value, _SIZE_MAX // _MB) * _MB
            i += 1
        config.max_non_split_rounding_size = config.large_segment_size
        minimum_split_mb = config.large_segment_size // _MB
    else:
        minimum_split_mb = _K_LARGE_BUFFER // _MB

    i = 0
    while i < len(tok):
        key = tok[i]
        if key == "large_segment_size_mb" and version >= (2, 11):
            i = tok.skip_key(i)
        elif key in ("max_split_size_mb", "max_non_split_rounding_mb"):
            tok.check_token(i + 1, ":")
            i += 2
            value = tok.to_size_t(i)
            if not value >= minimum_split_mb:
                raise ValueError(
                    f"CachingAllocator option {key} too small, must be >= "
                    f"{minimum_split_mb}"
                )
            value = min(value, _SIZE_MAX // _MB) * _MB
            if key == "max_split_size_mb":
                config.max_split_size = value
            else:
                config.max_non_split_rounding_size = value
        elif key == "garbage_collection_threshold":
            tok.check_token(i + 1, ":")
            i += 2
            value = tok.to_double(i)
            if not (value > 0 and value < 1.0):
                raise ValueError(
                    "garbage_collect_threshold is invalid, set it in (0.0, 1.0)"
                )
            config.garbage_collection_threshold = value
        elif key == "roundup_power2_divisions":
            i = _parse_accelerator_roundup(tok, i, config)
        elif key == "expandable_segments":
            tok.check_token(i + 1, ":")
            i += 2
            config.expandable_segments = tok.to_bool(i)
        elif key == "pinned_use_background_threads":
            tok.check_token(i + 1, ":")
            i += 2
            config.pinned_use_background_threads = tok.to_bool(i)
        elif key in (
            "pinned_max_round_threshold_mb",
            "pinned_max_cached_size_mb",
        ) and version >= (2, 13):
            tok.check_token(i + 1, ":")
            i += 2
            value = min(tok.to_size_t(i), _SIZE_MAX // _MB) * _MB
            if key == "pinned_max_round_threshold_mb":
                config.pinned_max_round_threshold = value
            else:
                config.pinned_max_cached_size = value
        else:
            # CUDA registers its parser hook, so unknown keys are rejected.
            if key not in known:
                raise ValueError(
                    f"Unrecognized key '{key}' in Accelerator allocator config."
                )
            i = tok.skip_key(i)
        if i + 1 < len(tok):
            tok.check_token(i + 1, ",")
            i += 1
        i += 1


def _parse_accelerator_roundup(
    tok: _ConfigTokenizer, i: int, config: _AllocatorConfig
) -> int:
    divisions = config.roundup_power2_divisions
    tok.check_token(i + 1, ":")
    i += 2
    first_value = True
    if tok[i] == "[":
        last_index = 0
        i += 1
        while i < len(tok) and tok[i] != "]":
            value_index = i
            tok.check_token(i + 1, ":")
            i += 2
            value = tok.to_size_t(i)
            if not (value == 0 or _is_power_of_2(value)):
                raise ValueError(
                    "For roundups, the divisions has to be power of 2 or 0 to "
                    "disable roundup "
                )
            if tok[value_index] == ">":
                for index in range(last_index + 1, _ROUNDUP_INTERVALS):
                    divisions[index] = value
            else:
                boundary = tok.to_size_t(value_index)
                if not _is_power_of_2(boundary):
                    raise ValueError(
                        "For roundups, the intervals have to be power of 2 "
                    )
                index = min(_log2_floor(boundary), _ROUNDUP_INTERVALS - 1)
                if first_value:
                    for fill in range(index):
                        divisions[fill] = value
                    first_value = False
                divisions[index] = value
                last_index = index
            if tok[i + 1] != "]":
                tok.check_token(i + 1, ",")
                i += 1
            i += 1
        if i >= len(tok):
            raise RuntimeError(
                "Expected closing bracket ']' in ConfigTokenizer but reached end "
                "of config"
            )
    else:
        value = tok.to_size_t(i)
        if not _is_power_of_2(value):
            raise ValueError("For roundups, the divisions has to be power of 2 ")
        for index in range(_ROUNDUP_INTERVALS):
            divisions[index] = value
    return i


def _environment_allocator_settings(
    environ: Any, version: tuple[int, int]
) -> tuple[str | None, str | None]:
    """Return the (CUDA parser, accelerator parser) settings strings.

    Up to 2.9 only ``PYTORCH_CUDA_ALLOC_CONF`` configures the CUDA
    allocator.  From 2.10 the CUDA parser reads ``PYTORCH_CUDA_ALLOC_CONF``
    and then ``PYTORCH_ALLOC_CONF``; the accelerator parser also accepts
    ``PYTORCH_HIP_ALLOC_CONF`` in between.
    """
    cuda = environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if version < (2, 10):
        return cuda, None
    unified = environ.get("PYTORCH_ALLOC_CONF")
    hip = environ.get("PYTORCH_HIP_ALLOC_CONF")
    cuda_env = cuda if cuda is not None else unified
    accelerator_env = next(
        (value for value in (cuda, hip, unified) if value is not None), None
    )
    return cuda_env, accelerator_env


def _parse_allocator_settings(
    env: str | None,
    accelerator_env: str | None,
    config: _AllocatorConfig,
    version: tuple[int, int],
    *,
    runtime: bool,
) -> None:
    if version < (2, 10):
        if env is not None:
            _LegacyConfigParser(env, config, runtime=runtime).parse()
        return
    if runtime:
        # c10::CachingAllocator::setAllocatorSettings(): the accelerator
        # parser first, then the CUDA hook.
        _parse_accelerator_config(env or "", config, version)
        _parse_cuda_config(env or "", config, version, runtime=True)
        return
    # At CUDA initialization the CUDA config is read first.
    if env is not None:
        _parse_cuda_config(env, config, version, runtime=False)
    if accelerator_env is not None:
        _parse_accelerator_config(accelerator_env, config, version)


# ---------------------------------------------------------------------------
# Statistics (c10/core/CachingDeviceAllocator.h)
# ---------------------------------------------------------------------------

# c10::CachingAllocator::StatType order and the names memory_stats() uses.
_STAT_TYPE_NAMES = ("all", "small_pool", "large_pool")
_STAT_ARRAY_NAMES = (
    "allocation",
    "segment",
    "active",
    "inactive_split",
    "allocated_bytes",
    "reserved_bytes",
    "active_bytes",
    "inactive_split_bytes",
    "requested_bytes",
)
# DeviceStats.max_split_size is the size_t limit cast to int64_t when
# max_split_size_mb is not configured.
_UNLIMITED_MAX_SPLIT_SIZE = -1


class _Stat:
    """Mirror of ``c10::CachingAllocator::Stat``."""

    __slots__ = ("current", "peak", "allocated", "freed")

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self.allocated = 0
        self.freed = 0

    def increase(self, amount: int) -> None:
        self.current += amount
        self.peak = max(self.current, self.peak)
        self.allocated += amount

    def decrease(self, amount: int) -> None:
        self.current -= amount
        self.freed += amount

    def as_dict(self) -> dict[str, int]:
        return {
            "current": self.current,
            "peak": self.peak,
            "allocated": self.allocated,
            "freed": self.freed,
        }


class _DeviceStats:
    """Mirror of ``c10::CachingDeviceAllocator::DeviceStats`` for one device."""

    def __init__(self, version: tuple[int, int] = _LATEST_MODELED_VERSION) -> None:
        self.version = version
        self.arrays = {
            name: tuple(_Stat() for _ in _STAT_TYPE_NAMES)
            for name in _STAT_ARRAY_NAMES
        }
        self.num_alloc_retries = 0
        self.num_ooms = 0
        self.num_sync_all_streams = 0
        self.num_device_alloc = 0
        self.num_device_free = 0
        self.num_oom_rejections = 0
        self.max_split_size = _UNLIMITED_MAX_SPLIT_SIZE
        self.oversize_allocations = _Stat()
        self.oversize_segments = _Stat()

    def change(self, name: str, small: bool, delta: int) -> None:
        if delta == 0:
            return
        stats = self.arrays[name]
        for stat in (stats[0], stats[1 if small else 2]):
            if delta > 0:
                stat.increase(delta)
            else:
                stat.decrease(-delta)

    def total(self, name: str) -> _Stat:
        return self.arrays[name][0]

    def reset_accumulated(self) -> None:
        for stats in self.arrays.values():
            for stat in stats:
                stat.allocated = 0
                stat.freed = 0
        for stat in (self.oversize_allocations, self.oversize_segments):
            stat.allocated = 0
            stat.freed = 0
        self.num_alloc_retries = 0
        self.num_ooms = 0
        self.num_sync_all_streams = 0
        self.num_device_alloc = 0
        self.num_device_free = 0
        self.num_oom_rejections = 0

    def reset_peak(self) -> None:
        for stats in self.arrays.values():
            for stat in stats:
                stat.peak = stat.current
        for stat in (self.oversize_allocations, self.oversize_segments):
            stat.peak = stat.current

    def as_nested_dict(self) -> dict[str, Any]:
        """Return the dictionary ``torch._C._cuda_memoryStats`` builds."""
        result: dict[str, Any] = {
            "num_alloc_retries": self.num_alloc_retries,
            "num_ooms": self.num_ooms,
            "max_split_size": self.max_split_size,
            "num_sync_all_streams": self.num_sync_all_streams,
            "num_device_alloc": self.num_device_alloc,
            "num_device_free": self.num_device_free,
        }
        if self.version >= (2, 12):
            result["num_oom_rejections"] = self.num_oom_rejections

        def _array(name: str) -> dict[str, Any]:
            return {
                type_name: stat.as_dict()
                for type_name, stat in zip(_STAT_TYPE_NAMES, self.arrays[name])
            }

        for name in _STAT_ARRAY_NAMES:
            result[name] = _array(name)
            if name == "reserved_bytes" and self.version >= (2, 13):
                # Only the default pool exists here.
                result["reserved_bytes_by_private_pools"] = {}
        result["oversize_allocations"] = self.oversize_allocations.as_dict()
        result["oversize_segments"] = self.oversize_segments.as_dict()
        return result


def _flatten_memory_stats(nested: dict[str, Any]) -> OrderedDict[str, Any]:
    """Flatten nested stats the way ``torch.cuda.memory_stats`` does."""
    result: list[tuple[str, Any]] = []

    def _add(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            if prefix:
                prefix += "."
            for key, item in value.items():
                _add(prefix + str(key), item)
        else:
            result.append((prefix, value))

    _add("", nested)
    result.sort()
    return OrderedDict(result)


def _host_memory_stats(version: tuple[int, int]) -> dict[str, Any]:
    """``torch._C._cuda_hostMemoryStats()`` for an unused pinned allocator."""

    def _stat() -> dict[str, int]:
        return {"current": 0, "peak": 0, "allocated": 0, "freed": 0}

    def _duration() -> dict[str, int]:
        return {"total": 0, "max": 0, "min": 0, "count": 0, "avg": 0}

    if version < (2, 10):
        names = ("allocation", "segment", "allocated_bytes", "reserved_bytes")
    else:
        names = ("allocations", "active_requests", "allocated_bytes", "active_bytes")
    result: dict[str, Any] = {"num_host_alloc": 0, "num_host_free": 0}
    for name in names:
        result[name] = _stat()
    result["host_alloc_time"] = _duration()
    result["host_free_time"] = _duration()
    return result


# ---------------------------------------------------------------------------
# Blocks, pools and segments (CUDACachingAllocator.cpp)
# ---------------------------------------------------------------------------

# RecordContext levels and trace actions.
_RECORD_NEVER, _RECORD_STATE, _RECORD_ALLOC, _RECORD_ALL = 0, 1, 2, 3
_TRACE_ACTIONS = (
    "alloc",
    "free_requested",
    "free_completed",
    "segment_alloc",
    "segment_free",
    "segment_map",
    "segment_unmap",
    "snapshot",
    "oom",
    "annotate",
)
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class _Block:
    __slots__ = (
        "device",
        "stream",
        "stream_uses",
        "size",
        "requested_size",
        "pool",
        "address",
        "allocated",
        "mapped",
        "prev",
        "next",
        "event_count",
        "gc_count_base",
        "context",
        "segment_context",
        "expandable",
    )

    def __init__(
        self, device: int, stream: int, size: int, pool: "_BlockPool", address: int
    ) -> None:
        self.device = device
        self.stream = stream
        self.stream_uses: set[int] = set()
        self.size = size
        self.requested_size = 0
        self.pool = pool
        self.address = address
        self.allocated = False
        self.mapped = True
        self.prev: _Block | None = None
        self.next: _Block | None = None
        self.event_count = 0
        self.gc_count_base = 0
        self.context: list[dict[str, Any]] | None = None
        self.segment_context: list[dict[str, Any]] | None = None
        self.expandable: _ExpandableSegment | None = None

    def is_split(self) -> bool:
        return self.prev is not None or self.next is not None

    def gc_count(self) -> int:
        return self.pool.get_free_blocks_call_count - self.gc_count_base

    def splice(self, before: "_Block | None", after: "_Block | None") -> None:
        if before is not None:
            before.next = self
        self.prev = before
        if after is not None:
            after.prev = self
        self.next = after


class _BlockPool:
    """``BlockPool``: free blocks ordered by (stream, size, address).

    PyTorch 2.11+ also orders by a registration counter that grows with each
    new segment; synthetic addresses grow the same way, so the order matches.
    """

    def __init__(self, is_small: bool) -> None:
        self.is_small = is_small
        self.keys: list[tuple[int, int, int]] = []
        self.by_key: dict[tuple[int, int, int], _Block] = {}
        self.unmapped_keys: list[tuple[int, int]] = []
        self.unmapped: dict[tuple[int, int], _Block] = {}
        self.get_free_blocks_call_count = 0

    def insert(self, block: _Block) -> None:
        block.gc_count_base = self.get_free_blocks_call_count
        key = (block.stream, block.size, block.address)
        insort(self.keys, key)
        self.by_key[key] = block

    def erase(self, block: _Block) -> bool:
        key = (block.stream, block.size, block.address)
        if self.by_key.get(key) is not block:
            return False
        del self.by_key[key]
        self.keys.pop(bisect_left(self.keys, key))
        return True

    def contains(self, block: _Block) -> bool:
        return self.by_key.get((block.stream, block.size, block.address)) is block

    def lower_bound(self, stream: int, size: int) -> int:
        return bisect_left(self.keys, (stream, size, -1))

    def at(self, index: int) -> _Block:
        return self.by_key[self.keys[index]]

    def __len__(self) -> int:
        return len(self.keys)

    def blocks(self) -> list[_Block]:
        return [self.by_key[key] for key in self.keys]

    def insert_unmapped(self, block: _Block) -> None:
        key = (block.stream, block.address)
        insort(self.unmapped_keys, key)
        self.unmapped[key] = block

    def erase_unmapped(self, block: _Block) -> bool:
        key = (block.stream, block.address)
        if self.unmapped.get(key) is not block:
            return False
        del self.unmapped[key]
        self.unmapped_keys.pop(bisect_left(self.unmapped_keys, key))
        return True

    def unmapped_from(self, stream: int) -> list[_Block]:
        start = bisect_left(self.unmapped_keys, (stream, -1))
        result = []
        for key in self.unmapped_keys[start:]:
            if key[0] != stream:
                break
            result.append(self.unmapped[key])
        return result


class _ExpandableSegment:
    """``ExpandableSegment``: a virtual range mapped page by page."""

    def __init__(self, address: int, segment_size: int, device_total: int) -> None:
        self.ptr = address
        self.segment_size = segment_size
        # Address space for 1 1/8 of the device memory.
        total = device_total + device_total // 8
        self.max_handles = (total + segment_size - 1) // segment_size
        self.handles: list[bool] = []

    def size(self) -> int:
        return self.max_handles * self.segment_size

    def _left(self, address: int) -> int:
        return (address - self.ptr) // self.segment_size

    def _right(self, address: int) -> int:
        return (address - self.ptr + self.segment_size - 1) // self.segment_size

    def map(self, address: int, size: int, available: int) -> tuple[int, int]:
        begin = self._left(address)
        end = self._right(address + size)
        start = self.ptr + self.segment_size * begin
        if begin == end:
            return start, 0
        needed = (end - begin) * self.segment_size
        if needed > available:
            # cuMemCreate fails with CUDA_ERROR_OUT_OF_MEMORY; the handles
            # created so far are released again.
            return start, 0
        if end > len(self.handles):
            self.handles.extend([False] * (end - len(self.handles)))
        for index in range(begin, end):
            self.handles[index] = True
        return start, needed

    def unmap(self, address: int, size: int) -> tuple[int, int]:
        begin = self._right(address)
        end = self._left(address + size)
        if begin >= end:
            return address, 0
        for index in range(begin, end):
            self.handles[index] = False
        while self.handles and not self.handles[-1]:
            self.handles.pop()
        return self.ptr + self.segment_size * begin, (end - begin) * self.segment_size


class _RingBuffer:
    """``c10::RingBuffer`` holding the allocation trace."""

    def __init__(self) -> None:
        self.max_entries = 1
        self.entries: list[dict[str, Any]] = []
        self.next = 0

    def set_max_entries(self, size: int) -> None:
        self.max_entries = max(1, int(size))

    def insert(self, entry: dict[str, Any]) -> None:
        if len(self.entries) < self.max_entries:
            self.entries.append(entry)
        else:
            self.entries[self.next] = entry
            self.next += 1
            if self.next == self.max_entries:
                self.next = 0

    def get(self) -> list[dict[str, Any]]:
        return self.entries[self.next :] + self.entries[: self.next]

    def clear(self) -> None:
        self.next = 0
        self.entries = []


class _AllocParams:
    __slots__ = ("size", "stream", "pool", "alloc_size", "expandable", "block")

    def __init__(
        self,
        size: int,
        stream: int,
        pool: _BlockPool,
        alloc_size: int,
        expandable: bool,
    ) -> None:
        self.size = size
        self.stream = stream
        self.pool = pool
        self.alloc_size = alloc_size
        self.expandable = expandable
        self.block: _Block | None = None


def _gather_python_frames() -> list[dict[str, Any]]:
    """Python frames for an allocation context, innermost first.

    FakeGPU's own frames are left out so the trace starts at the caller.
    PyTorch's records also carry C++ frames; FakeCUDA has none.
    """
    frames: list[dict[str, Any]] = []
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith(_PACKAGE_DIR):
            frames.append(
                {
                    "name": frame.f_code.co_name,
                    "filename": filename,
                    "line": frame.f_lineno,
                }
            )
        frame = frame.f_back
    return frames


class _OutOfMemory(Exception):
    """Raised inside the allocator; the tracker turns it into torch's error."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _DeviceAllocator:
    """``DeviceCachingAllocator`` for one device."""

    def __init__(
        self,
        tracker: "_DeviceMemoryTracker",
        device: int,
        total: int,
    ) -> None:
        self.tracker = tracker
        self.config = tracker._config
        self.version = tracker._version
        self.device = device
        self.total = int(total)
        self.stats = _DeviceStats(self.version)
        max_split = self.config.max_split_size
        self.stats.max_split_size = _int64(max_split)
        self.small = _BlockPool(True)
        self.large = _BlockPool(False)
        # active_blocks: allocated blocks and freed blocks with pending events.
        self.active: dict[int, _Block] = {}
        self.total_allocated_memory = 0
        self.allowed_memory_maximum: int | None = None
        if self.version >= (2, 10):
            self.set_memory_fraction(self.config.per_process_memory_fraction)
        self.expandable_segments: list[_ExpandableSegment] = []
        self.cuda_events: dict[int, deque[_Block]] = {}
        self.next_address = _SNAPSHOT_ADDRESS_BASE + device * _DEVICE_ADDRESS_STRIDE
        self.record_history = False
        self.record_context = _RECORD_NEVER
        self.alloc_buffer = _RingBuffer()
        self.skip_actions: set[str] = set()

    # -- configuration -----------------------------------------------------

    @property
    def max_split_size(self) -> int:
        return self.config.max_split_size

    def set_memory_fraction(self, fraction: float) -> None:
        if self.version < (2, 10):
            # Legacy: any call caps cudaMalloc, even at 1.0 (set_fraction).
            self.allowed_memory_maximum = int(fraction * float(self.total))
            return
        if not 0 <= fraction <= 1:
            raise RuntimeError(f"invalid fraction:{fraction}. Please set within [0, 1].")
        self.allowed_memory_maximum = None
        if fraction < 1.0:
            value = fraction * float(self.total)
            allowed = int(value)
            # std::round() from 2.13 on rounds halves away from zero.
            if self.version >= (2, 13) and value - allowed >= 0.5:
                allowed += 1
            self.allowed_memory_maximum = allowed

    def get_memory_fraction(self) -> float:
        if self.allowed_memory_maximum is None:
            return 1.0
        return self.allowed_memory_maximum / float(self.total)

    # -- history -------------------------------------------------------------

    def gather_context(self, level: int) -> list[dict[str, Any]] | None:
        if self.record_context < level:
            return None
        return _gather_python_frames()

    def record_trace(
        self,
        action: str,
        addr: int,
        size: int,
        stream: int,
        context: list[dict[str, Any]] | None,
    ) -> None:
        if not self.record_history or action in self.skip_actions:
            return
        self.alloc_buffer.insert(
            {
                "action": action,
                "addr": addr,
                "size": size,
                "stream": stream,
                "time_us": time.time_ns() // 1000,
                "context": context if self.record_context >= _RECORD_ALLOC else None,
            }
        )

    # -- sizes ---------------------------------------------------------------

    def round_size(self, size: int) -> int:
        if size < _K_MIN_BLOCK_SIZE:
            return _K_MIN_BLOCK_SIZE
        divisions = self.config.divisions_for(size)
        if divisions > 1 and size > _K_MIN_BLOCK_SIZE * divisions:
            return _roundup_power2_next_division(size, divisions)
        return _K_MIN_BLOCK_SIZE * ((size + _K_MIN_BLOCK_SIZE - 1) // _K_MIN_BLOCK_SIZE)

    def allocation_size(self, size: int) -> int:
        if size <= _K_SMALL_SIZE:
            return _K_SMALL_BUFFER
        if size < _K_MIN_LARGE_ALLOC:
            large = self.config.large_segment_size if self.version >= (2, 11) else _K_LARGE_BUFFER
            return large
        return _K_ROUND_LARGE * ((size + _K_ROUND_LARGE - 1) // _K_ROUND_LARGE)

    def large_segment_size(self) -> int:
        if self.version >= (2, 11):
            return self.config.large_segment_size
        return _K_LARGE_BUFFER

    def new_address(self, size: int) -> int:
        address = self.next_address
        self.next_address += (size + _K_ROUND_LARGE - 1) // _K_ROUND_LARGE * _K_ROUND_LARGE
        return address

    def gc_enabled(self) -> bool:
        return (
            self.allowed_memory_maximum is not None
            and self.config.garbage_collection_threshold > 0.0
        )

    # -- malloc --------------------------------------------------------------

    def malloc(self, orig_size: int, stream: int) -> _Block:
        context = self.gather_context(_RECORD_STATE)
        self.process_events(context)
        size = self.round_size(orig_size)
        pool = self.small if size <= _K_SMALL_SIZE else self.large
        alloc_size = self.allocation_size(size)
        params = _AllocParams(
            size, stream, pool, alloc_size, self.config.expandable_segments
        )
        found = self.get_free_block(params)
        if not found:
            if self.gc_enabled():
                self.garbage_collect_cached_blocks(context)
            found = self.alloc_block(params, False, context)
            if not found:
                found = (
                    self.release_available_cached_blocks(params, context)
                    and self.alloc_block(params, False, context)
                ) or (
                    self.release_cached_blocks(context)
                    and self.alloc_block(params, True, context)
                )
        if not found:
            raise _OutOfMemory(self.out_of_memory_message(params, context))
        split = self.should_split(params.block, params.size, params.expandable)
        return self.alloc_found_block(params, orig_size, context, split)

    def out_of_memory_message(
        self, params: _AllocParams, context: list[dict[str, Any]] | None
    ) -> str:
        device_free = self.tracker._device_free(self.device)
        self.record_trace("oom", device_free, params.size, params.stream, context)
        self.stats.num_ooms += 1
        allocated = self.stats.total("allocated_bytes").current
        reserved = self.stats.total("reserved_bytes").current
        allowed = self.allowed_memory_maximum
        allowed_info = "" if allowed is None else f"{_format_size(allowed)} allowed; "
        # The "Including non-PyTorch memory" sentence needs NVML process
        # accounting, which FakeCUDA does not model; PyTorch omits it too
        # when NVML is unavailable.
        message = (
            f"CUDA out of memory. Tried to allocate {_format_size(params.alloc_size)}. "
            f"GPU {self.device} has a total capacity of {_format_size(self.total)} "
            f"of which {_format_size(device_free)} is free. {allowed_info}"
            f"Of the allocated memory {_format_size(allocated)} is allocated by "
            f"PyTorch, and {_format_size(reserved - allocated)} is reserved by "
            "PyTorch but unallocated."
        )
        if self.version >= (2, 14) and self.config.expandable_segments:
            return message
        variable = (
            "PYTORCH_ALLOC_CONF" if self.version == (2, 10) else "PYTORCH_CUDA_ALLOC_CONF"
        )
        url = (
            "https://pytorch.org/docs/stable/notes/cuda.html#environment-variables"
            if self.version < (2, 11)
            else "https://docs.pytorch.org/docs/stable/notes/cuda.html"
            "#optimizing-memory-usage-with-pytorch-cuda-alloc-conf"
        )
        return (
            message
            + " If reserved but unallocated memory is large try setting "
            f"{variable}=expandable_segments:True to avoid fragmentation.  See "
            f"documentation for Memory Management  ({url})"
        )

    def get_free_block(self, p: _AllocParams) -> bool:
        pool = p.pool
        if self.gc_enabled():
            pool.get_free_blocks_call_count += 1
        index = pool.lower_bound(p.stream, p.size)
        if index >= len(pool) or pool.at(index).stream != p.stream:
            return False
        block = pool.at(index)
        if block.expandable is not None:
            if p.expandable:

                def expandable_size(b: _Block) -> int:
                    nxt = b.next
                    return b.size + (nxt.size if nxt is not None and not nxt.mapped else 0)

                nxt_index = index + 1
                while (
                    pool.at(index).expandable is not None
                    and nxt_index < len(pool)
                    and pool.at(nxt_index).stream == p.stream
                    and expandable_size(pool.at(nxt_index))
                    < expandable_size(pool.at(index))
                ):
                    index = nxt_index
                    nxt_index += 1
            else:
                index += 1
                while (
                    index < len(pool)
                    and pool.at(index).expandable is not None
                    and pool.at(index).stream == p.stream
                ):
                    index += 1
                if index >= len(pool) or pool.at(index).stream != p.stream:
                    return False
            block = pool.at(index)
        max_split = self.max_split_size
        if p.size < max_split and block.size >= max_split:
            return False
        if (
            p.size >= max_split
            and block.size >= p.size + self.config.max_non_split_rounding_size
        ):
            return False
        p.block = block
        pool.erase(block)
        return True

    def should_split(self, block: _Block, size: int, expandable: bool) -> bool:
        remaining = block.size - size
        if block.pool.is_small or expandable:
            return remaining >= _K_MIN_BLOCK_SIZE
        return size < self.max_split_size and remaining > _K_SMALL_SIZE

    def alloc_found_block(
        self,
        params: _AllocParams,
        orig_size: int,
        context: list[dict[str, Any]] | None,
        split_remainder: bool,
    ) -> _Block:
        size = params.size
        pool = params.pool
        small = pool.is_small
        block = params.block
        assert block is not None
        already_split = block.is_split()
        if split_remainder:
            remaining = block
            block = _Block(self.device, params.stream, size, pool, remaining.address)
            block.expandable = remaining.expandable
            block.prev = remaining.prev
            if block.prev is not None:
                block.prev.next = block
            block.next = remaining
            remaining.prev = block
            remaining.address += size
            remaining.size -= size
            pool.insert(remaining)
            if already_split and block.expandable is None:
                # An already-split inactive block shrinks by size bytes.
                self.stats.change("inactive_split_bytes", small, -block.size)
            elif block.expandable is None:
                self.stats.change("inactive_split_bytes", small, remaining.size)
                self.stats.change("inactive_split", small, 1)
        elif already_split and block.expandable is None:
            self.stats.change("inactive_split_bytes", small, -block.size)
            self.stats.change("inactive_split", small, -1)

        block.allocated = True
        block.requested_size = orig_size
        block.context = context
        self.record_trace("alloc", block.address, orig_size, block.stream, context)
        self.active[block.address] = block
        self.stats.change("allocation", small, 1)
        self.stats.change("allocated_bytes", small, block.size)
        self.stats.change("active", small, 1)
        self.stats.change("active_bytes", small, block.size)
        self.stats.change("requested_bytes", small, block.requested_size)
        if block.size >= self.max_split_size:
            self.stats.oversize_allocations.increase(1)
        return block

    def alloc_block(
        self,
        p: _AllocParams,
        is_retry: bool,
        context: list[dict[str, Any]] | None,
    ) -> bool:
        size = p.alloc_size
        if is_retry:
            self.stats.num_alloc_retries += 1
        if (
            self.allowed_memory_maximum is not None
            and self.total_allocated_memory + size > self.allowed_memory_maximum
        ):
            return False
        if p.expandable:
            p.block = self.try_allocate_expandable_block(p.stream, p.pool, p.size, context)
            return p.block is not None
        # cudaMalloc fails once the device memory is used up.
        if self.total_allocated_memory + size > self.total:
            return False
        self.total_allocated_memory += size
        block = _Block(self.device, p.stream, size, p.pool, self.new_address(size))
        self.stats.change("segment", p.pool.is_small, 1)
        self.stats.change("reserved_bytes", p.pool.is_small, size)
        if size >= self.max_split_size:
            self.stats.oversize_segments.increase(1)
        self.stats.num_device_alloc += 1
        self.record_trace("segment_alloc", block.address, size, p.stream, context)
        block.segment_context = context
        p.block = block
        return True

    # -- expandable segments -------------------------------------------------

    def find_expandable_block(
        self, stream: int, pool: _BlockPool, size: int
    ) -> _Block:
        def allocatable(b: _Block | None) -> bool:
            return (
                b is not None
                and not b.allocated
                and b.event_count == 0
                and not b.stream_uses
            )

        def has_available_address_space(b: _Block | None) -> bool:
            total = 0
            while total < size and allocatable(b):
                assert b is not None
                total += b.size
                b = b.next
            return total >= size

        for candidate in pool.unmapped_from(stream):
            c = candidate
            if allocatable(c.prev):
                c = c.prev
            if has_available_address_space(c):
                return c
        segment_size = _K_SMALL_BUFFER if pool.is_small else self.large_segment_size()
        segment = _ExpandableSegment(0, segment_size, self.total)
        segment.ptr = self.new_address(segment.size())
        self.expandable_segments.append(segment)
        candidate = _Block(self.device, stream, segment.size(), pool, segment.ptr)
        candidate.mapped = False
        candidate.expandable = segment
        pool.insert_unmapped(candidate)
        return candidate

    def map_block(
        self, to_map: _Block, size: int, context: list[dict[str, Any]] | None
    ) -> bool:
        segment = to_map.expandable
        assert segment is not None
        start, mapped = segment.map(
            to_map.address, size, self.total - self.total_allocated_memory
        )
        if mapped == 0:
            return False
        pool = to_map.pool
        pool.erase_unmapped(to_map)
        to_map.mapped = True
        if mapped < to_map.size:
            remaining = _Block(
                self.device,
                to_map.stream,
                to_map.size - mapped,
                pool,
                to_map.address + mapped,
            )
            remaining.mapped = False
            remaining.expandable = segment
            remaining.splice(to_map, to_map.next)
            pool.insert_unmapped(remaining)
            to_map.size = mapped
        self.try_merge_blocks(to_map, to_map.prev, pool)
        self.try_merge_blocks(to_map, to_map.next, pool)
        pool.insert(to_map)
        self.total_allocated_memory += mapped
        self.stats.change("reserved_bytes", pool.is_small, mapped)
        self.stats.num_device_alloc += 1
        self.record_trace("segment_map", start, mapped, to_map.stream, context)
        if to_map.prev is None and to_map.segment_context is None:
            to_map.segment_context = context
        return True

    def try_allocate_expandable_block(
        self,
        stream: int,
        pool: _BlockPool,
        size: int,
        context: list[dict[str, Any]] | None,
    ) -> _Block | None:
        candidate = self.find_expandable_block(stream, pool, size)
        if not candidate.mapped and not self.map_block(
            candidate, min(candidate.size, size), context
        ):
            return None
        while candidate.size < size:
            remaining = size - candidate.size
            new_candidate = candidate.next
            assert new_candidate is not None
            if not self.map_block(
                new_candidate, min(remaining, new_candidate.size), context
            ):
                return None
            candidate = new_candidate
        pool.erase(candidate)
        return candidate

    # -- free ----------------------------------------------------------------

    def free(self, block: _Block) -> None:
        context = self.gather_context(_RECORD_ALL)
        block.allocated = False
        small = block.pool.is_small
        self.stats.change("allocation", small, -1)
        self.stats.change("allocated_bytes", small, -block.size)
        self.record_trace(
            "free_requested",
            block.address,
            block.requested_size,
            block.stream,
            context if context is not None else block.context,
        )
        if block.size >= self.max_split_size:
            self.stats.oversize_allocations.decrease(1)
        if block.stream_uses:
            self.insert_events(block)
        else:
            self.free_block(block, context)

    def free_block(
        self, block: _Block, context: list[dict[str, Any]] | None
    ) -> None:
        self.record_trace(
            "free_completed",
            block.address,
            block.requested_size,
            block.stream,
            context if context is not None else block.context,
        )
        block.context = None
        original_size = block.size
        requested_size = block.requested_size
        pool = block.pool
        self.active.pop(block.address, None)
        net_blocks = 0
        net_bytes = 0
        for candidate in (block.prev, block.next):
            subsumed = self.try_merge_blocks(block, candidate, pool)
            if subsumed > 0:
                net_blocks -= 1
                net_bytes -= subsumed
        pool.insert(block)
        if block.is_split():
            net_blocks += 1
            net_bytes += block.size
        small = pool.is_small
        if block.expandable is None:
            self.stats.change("inactive_split", small, net_blocks)
            self.stats.change("inactive_split_bytes", small, net_bytes)
        self.stats.change("active", small, -1)
        self.stats.change("active_bytes", small, -original_size)
        self.stats.change("requested_bytes", small, -requested_size)
        if not self.tracker._caching_allocator and not block.is_split():
            # FAKEGPU_CACHING_ALLOCATOR=0 returns whole free segments at once.
            if block.expandable is None:
                self.release_block(block, context)

    def try_merge_blocks(
        self, dst: _Block, src: _Block | None, pool: _BlockPool
    ) -> int:
        if (
            src is None
            or src.allocated
            or src.event_count > 0
            or src.stream_uses
            or dst.mapped != src.mapped
        ):
            return 0
        if src.mapped:
            pool.erase(src)
        else:
            pool.erase_unmapped(src)
        if dst.prev is src:
            dst.address = src.address
            dst.prev = src.prev
            if dst.prev is not None:
                dst.prev.next = dst
            dst.segment_context = src.segment_context
        else:
            dst.next = src.next
            if dst.next is not None:
                dst.next.prev = dst
        dst.size += src.size
        return src.size

    def record_stream(self, block: _Block, stream: int) -> None:
        if stream == block.stream:
            return
        block.stream_uses.add(stream)

    def insert_events(self, block: _Block) -> None:
        streams = list(block.stream_uses)
        block.stream_uses = set()
        for stream in streams:
            block.event_count += 1
            self.cuda_events.setdefault(stream, deque()).append(block)

    def process_events(self, context: list[dict[str, Any]] | None) -> None:
        # FakeCUDA runs stream work synchronously, so every recorded event
        # has completed by the next allocation.
        for stream in list(self.cuda_events):
            queue = self.cuda_events.pop(stream)
            while queue:
                block = queue.popleft()
                block.event_count -= 1
                if block.event_count == 0:
                    self.free_block(block, context)

    def synchronize_and_free_events(self, context: list[dict[str, Any]] | None) -> None:
        self.stats.num_sync_all_streams += 1
        self.process_events(context)

    # -- releasing cached memory ----------------------------------------------

    def garbage_collect_cached_blocks(self, context: list[dict[str, Any]] | None) -> None:
        allowed = self.allowed_memory_maximum
        if allowed is None:
            return
        gc_threshold = int(self.config.garbage_collection_threshold * float(allowed))
        if self.total_allocated_memory <= gc_threshold:
            return
        target_size = self.total_allocated_memory - gc_threshold
        gc_reclaimed = 0
        total_age = 0
        freeable = 0
        for block in self.large.blocks():
            if not block.is_split():
                total_age += block.gc_count()
                freeable += 1
        if freeable == 0:
            return
        block_freed = True
        while gc_reclaimed < target_size and block_freed and freeable > 0:
            age_threshold = total_age / freeable
            block_freed = False
            for block in self.large.blocks():
                if (
                    not block.is_split()
                    and block.expandable is None
                    and block.gc_count() >= age_threshold
                ):
                    block_freed = True
                    gc_reclaimed += block.size
                    total_age -= block.gc_count()
                    freeable -= 1
                    self.release_block(block, context)

    def release_available_cached_blocks(
        self, p: _AllocParams, context: list[dict[str, Any]] | None
    ) -> bool:
        max_split = self.max_split_size
        if max_split == _SIZE_MAX:
            return False
        pool = p.pool
        key_size = max_split if p.size < max_split else p.size
        index = pool.lower_bound(p.stream, key_size)
        if (
            index >= len(pool)
            or pool.at(index).stream != p.stream
            or pool.at(index).expandable is not None
        ):
            if index == 0:
                return False
            total_released = 0
            index -= 1
            while (
                total_released < key_size
                and pool.at(index).size >= max_split
                and pool.at(index).stream == p.stream
            ):
                current = pool.at(index)
                is_first = index == 0
                if not is_first:
                    index -= 1
                if current.expandable is None:
                    total_released += current.size
                    self.release_block(current, context)
                if is_first:
                    break
            if total_released < key_size:
                return False
        else:
            self.release_block(pool.at(index), context)
        return True

    def release_cached_blocks(self, context: list[dict[str, Any]] | None) -> bool:
        self.synchronize_and_free_events(context)
        self.release_blocks(self.large, context)
        self.release_blocks(self.small, context)
        return True

    def release_block(self, block: _Block, context: list[dict[str, Any]] | None) -> None:
        self.stats.num_device_free += 1
        self.record_trace(
            "segment_free",
            block.address,
            block.size,
            block.stream,
            context if context is not None else block.segment_context,
        )
        self.total_allocated_memory -= block.size
        small = block.pool.is_small
        self.stats.change("segment", small, -1)
        self.stats.change("reserved_bytes", small, -block.size)
        if block.size >= self.max_split_size:
            self.stats.oversize_segments.decrease(1)
        block.pool.erase(block)

    def unmap_block(self, block: _Block, context: list[dict[str, Any]] | None) -> None:
        segment = block.expandable
        assert segment is not None
        start, unmapped = segment.unmap(block.address, block.size)
        if unmapped == 0:
            return
        pool = block.pool
        pool.erase(block)
        before_size = start - block.address
        if before_size > 0:
            before = _Block(self.device, block.stream, before_size, pool, block.address)
            before.expandable = segment
            before.splice(block.prev, block)
            pool.insert(before)
        after_size = block.size - (before_size + unmapped)
        if after_size > 0:
            after = _Block(
                self.device, block.stream, after_size, pool, start + unmapped
            )
            after.expandable = segment
            after.splice(block, block.next)
            pool.insert(after)
        block.address = start
        block.size = unmapped
        block.mapped = False
        self.try_merge_blocks(block, block.prev, pool)
        self.try_merge_blocks(block, block.next, pool)
        pool.insert_unmapped(block)
        self.total_allocated_memory -= unmapped
        self.stats.change("reserved_bytes", pool.is_small, -unmapped)
        self.stats.num_device_free += 1
        self.record_trace(
            "segment_unmap",
            start,
            unmapped,
            block.stream,
            context if context is not None else block.segment_context,
        )

    def release_blocks(self, pool: _BlockPool, context: list[dict[str, Any]] | None) -> None:
        to_unmap = []
        for block in pool.blocks():
            if block.expandable is not None:
                to_unmap.append(block)
            elif block.prev is None and block.next is None:
                self.release_block(block, context)
        for block in to_unmap:
            self.unmap_block(block, context)
            if block.prev is None and block.next is None:
                self.release_expandable_segment(block)

    def release_expandable_segment(self, block: _Block) -> None:
        segment = block.expandable
        if segment in self.expandable_segments:
            self.expandable_segments.remove(segment)
        block.pool.erase_unmapped(block)

    def empty_cache(self) -> None:
        context = self.gather_context(_RECORD_ALL)
        self.release_cached_blocks(context)

    # -- snapshot ------------------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        blocks = self.small.blocks() + self.large.blocks() + list(self.active.values())
        result: list[dict[str, Any]] = []
        total_active = 0
        for head in blocks:
            if head.prev is not None and head.prev.mapped:
                continue
            info: dict[str, Any] = {
                "address": head.address,
                "stream": head.stream,
                "is_large": not head.pool.is_small,
                "is_expandable": head.expandable is not None,
                "context": head.segment_context,
                "total_size": 0,
                "allocated_size": 0,
                "active_size": 0,
                "requested_size": 0,
                "blocks": [],
            }
            block: _Block | None = head
            while block is not None and block.mapped:
                active = (
                    block.allocated or block.event_count > 0 or bool(block.stream_uses)
                )
                info["blocks"].append(
                    {
                        "size": block.size,
                        "requested_size": block.requested_size,
                        "allocated": block.allocated,
                        "active": active,
                        "context": block.context,
                    }
                )
                info["total_size"] += block.size
                if block.allocated:
                    info["allocated_size"] += block.size
                if active:
                    info["active_size"] += block.size
                    info["requested_size"] += block.requested_size
                block = block.next
            total_active += info["active_size"]
            result.append(info)
        result.sort(key=lambda item: item["address"])
        self.record_trace("snapshot", 0, total_active, 0, None)
        return result


def _roundup_power2_next_division(size: int, divisions: int) -> int:
    if _is_power_of_2(size):
        return size
    power2_floor = 1 << _log2_floor(size)
    power2_division = power2_floor >> _log2_floor(divisions)
    if power2_division == 0:
        return power2_floor << 1
    round_size_floor = size & ~(power2_division - 1)
    return size if round_size_floor == size else round_size_floor + power2_division


# ---------------------------------------------------------------------------
# The tracker torch_patch uses
# ---------------------------------------------------------------------------

_stream_resolver: Callable[[int], int] | None = None


def set_stream_resolver(resolver: Callable[[int], int] | None) -> None:
    """Set the function that returns a device's current stream id."""
    global _stream_resolver
    _stream_resolver = resolver


def _current_stream(device: int) -> int:
    resolver = _stream_resolver
    if resolver is None:
        return 0
    try:
        return int(resolver(device))
    except Exception:
        return 0


class _DeviceMemoryTracker:
    """Track tensor bytes and a modeled CUDA caching allocator per device."""

    def __init__(
        self,
        per_device_bytes: list[int],
        *,
        caching_allocator: bool | None = None,
        torch_version: tuple[int, int] | None = None,
        alloc_conf: str | None = None,
        context_bytes: int | None = None,
        environ: Any = None,
    ):
        environ = os.environ if environ is None else environ
        self._version = tuple(torch_version) if torch_version else _detect_torch_version()
        self._total = list(per_device_bytes)
        self._context_bytes = (
            _configured_context_bytes() if context_bytes is None else max(0, int(context_bytes))
        )
        # Requested tensor bytes, which FakeGPU's own reports use.  The
        # torch.cuda memory API reports the allocator stats instead.
        self._used = [0] * len(per_device_bytes)
        self._peak = [0] * len(per_device_bytes)
        self._alloc_calls = [0] * len(per_device_bytes)
        self._free_calls = [0] * len(per_device_bytes)
        # Unlike torch's num_ooms, not cleared by its stat resets: `fakegpu
        # run` reports every simulated OOM, including ones a script caught.
        self._oom_events = [0] * len(per_device_bytes)
        self._last_oom: list[dict[str, int] | None] = [None] * len(per_device_bytes)
        self._peak_by_stage: list[dict[str, int]] = [dict() for _ in per_device_bytes]
        self._reserved_peak_by_stage: list[dict[str, int]] = [
            dict() for _ in per_device_bytes
        ]
        self._largest_allocations: list[list[dict[str, Any]]] = [
            [] for _ in per_device_bytes
        ]
        self._current_bytes_by_category: list[dict[str, int]] = [
            {} for _ in per_device_bytes
        ]
        # data_ptr -> allocation record (holds the allocator block)
        self._allocs: dict[int, dict[str, Any]] = {}
        # caching_allocator_alloc() pointers -> data_ptr keys
        self._raw_pointers: dict[int, int] = {}
        if caching_allocator is None:
            caching_allocator = environ.get("FAKEGPU_CACHING_ALLOCATOR", "1") != "0"
        self._caching_allocator = bool(caching_allocator)
        self._lock = threading.RLock()
        self._next_synthetic_data_ptr = -1
        self._synthetic_saved_raw_ptrs: dict[int, dict[str, int]] = {}
        self._held_saved_raw_ptrs: dict[int, int] = {}
        self._pending_saved_releases: set[int] = set()

        self._config = _AllocatorConfig()
        # A malformed allocator config fails CUDA initialization in torch;
        # FakeCUDA raises it at torch.cuda.init() and on every allocation.
        self._config_error: BaseException | None = None
        if alloc_conf is not None:
            cuda_env, accelerator_env = alloc_conf, alloc_conf
            if self._version < (2, 10):
                accelerator_env = None
        else:
            cuda_env, accelerator_env = _environment_allocator_settings(
                environ, self._version
            )
        try:
            _parse_allocator_settings(
                cuda_env, accelerator_env, self._config, self._version, runtime=False
            )
        except (RuntimeError, ValueError) as exc:
            self._config_error = exc
            self._config = _AllocatorConfig()
        self._devices = [
            _DeviceAllocator(self, index, total)
            for index, total in enumerate(per_device_bytes)
        ]

    # -- compatibility views ---------------------------------------------------

    @property
    def _stats(self) -> list[_DeviceStats]:
        return [device.stats for device in self._devices]

    @property
    def _reserved(self) -> list[int]:
        return [stats.total("reserved_bytes").current for stats in self._stats]

    @property
    def _reserved_peak(self) -> list[int]:
        return [stats.total("reserved_bytes").peak for stats in self._stats]

    @property
    def config(self) -> _AllocatorConfig:
        return self._config

    @property
    def torch_version(self) -> tuple[int, int]:
        return self._version

    @property
    def context_bytes(self) -> int:
        return self._context_bytes

    def _valid(self, device: int) -> bool:
        return 0 <= device < len(self._total)

    def check_initialized(self) -> None:
        """Raise the allocator config error torch raises at CUDA init."""
        error = self._config_error
        if error is not None:
            raise type(error)(*error.args)

    def _device_free(self, device: int) -> int:
        reserved = self._devices[device].stats.total("reserved_bytes").current
        return max(0, self._total[device] - reserved - self._context_bytes)

    # -- allocation ------------------------------------------------------------

    def allocate(
        self,
        data_ptr: int,
        nbytes: int,
        device: int,
        metadata: dict[str, Any] | None = None,
        *,
        stream: int | None = None,
    ) -> bool:
        """Register an allocation; raise OutOfMemoryError past the limit."""
        with self._lock:
            if not self._valid(device):
                return False
            if data_ptr in self._allocs:
                return False  # already tracked
            self.check_initialized()
            nbytes = max(0, int(nbytes))
            allocator = self._devices[device]
            block = None
            if nbytes > 0:
                if stream is None:
                    stream = _current_stream(device)
                try:
                    block = allocator.malloc(nbytes, int(stream))
                except _OutOfMemory as exc:
                    self._oom_events[device] += 1
                    self._last_oom[device] = {
                        "requested_bytes": nbytes,
                        "allocated_bytes": int(self._used[device]),
                        "reserved_bytes": int(
                            allocator.stats.total("reserved_bytes").current
                        ),
                    }
                    import torch

                    raise torch.cuda.OutOfMemoryError(exc.message) from None
            meta = dict(metadata or {})
            stage = str(
                meta.get("stage")
                or os.environ.get("FAKEGPU_PREFLIGHT_STAGE")
                or "unknown"
            )
            meta["stage"] = stage
            block_bytes = block.size if block is not None else 0
            self._allocs[data_ptr] = {
                "device": device,
                "bytes": nbytes,
                "block_bytes": block_bytes,
                **meta,
                "pool": (
                    "small" if block is not None and block.pool.is_small else "large"
                ),
                "_block": block,
            }
            category = str(meta.get("category") or "unknown")
            categories = self._current_bytes_by_category[device]
            categories[category] = categories.get(category, 0) + nbytes
            self._used[device] += nbytes
            self._peak[device] = max(self._peak[device], self._used[device])
            stage_peaks = self._peak_by_stage[device]
            stage_peaks[stage] = max(stage_peaks.get(stage, 0), self._used[device])
            reserved_stage_peaks = self._reserved_peak_by_stage[device]
            reserved_stage_peaks[stage] = max(
                reserved_stage_peaks.get(stage, 0),
                allocator.stats.total("reserved_bytes").current,
            )
            self._alloc_calls[device] += 1
            self._record_largest_allocation(data_ptr, device, nbytes, meta)
            return True

    def release(self, data_ptr: int) -> None:
        """Unregister an allocation."""
        with self._lock:
            if self._held_saved_raw_ptrs.get(data_ptr, 0) > 0:
                self._pending_saved_releases.add(data_ptr)
                return
            self._release_now(data_ptr)

    def _release_now(self, data_ptr: int) -> None:
        rec = self._allocs.pop(data_ptr, None)
        if not rec:
            return
        dev = rec["device"]
        nbytes = rec["bytes"]
        category = str(rec.get("category") or "unknown")
        categories = self._current_bytes_by_category[dev]
        remaining = max(0, categories.get(category, 0) - nbytes)
        if remaining:
            categories[category] = remaining
        else:
            categories.pop(category, None)
        self._used[dev] = max(0, self._used[dev] - nbytes)
        self._free_calls[dev] += 1
        block = rec.get("_block")
        if block is not None:
            self._devices[dev].free(block)

    def record_stream(self, data_ptr: int, stream: int) -> None:
        """``Tensor.record_stream``: delay reuse until ``stream``'s work ends."""
        with self._lock:
            rec = self._allocs.get(int(data_ptr))
            block = rec.get("_block") if rec else None
            if block is not None:
                self._devices[rec["device"]].record_stream(block, int(stream))

    def raw_alloc(self, nbytes: int, stream: int, device: int) -> int:
        """``caching_allocator_alloc``: return the block's device address."""
        with self._lock:
            nbytes = int(nbytes)
            if nbytes == 0:
                return 0
            if not self._valid(device):
                raise RuntimeError(
                    f"Allocator not initialized for device {device}: did you call init?"
                )
            key = self._next_synthetic_data_ptr
            self._next_synthetic_data_ptr -= 1
            self.allocate(
                key,
                nbytes,
                device,
                metadata={"category": "raw", "source": "caching_allocator_alloc"},
                stream=int(stream),
            )
            address = self._allocs[key]["_block"].address
            self._raw_pointers[address] = key
            return address

    def raw_delete(self, pointer: int) -> None:
        with self._lock:
            pointer = int(pointer)
            if pointer == 0:
                return
            key = self._raw_pointers.pop(pointer, None)
            if key is None:
                raise RuntimeError(f"invalid device pointer: {hex(pointer)}")
            self.release(key)

    def allocate_saved_tensor(
        self,
        *,
        raw_data_ptr: int,
        nbytes: int,
        device: int,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        """Track an autograd-saved raw tensor that is hidden behind a high-level op."""
        with self._lock:
            if raw_data_ptr in self._allocs:
                self._held_saved_raw_ptrs[raw_data_ptr] = (
                    self._held_saved_raw_ptrs.get(raw_data_ptr, 0) + 1
                )
                return raw_data_ptr
            existing = self._synthetic_saved_raw_ptrs.get(raw_data_ptr)
            if existing is not None:
                existing["refs"] += 1
                return int(existing["synthetic_data_ptr"])

            synthetic_data_ptr = self._next_synthetic_data_ptr
            self._next_synthetic_data_ptr -= 1
            meta = {
                "category": "activation",
                "synthetic": True,
                "source": "autograd_saved_tensor",
                **(metadata or {}),
            }
            self.allocate(synthetic_data_ptr, nbytes, device, metadata=meta)
            self._synthetic_saved_raw_ptrs[raw_data_ptr] = {
                "synthetic_data_ptr": synthetic_data_ptr,
                "refs": 1,
            }
            return synthetic_data_ptr

    def release_saved_tensor(self, raw_data_ptr: int) -> None:
        with self._lock:
            held_refs = self._held_saved_raw_ptrs.get(raw_data_ptr)
            if held_refs is not None:
                held_refs -= 1
                if held_refs > 0:
                    self._held_saved_raw_ptrs[raw_data_ptr] = held_refs
                    return
                self._held_saved_raw_ptrs.pop(raw_data_ptr, None)
                if raw_data_ptr in self._pending_saved_releases:
                    self._pending_saved_releases.remove(raw_data_ptr)
                    self._release_now(raw_data_ptr)
                return

            existing = self._synthetic_saved_raw_ptrs.get(raw_data_ptr)
            if existing is None:
                return
            refs = int(existing.get("refs", 1)) - 1
            if refs > 0:
                existing["refs"] = refs
                return
            synthetic_data_ptr = int(existing["synthetic_data_ptr"])
            self._synthetic_saved_raw_ptrs.pop(raw_data_ptr, None)
            self._release_now(synthetic_data_ptr)

    def has_allocation(self, data_ptr: int) -> bool:
        with self._lock:
            return int(data_ptr) in self._allocs

    # -- statistics ------------------------------------------------------------

    def _device_stat(self, device: int, name: str) -> _Stat | None:
        if not self._valid(device):
            return None
        return self._devices[device].stats.total(name)

    def memory_allocated(self, device: int) -> int:
        # torch.cuda.memory_allocated() is allocated_bytes.all.current: the
        # allocator's block sizes, not the requested tensor bytes.
        with self._lock:
            stat = self._device_stat(device, "allocated_bytes")
            return 0 if stat is None else stat.current

    def max_memory_allocated(self, device: int) -> int:
        with self._lock:
            stat = self._device_stat(device, "allocated_bytes")
            return 0 if stat is None else stat.peak

    def memory_reserved(self, device: int) -> int:
        with self._lock:
            stat = self._device_stat(device, "reserved_bytes")
            return 0 if stat is None else stat.current

    def max_memory_reserved(self, device: int) -> int:
        with self._lock:
            stat = self._device_stat(device, "reserved_bytes")
            return 0 if stat is None else stat.peak

    def mem_get_info(self, device: int) -> tuple[int, int]:
        """``cudaMemGetInfo``: free memory excludes the modeled CUDA context."""
        with self._lock:
            if not self._valid(device):
                return (0, 0)
            return (self._device_free(device), self._total[device])

    def _check_fraction_device(self, device: int) -> None:
        if not self._valid(device):
            raise RuntimeError(
                f"Allocator not initialized for device {device}: did you call init?"
            )

    def set_memory_fraction(self, fraction: float, device: int) -> None:
        """Mirror ``DeviceCachingAllocator::setMemoryFraction``."""
        with self._lock:
            self._check_fraction_device(device)
            self._devices[device].set_memory_fraction(float(fraction))

    def get_memory_fraction(self, device: int) -> float:
        with self._lock:
            self._check_fraction_device(device)
            return self._devices[device].get_memory_fraction()

    def reset_peak(self, device: int) -> None:
        with self._lock:
            if self._valid(device):
                self._peak[device] = self._used[device]
                self._devices[device].stats.reset_peak()
                self._peak_by_stage[device] = {}
                self._reserved_peak_by_stage[device] = {}

    def reset_accumulated(self, device: int) -> None:
        with self._lock:
            if self._valid(device):
                self._devices[device].stats.reset_accumulated()

    def empty_cache(self, device: int | None = None) -> None:
        with self._lock:
            devices = range(len(self._total)) if device is None else (int(device),)
            for index in devices:
                if self._valid(index):
                    self._devices[index].empty_cache()

    def memory_stats_as_nested_dict(self, device: int) -> dict[str, Any]:
        with self._lock:
            if not self._valid(device):
                return _DeviceStats(self._version).as_nested_dict()
            return self._devices[device].stats.as_nested_dict()

    def memory_stats(self, device: int) -> OrderedDict[str, Any]:
        return _flatten_memory_stats(self.memory_stats_as_nested_dict(device))

    def host_memory_stats(self) -> dict[str, Any]:
        return _host_memory_stats(self._version)

    # -- settings ----------------------------------------------------------------

    def allocator_backend(self) -> str:
        return self._config.backend

    def allocator_settings(self) -> str:
        return self._config.last_allocator_settings

    def set_allocator_settings(self, env: str) -> None:
        """``torch.cuda.memory._set_allocator_settings``."""
        with self._lock:
            _parse_allocator_settings(
                str(env), str(env), self._config, self._version, runtime=True
            )

    # -- history and snapshots ---------------------------------------------------

    def record_memory_history(
        self,
        enabled: str | None,
        context: str | None = "all",
        stacks: str = "all",
        max_entries: int = sys.maxsize,
        clear_history: bool = False,
        compile_context: bool = False,
        global_record_annotations: bool = False,
        skip_actions: Any = (),
        record_pinned_host_memory: bool = False,
        record_cuda: bool = True,
    ) -> None:
        """``torch::cuda::_record_memory_history`` (string form)."""
        if record_pinned_host_memory and enabled is None:
            raise RuntimeError(
                "record_pinned_host_memory requires memory history recording to "
                "be enabled"
            )
        if enabled is not None and enabled not in ("state", "all"):
            raise RuntimeError("expected state to be 'state', 'all', or None")
        if context is not None and context not in ("state", "alloc", "all"):
            raise RuntimeError(
                "expected context to be 'state', 'alloc', 'all', or None"
            )
        if stacks not in ("python", "all"):
            raise RuntimeError("expected stacks to be 'python', or 'all'")
        entries = int(max_entries) if enabled == "all" else 1
        when = {
            None: _RECORD_NEVER,
            "state": _RECORD_STATE,
            "alloc": _RECORD_ALLOC,
            "all": _RECORD_ALL,
        }[context]
        self._record_history(
            enabled is not None and bool(record_cuda),
            entries,
            when,
            bool(clear_history),
            skip_actions,
        )

    def record_memory_history_legacy(
        self,
        enabled: bool,
        record_context: bool = True,
        trace_alloc_max_entries: int = 1,
        trace_alloc_record_context: bool = False,
        record_context_cpp: bool = False,
        clear_history: bool = False,
        compile_context: bool = False,
        global_record_annotations: bool = False,
        skip_actions: Any = (),
        record_pinned_host_memory: bool = False,
        record_cuda: bool = True,
    ) -> None:
        """``torch::cuda::_record_memory_history`` (boolean form)."""
        if record_pinned_host_memory and not enabled:
            raise RuntimeError(
                "record_pinned_host_memory requires memory history recording to "
                "be enabled"
            )
        when = _RECORD_NEVER
        if trace_alloc_record_context:
            when = _RECORD_ALLOC
        elif record_context:
            when = _RECORD_STATE
        self._record_history(
            bool(enabled) and bool(record_cuda),
            int(trace_alloc_max_entries),
            when,
            bool(clear_history),
            skip_actions,
        )

    def _record_history(
        self,
        enabled: bool,
        max_entries: int,
        when: int,
        clear_history: bool,
        skip_actions: Any,
    ) -> None:
        skip = set()
        for action in skip_actions or ():
            if action not in _TRACE_ACTIONS:
                raise RuntimeError(f"Unknown TraceEntry action: {action}")
            skip.add(action)
        with self._lock:
            for allocator in self._devices:
                allocator.record_history = enabled
                allocator.skip_actions = skip
                allocator.alloc_buffer.set_max_entries(max_entries)
                allocator.record_context = when if enabled else _RECORD_NEVER
                if not enabled or clear_history:
                    allocator.alloc_buffer.clear()

    def is_history_enabled(self) -> bool:
        with self._lock:
            return any(allocator.record_history for allocator in self._devices)

    def snapshot_dict(self, arg: Any = None) -> dict[str, Any]:
        """``torch._C._cuda_memorySnapshot(arg)`` for the installed torch."""
        mempool_id, include_traces = self._snapshot_arguments(arg)
        version = self._version
        with self._lock:
            segments: list[dict[str, Any]] = []
            traces: list[list[dict[str, Any]]] = []
            for allocator in self._devices:
                if include_traces:
                    traces.append(
                        [self._trace_entry(entry) for entry in allocator.alloc_buffer.get()]
                    )
                infos = allocator.snapshot()
                if mempool_id == (0, 0):
                    segments.extend(
                        self._segment_dict(allocator.device, info) for info in infos
                    )
            config = self._config
            settings: dict[str, Any] = {
                "PYTORCH_CUDA_ALLOC_CONF": config.last_allocator_settings,
                "max_split_size": _int64(config.max_split_size),
                "garbage_collection_threshold": config.garbage_collection_threshold,
                "expandable_segments": config.expandable_segments,
                "pinned_num_register_threads": config.pinned_num_register_threads,
                "release_lock_on_cudamalloc": config.release_lock_on_cudamalloc,
                "pinned_use_cuda_host_register": config.pinned_use_cuda_host_register,
                "graph_capture_record_stream_reuse": (
                    config.graph_capture_record_stream_reuse
                ),
            }
            if version >= (2, 13):
                settings["max_round_threshold"] = _int64(config.pinned_max_round_threshold)
                settings["max_cached_size"] = _int64(config.pinned_max_cached_size)
            settings["roundup_power2_divisions"] = {
                str(1 << index): value
                for index, value in enumerate(config.roundup_power2_divisions)
            }
            result: dict[str, Any] = {
                "segments": segments,
                "device_traces": traces,
                "allocator_settings": settings,
                "external_annotations": [],
            }
            if version >= (2, 14):
                result["host_segments"] = []
                result["host_traces"] = []
            return result

    def _snapshot_arguments(self, arg: Any) -> tuple[tuple[int, int], bool]:
        if arg is None:
            return (0, 0), True
        if self._version < (2, 11):
            if not isinstance(arg, tuple):
                raise RuntimeError("mempool_id must be a tuple")
            if len(arg) != 2:
                raise RuntimeError("mempool_id must be a tuple of 2 integers")
        else:
            if not isinstance(arg, tuple):
                raise RuntimeError("Expected tuple or None")
            if len(arg) not in (2, 3):
                raise RuntimeError("Expected tuple of size 2 or 3")
        first, second = arg[0], arg[1]
        if not (
            isinstance(first, int)
            and isinstance(second, int)
            and not isinstance(first, bool)
            and not isinstance(second, bool)
        ):
            raise RuntimeError("mempool_id elements must be integers")
        include_traces = True
        if len(arg) == 3:
            if not isinstance(arg[2], bool):
                raise RuntimeError("include_traces must be a boolean")
            include_traces = arg[2]
        return (int(first), int(second)), include_traces

    def _segment_dict(self, device: int, info: dict[str, Any]) -> dict[str, Any]:
        segment: dict[str, Any] = {
            "device": device,
            "address": info["address"],
            "total_size": info["total_size"],
            "allocated_size": info["allocated_size"],
            "active_size": info["active_size"],
            "requested_size": info["requested_size"],
            "stream": info["stream"],
            "segment_type": "large" if info["is_large"] else "small",
            "segment_pool_id": (0, 0),
            "is_expandable": info["is_expandable"],
        }
        # Frames are added after the other keys when there is a context,
        # as torch fills them in after symbolizing.
        if info["context"] is None:
            segment["frames"] = []
        address = info["address"]
        blocks = []
        for block in info["blocks"]:
            entry: dict[str, Any] = {
                "address": address,
                "size": block["size"],
                "requested_size": block["requested_size"],
                "state": (
                    "active_allocated"
                    if block["allocated"]
                    else ("active_pending_free" if block["active"] else "inactive")
                ),
                "frames": list(block["context"] or []),
            }
            blocks.append(entry)
            address += block["size"]
        segment["blocks"] = blocks
        if info["context"] is not None:
            segment["frames"] = list(info["context"])
        return segment

    def _trace_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {"action": entry["action"]}
        result["device_free" if entry["action"] == "oom" else "addr"] = entry["addr"]
        result["size"] = entry["size"]
        result["stream"] = entry["stream"]
        result["time_us"] = entry["time_us"]
        result["compile_context"] = "N/A"
        if self._version >= (2, 10):
            result["user_metadata"] = ""
        if self._version >= (2, 12):
            result["pool_id"] = (0, 0)
        if entry["context"] is not None:
            result["frames"] = list(entry["context"])
        return result

    def allocator_snapshot(self, mempool_id: Any = None) -> list[dict[str, Any]]:
        """Return segments in the layout of ``torch.cuda.memory_snapshot()``."""
        arg = None if mempool_id is None else tuple(mempool_id)
        if arg is not None and len(arg) == 2 and self._version >= (2, 11):
            arg = (arg[0], arg[1], False)
        return self.snapshot_dict(arg)["segments"]

    # -- FakeGPU reports -----------------------------------------------------------

    def peak_by_stage(self, device: int) -> dict[str, int]:
        with self._lock:
            if not self._valid(device):
                return {}
            return dict(self._peak_by_stage[device])

    def reserved_peak_by_stage(self, device: int) -> dict[str, int]:
        with self._lock:
            if not self._valid(device):
                return {}
            return dict(self._reserved_peak_by_stage[device])

    def largest_allocations(self, device: int, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            if not self._valid(device):
                return []
            return [
                _public_allocation_record(item)
                for item in self._largest_allocations[device][:limit]
            ]

    def current_bytes_by_category(self, device: int) -> dict[str, int]:
        with self._lock:
            if not self._valid(device):
                return {}
            return dict(self._current_bytes_by_category[device])

    def mark_category(self, data_ptr: int, category: str) -> None:
        with self._lock:
            record = self._allocs.get(data_ptr)
            if record is not None:
                device = record["device"]
                normalized_category = str(category or "unknown")
                previous_category = str(
                    record.get("category") or "unknown"
                )
                if (
                    0 <= device < len(self._current_bytes_by_category)
                    and normalized_category != previous_category
                ):
                    categories = self._current_bytes_by_category[device]
                    nbytes = record["bytes"]
                    remaining = max(
                        0,
                        categories.get(previous_category, 0) - nbytes,
                    )
                    if remaining:
                        categories[previous_category] = remaining
                    else:
                        categories.pop(previous_category, None)
                    categories[normalized_category] = (
                        categories.get(normalized_category, 0) + nbytes
                    )
                record["category"] = category
                if 0 <= device < len(self._largest_allocations):
                    for item in self._largest_allocations[device]:
                        if item["_data_ptr"] == data_ptr:
                            item["category"] = category

    def _segment_count(self, device: int) -> int:
        allocator = self._devices[device]
        blocks = allocator.small.blocks() + allocator.large.blocks() + list(
            allocator.active.values()
        )
        return sum(
            1
            for block in blocks
            if block.prev is None or not block.prev.mapped
        )

    def snapshot(self, profiles: list[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            allocator_model = (
                "cuda_caching_allocator.v1"
                if self._caching_allocator
                else "direct_segments.v1"
            )
            tracking_confidence = (
                "C3_torch_dispatch_lifetime"
                if _dispatch_tracking_stats.get("enabled")
                else "C2_torch_tensor_lifetime"
            )
            devices: list[dict[str, Any]] = []
            for index, prof in enumerate(profiles):
                if index >= len(self._total):
                    break
                total = int(self._total[index])
                peak = int(self._peak[index])
                current = int(self._used[index])
                reserved = int(self._reserved[index])
                reserved_peak = int(self._reserved_peak[index])
                conservative_peak = max(peak, reserved_peak)
                headroom = total - conservative_peak
                headroom_percent = (100.0 * headroom / total) if total > 0 else None
                devices.append(
                    {
                        "index": index,
                        "name": str(prof.get("name", "")),
                        "profile_id": str(prof.get("profile_id", "")),
                        "total_memory": total,
                        "current_memory": current,
                        "peak_memory": peak,
                        "current_reserved_memory": reserved,
                        "peak_reserved_memory": reserved_peak,
                        "inactive_split_bytes": int(
                            self._devices[index]
                            .stats.total("inactive_split_bytes")
                            .current
                        ),
                        "segment_count": self._segment_count(index),
                        "headroom_bytes": headroom,
                        "headroom_percent": (
                            round(headroom_percent, 3)
                            if headroom_percent is not None
                            else None
                        ),
                        "allocation_count": int(self._alloc_calls[index]),
                        "free_count": int(self._free_calls[index]),
                        "oom_events": int(self._oom_events[index]),
                        "last_oom": dict(self._last_oom[index]) if self._last_oom[index] else None,
                        "current_bytes_by_category": self.current_bytes_by_category(
                            index
                        ),
                        "peak_by_stage": self.peak_by_stage(index),
                        "reserved_peak_by_stage": self.reserved_peak_by_stage(index),
                        "largest_allocations": self.largest_allocations(index),
                        "allocator_model": allocator_model,
                        "tracking_confidence": tracking_confidence,
                    }
                )
            return {
                "tracking_confidence": tracking_confidence,
                "allocator_model": allocator_model,
                "dispatch_tracking": _dispatch_tracking_snapshot(),
                "devices": devices,
            }

    def _record_largest_allocation(
        self, data_ptr: int, device: int, nbytes: int, metadata: dict[str, Any]
    ) -> None:
        entries = self._largest_allocations[device]
        if len(entries) >= 10 and nbytes <= entries[-1]["bytes"]:
            return
        item = {
            "_data_ptr": int(data_ptr),
            "bytes": int(nbytes),
            "device": int(device),
            "dtype": metadata.get("dtype"),
            "shape": metadata.get("shape"),
            "stage": metadata.get("stage"),
            "category": metadata.get("category", "tensor"),
        }
        if metadata.get("source"):
            item["source"] = str(metadata["source"])
        if metadata.get("operator"):
            item["operator"] = str(metadata["operator"])
        if metadata.get("stack"):
            item["stack"] = metadata["stack"]
        position = next(
            (
                index
                for index, allocation in enumerate(entries)
                if nbytes > allocation["bytes"]
            ),
            len(entries),
        )
        entries.insert(position, item)
        del entries[10:]
