#!/usr/bin/env python3
"""Finite CPU_SIM LoRA/attention demand probe with independent scalar references.

No CUDA installation or physical GPU is used. Reference reductions use Python
float64 and independently encode FP32/FP16/BF16 at every stored intermediate.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys


P, Int, L = ct.c_void_p, ct.c_int, ct.c_longlong


def encode(value, dtype):
    if dtype == 0:
        return struct.pack("<f", value)
    if dtype == 2:
        return struct.pack("<e", value)
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)


def decode(raw, dtype):
    if dtype == 0:
        return struct.unpack("<f", raw)[0]
    if dtype == 2:
        return struct.unpack("<e", raw)[0]
    return struct.unpack("<f", b"\0\0" + raw)[0]


def stored(value, dtype):
    return decode(encode(value, dtype), dtype)


def reference(a, b, m, n, k, dtype, *, alpha=1.0, previous=None, transpose_b=False):
    alpha = stored(alpha, 0)
    result = []
    for col in range(n):
        for row in range(m):
            total = sum(
                a[row + p * m] * b[(col + p * n) if transpose_b else (p + col * k)]
                for p in range(k)
            )
            result.append(
                stored(
                    stored(
                        alpha * total
                        + (previous[row + col * m] if previous is not None else 0),
                        0,
                    ),
                    dtype,
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.build_dir.resolve()
    suffix = "dylib" if sys.platform == "darwin" else "so.12"
    cuda = ct.CDLL(str(root / f"libcudart.{suffix}"))
    blas = ct.CDLL(str(root / f"libcublas.{suffix}"))

    def bind(lib, name, types):
        function = getattr(lib, name)
        function.argtypes, function.restype = types, Int
        return function

    malloc = bind(cuda, "cudaMalloc", [P, ct.c_size_t])
    copy = bind(cuda, "cudaMemcpy", [P, P, ct.c_size_t, Int])
    free = bind(cuda, "cudaFree", [P])
    create = bind(blas, "cublasCreate_v2", [P])
    destroy = bind(blas, "cublasDestroy_v2", [P])
    gemm = bind(
        blas,
        "cublasGemmEx",
        [P] + [Int] * 5 + [P, P, Int, Int, P, Int, Int, P, P, Int, Int, Int, Int],
    )
    batched = bind(
        blas,
        "cublasGemmStridedBatchedEx",
        [P]
        + [Int] * 5
        + [P, P, Int, Int, L, P, Int, Int, L, P, P, Int, Int, L, Int, Int, Int],
    )
    handle = P()
    assert create(ct.byref(handle)) == 0
    allocations = []

    def upload(values, dtype):
        raw = b"".join(encode(v, dtype) for v in values)
        host = ct.create_string_buffer(raw)
        pointer = P()
        assert malloc(ct.byref(pointer), len(raw)) == 0
        allocations.append(pointer)
        assert copy(pointer, host, len(raw), 1) == 0
        return pointer

    def download(pointer, count, dtype):
        size = 4 if dtype == 0 else 2
        host = ct.create_string_buffer(size * count)
        assert copy(host, pointer, len(host), 2) == 0
        return [
            decode(host.raw[i : i + size], dtype) for i in range(0, len(host), size)
        ]

    rng = random.Random(73021)
    cases = []
    try:
        for label, dtype, atol in (
            ("fp32", 0, 2e-6),
            ("fp16", 2, 2e-4),
            ("bf16", 14, 2e-3),
        ):

            def values(count):
                return [stored(rng.uniform(-0.5, 0.5), dtype) for _ in range(count)]

            m, k, n, rank = 7, 11, 5, 3
            x, w, a, b = (
                values(m * k),
                values(k * n),
                values(k * rank),
                values(rank * n),
            )
            dx, dw, da, db = (upload(v, dtype) for v in (x, w, a, b))
            dy, dz = (
                upload([float("nan")] * (m * n), dtype),
                upload([float("nan")] * (m * rank), dtype),
            )
            one, zero, scale = ct.c_float(1), ct.c_float(0), ct.c_float(0.3)
            assert (
                gemm(
                    handle,
                    0,
                    0,
                    m,
                    n,
                    k,
                    ct.byref(one),
                    dx,
                    dtype,
                    m,
                    dw,
                    dtype,
                    k,
                    ct.byref(zero),
                    dy,
                    dtype,
                    m,
                    68,
                    -1,
                )
                == 0
            )
            assert (
                gemm(
                    handle,
                    0,
                    0,
                    m,
                    rank,
                    k,
                    ct.byref(one),
                    dx,
                    dtype,
                    m,
                    da,
                    dtype,
                    k,
                    ct.byref(zero),
                    dz,
                    dtype,
                    m,
                    68,
                    -1,
                )
                == 0
            )
            assert (
                gemm(
                    handle,
                    0,
                    0,
                    m,
                    n,
                    rank,
                    ct.byref(scale),
                    dz,
                    dtype,
                    m,
                    db,
                    dtype,
                    rank,
                    ct.byref(one),
                    dy,
                    dtype,
                    m,
                    68,
                    -1,
                )
                == 0
            )
            z = reference(x, a, m, rank, k, dtype)
            expected = reference(
                z,
                b,
                m,
                n,
                rank,
                dtype,
                alpha=scale.value,
                previous=reference(x, w, m, n, k, dtype),
            )
            got = download(dy, m * n, dtype)
            error = max(abs(g - e) for g, e in zip(got, expected))
            assert all(math.isfinite(v) for v in got) and error <= atol, (
                label,
                "lora",
                error,
            )
            cases.append(
                {
                    "dtype": label,
                    "workload": "XW + 0.3*(XA)B",
                    "shape": [m, k, n, rank],
                    "max_absolute_error": error,
                    "absolute_tolerance": atol,
                    "output_sha256": hashlib.sha256(
                        b"".join(encode(v, dtype) for v in got)
                    ).hexdigest(),
                }
            )
            batches, tokens, head = 4, 5, 7
            q, key = values(batches * tokens * head), values(batches * tokens * head)
            dq, dk = upload(q, dtype), upload(key, dtype)
            scores = upload([float("nan")] * (batches * tokens * tokens), dtype)
            attention_scale = ct.c_float(1 / math.sqrt(head))
            assert (
                batched(
                    handle,
                    0,
                    1,
                    tokens,
                    tokens,
                    head,
                    ct.byref(attention_scale),
                    dq,
                    dtype,
                    tokens,
                    tokens * head,
                    dk,
                    dtype,
                    tokens,
                    tokens * head,
                    ct.byref(zero),
                    scores,
                    dtype,
                    tokens,
                    tokens * tokens,
                    batches,
                    68,
                    -1,
                )
                == 0
            )
            expected = []
            for batch in range(batches):
                start = batch * tokens * head
                expected.extend(
                    reference(
                        q[start : start + tokens * head],
                        key[start : start + tokens * head],
                        tokens,
                        tokens,
                        head,
                        dtype,
                        alpha=attention_scale.value,
                        transpose_b=True,
                    )
                )
            got = download(scores, batches * tokens * tokens, dtype)
            error = max(abs(g - e) for g, e in zip(got, expected))
            assert all(math.isfinite(v) for v in got) and error <= atol, (
                label,
                "attention",
                error,
            )
            cases.append(
                {
                    "dtype": label,
                    "workload": "four-head scaled QK^T",
                    "shape": [batches, tokens, head],
                    "max_absolute_error": error,
                    "absolute_tolerance": atol,
                }
            )
        # Strict unsupported integer GEMM must reject and preserve the output.
        integer_output = (ct.c_int8 * 1)(91)
        integer_input = (ct.c_int8 * 1)(2)
        status = gemm(
            handle,
            0,
            0,
            1,
            1,
            1,
            ct.byref(one),
            integer_input,
            3,
            1,
            integer_input,
            3,
            1,
            ct.byref(zero),
            integer_output,
            3,
            1,
            68,
            -1,
        )
        assert status == 15 and integer_output[0] == 91
        report = {
            "schema_version": "fakegpu.research_numerics.v1",
            "physical_gpu_used": False,
            "cases": cases,
            "strict_integer_rejection": {"status": status, "output_unchanged": True},
            "required_apis": [
                "cudaMalloc",
                "cudaMemcpy",
                "cudaFree",
                "cublasCreate_v2",
                "cublasDestroy_v2",
                "cublasGemmEx",
                "cublasGemmStridedBatchedEx",
            ],
            "scope": "Finite direct host-API math, float64 scalar reference with intermediate dtype rounding; excludes arbitrary CUDA kernels, softmax, backward, optimizer and whole-model GPU execution.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "cases": len(cases),
                    "max_absolute_error": max(c["max_absolute_error"] for c in cases),
                    "strict_integer_rejection": status,
                }
            )
        )
    finally:
        for pointer in allocations:
            assert free(pointer) == 0
        assert destroy(handle) == 0


if __name__ == "__main__":
    main()
