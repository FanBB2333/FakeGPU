#!/usr/bin/env python3
"""Exercise cuBLAS fallback policies against compiled libraries, without CUDA."""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


P, Int, L = ct.c_void_p, ct.c_int, ct.c_longlong


def probe(library: Path, *, cpu_simulation: bool) -> dict[str, int]:
    lib = ct.CDLL(str(library))
    strict = os.environ["FAKEGPU_UNSUPPORTED_API"] == "error"
    skip_gemm = not cpu_simulation or os.environ["FAKEGPU_CPU_GEMM_MODE"] == "skip"
    unsupported: Counter[str] = Counter()
    lt_matmuls = 0
    typed_gemms = 0

    def bind(name, types):
        func = getattr(lib, name)
        func.argtypes = types
        func.restype = Int
        return func

    handle = P()
    assert bind("cublasCreate_v2", [P])(ct.byref(handle)) == 0

    def check(name, types, args, outputs, *, fallback=True, value=None, call_handle=None, reject=False,
              unchanged=False, synthetic_value=False, typed=False):
        func = bind(name, [P, *types])
        nonlocal lt_matmuls, typed_gemms
        snapshots = [bytes(output) for output in outputs]
        expected = 15 if reject or (strict and fallback) else 0  # CUBLAS_STATUS_NOT_SUPPORTED
        for _ in range(2):
            for output, snapshot in zip(outputs, snapshots):
                ct.memmove(output, snapshot, len(snapshot))
            status = func(handle if call_handle is None else call_handle, *args)
            assert status == expected, (name, status, expected)
            if name == "cublasLtMatmul" and status == 0:
                lt_matmuls += 1
            if typed and status == 0:
                typed_gemms += 1
            if unchanged or reject or (strict and fallback):
                assert [bytes(output) for output in outputs] == snapshots, name
            if value is not None and expected == 0 and (not fallback or synthetic_value):
                assert outputs[0][0] == value, (name, outputs[0][0], value)
        for output, snapshot in zip(outputs, snapshots):
            ct.memmove(output, snapshot, len(snapshot))
        if fallback:
            unsupported[name] += 2

    # The same small matrix checks each precision entry point and its metadata.
    for prefix, scalar in (("S", ct.c_float), ("D", ct.c_double),
                           ("C", ct.c_float), ("Z", ct.c_double)):
        a, b, c = (scalar * 2)(2, 0), (scalar * 2)(3, 0), (scalar * 2)(123, 0)
        alpha, beta = (scalar * 2)(1, 0), (scalar * 2)(0, 0)
        aa, bb, cc = ((P * 1)(ct.addressof(buf)) for buf in (a, b, c))
        pivot, info, dev_info = (Int * 1)(123), (Int * 1)(123), (Int * 1)(123)
        base = "cublas" + prefix
        check(base + "getrfBatched", [Int, P, Int, P, P, Int],
              [1, aa, 1, pivot, info, 1], [a, pivot, info])
        check(base + "getrsBatched", [Int, Int, Int, P, Int, P, P, Int, P, Int],
              [0, 1, 1, aa, 1, pivot, bb, 1, info, 1], [b, info])
        check(base + "geqrfBatched", [Int, Int, P, Int, P, P, Int],
              [1, 1, aa, 1, bb, info, 1], [a, b, info])
        check(base + "gelsBatched", [Int, Int, Int, Int, P, Int, P, Int, P, P, Int],
              [0, 1, 1, 1, aa, 1, cc, 1, info, dev_info, 1], [a, c, info, dev_info])
        check(base + "trsm_v2", [Int] * 6 + [P, P, Int, P, Int],
              [0, 0, 0, 0, 1, 1, alpha, a, 1, b, 1], [b])
        check(base + "trsmBatched", [Int] * 6 + [P, P, Int, P, Int, Int],
              [0, 0, 0, 0, 1, 1, alpha, aa, 1, bb, 1, 1], [b])
        if prefix in ("S", "D"):
            check(base + "symm_v2", [Int] * 4 + [P, P, Int, P, Int, P, P, Int],
                  [0, 0, 1, 1, alpha, a, 1, b, 1, beta, c, 1], [c])
            check(base + "syrk_v2", [Int] * 4 + [P, P, Int, P, P, Int],
                  [0, 0, 1, 1, alpha, a, 1, beta, c, 1], [c])
            for suffix in ("syr2k_v2", "syrkx"):
                check(base + suffix, [Int] * 4 + [P, P, Int, P, Int, P, P, Int],
                      [0, 0, 1, 1, alpha, a, 1, b, 1, beta, c, 1], [c])
            check(base + "trmm_v2", [Int] * 6 + [P, P, Int, P, Int, P, Int],
                  [0, 0, 0, 0, 1, 1, alpha, a, 1, b, 1, c, 1], [c])
        else:
            for suffix in ("dotu_v2", "dotc_v2"):
                check(base + suffix, [Int, P, Int, P, Int, P], [1, a, 1, b, 1, c], [c])
            check(base + "gemv_v2", [Int] * 3 + [P, P, Int, P, Int, P, P, Int],
                  [0, 1, 1, alpha, a, 1, b, 1, beta, c, 1], [c])
            check(base + "gemm_v2", [Int] * 5 + [P, P, Int, P, Int, P, P, Int],
                  [0, 0, 1, 1, 1, alpha, a, 1, b, 1, beta, c, 1], [c])
            check(base + "gemmStridedBatched", [Int] * 5 + [P, P, Int, L, P, Int, L, P, P, Int, L, Int],
                  [0, 0, 1, 1, 1, alpha, a, 1, 1, b, 1, 1, beta, c, 1, 1, 1], [c])

    result = (ct.c_float * 1)(123)
    check("cublasDotEx", [Int, P, Int, Int, P, Int, Int, P, Int, Int],
          [1, a, 0, 1, b, 0, 1, result, 0, 0], [result])

    for prefix, scalar in (("S", ct.c_float), ("D", ct.c_double)):
        a, b, c = (scalar * 1)(2), (scalar * 1)(3), (scalar * 1)(123)
        alpha, beta = (scalar * 1)(1), (scalar * 1)(0)
        base = "cublas" + prefix
        for suffix in ("asum_v2", "nrm2_v2"):
            check(base + suffix, [Int, P, Int, P], [1, a, 1, c], [c],
                  fallback=not cpu_simulation, value=2)
        for suffix in ("amax_v2", "amin_v2"):
            index = (Int * 1)(123)
            check("cublasI" + prefix.lower() + suffix, [Int, P, Int, P],
                  [1, a, 1, index], [index], fallback=not cpu_simulation, value=1,
                  synthetic_value=True)  # cuBLAS indices are 1-based, so n=1 always gives 1
        check(base + "axpy_v2", [Int, P, P, Int, P, Int], [1, alpha, a, 1, c, 1], [c],
              fallback=not cpu_simulation, value=125)
        check(base + "copy_v2", [Int, P, Int, P, Int], [1, a, 1, c, 1], [c],
              fallback=not cpu_simulation, value=2)
        check(base + "dot_v2", [Int, P, Int, P, Int, P], [1, a, 1, b, 1, c], [c],
              fallback=not cpu_simulation, value=6)
        check(base + "scal_v2", [Int, P, P, Int], [1, alpha, a, 1], [a],
              fallback=not cpu_simulation, value=2)
        check(base + "gemv_v2", [Int] * 3 + [P, P, Int, P, Int, P, P, Int],
              [0, 1, 1, alpha, a, 1, b, 1, beta, c, 1], [c],
              fallback=not cpu_simulation, value=6)
        check(base + "ger_v2", [Int, Int, P, P, Int, P, Int, P, Int],
              [1, 1, alpha, a, 1, b, 1, c, 1], [c], fallback=not cpu_simulation, value=129)

    for prefix, scalar in (("S", ct.c_float), ("D", ct.c_double), ("H", ct.c_uint16)):
        half = prefix == "H"
        a, b = (scalar * 1)(0x4000 if half else 2), (scalar * 1)(0x4200 if half else 3)
        # beta=0 must not read the previous C, including its NaN payload.
        c = (scalar * 1)(0x7e00 if half else float("nan"))
        alpha, beta = (scalar * 1)(0x3c00 if half else 1), (scalar * 1)(0)
        base = "cublas" + prefix + "gemm"
        value = 0x4600 if half else 6
        check(base + ("" if half else "_v2"), [Int] * 5 + [P, P, Int, P, Int, P, P, Int],
              [0, 0, 1, 1, 1, alpha, a, 1, b, 1, beta, c, 1], [c], fallback=skip_gemm, value=value)
        check(base + "StridedBatched", [Int] * 5 + [P, P, Int, L, P, Int, L, P, P, Int, L, Int],
              [0, 0, 1, 1, 1, alpha, a, 1, 1, b, 1, 1, beta, c, 1, 1, 1], [c], fallback=skip_gemm, value=value)
        if not half:
            aa, bb, cc = ((P * 1)(ct.addressof(buf)) for buf in (a, b, c))
            check(base + "Batched", [Int] * 5 + [P, P, Int, P, Int, P, P, Int, Int],
                  [0, 0, 1, 1, 1, alpha, aa, 1, bb, 1, beta, cc, 1, 1], [c], fallback=skip_gemm, value=6)

    a, b, c = (ct.c_float * 1)(2), (ct.c_float * 1)(3), (ct.c_float * 1)(float("nan"))
    alpha, beta = (ct.c_float * 1)(1), (ct.c_float * 1)(0)
    aa, bb, cc = ((P * 1)(ct.addressof(buf)) for buf in (a, b, c))
    for dtype in (0, 3):  # CUDA_R_32F executes; CUDA_R_8I has no CPU implementation.
        fallback = skip_gemm or dtype == 3
        check("cublasGemmEx", [Int] * 5 + [P, P, Int, Int, P, Int, Int, P, P, Int, Int, Int, Int],
              [0, 0, 1, 1, 1, alpha, a, dtype, 1, b, dtype, 1, beta, c, dtype, 1, 68, -1],
              [c], fallback=fallback, value=6, typed=True)
        check("cublasSgemmEx", [Int] * 5 + [P, P, Int, Int, P, Int, Int, P, P, Int, Int],
              [0, 0, 1, 1, 1, alpha, a, dtype, 1, b, dtype, 1, beta, c, dtype, 1],
              [c], fallback=fallback, value=6, typed=True)
        check("cublasGemmBatchedEx", [Int] * 5 + [P, P, Int, Int, P, Int, Int, P, P, Int, Int, Int, Int, Int],
              [0, 0, 1, 1, 1, alpha, aa, dtype, 1, bb, dtype, 1, beta, cc, dtype, 1, 1, 68, -1],
              [c], fallback=fallback, value=6, typed=True)
        check("cublasGemmStridedBatchedEx", [Int] * 5 + [P, P, Int, Int, L, P, Int, Int, L, P, P, Int, Int, L, Int, Int, Int],
              [0, 0, 1, 1, 1, alpha, a, dtype, 1, 1, b, dtype, 1, 1, beta, c, dtype, 1, 1, 1, 68, -1],
              [c], fallback=fallback, value=6, typed=True)

    # Scale storage follows computeType, not the matrix/output dtype.
    for compute_type, dtype in ((64, 2), (65, 2), (68, 2), (70, 1), (71, 1),
                                (2, 2), (0, 0), (1, 1), (999, 0)):
        matrix_scalar = {0: ct.c_float, 1: ct.c_double, 2: ct.c_uint16}[dtype]
        ma, mb, mc = ((matrix_scalar * 1)(v) for v in (
            (0x4000, 0x4200, 0x4400) if dtype == 2 else (2, 3, 4)
        ))
        if compute_type in (2, 64, 65):
            # Adjacent poison bytes make an erroneous float32 read deterministic.
            sa, sb = (ct.c_uint16 * 2)(0x3e00, 0x7fc0), (ct.c_uint16 * 2)(0xb800, 0x7fc0)
        else:
            scale_scalar = ct.c_double if dtype == 1 else ct.c_float
            sa, sb = (scale_scalar * 1)(1.5), (scale_scalar * 1)(-0.5)
        pa, pb, pc = ((P * 1)(ct.addressof(buf)) for buf in (ma, mb, mc))
        fallback = skip_gemm or compute_type == 999
        # CPU builds return early for an unsupported compute type, before recording.
        recorded = not cpu_simulation or compute_type != 999
        expected = 0x4700 if dtype == 2 else 7
        check("cublasGemmEx", [Int] * 5 + [P, P, Int, Int, P, Int, Int, P, P, Int, Int, Int, Int],
              [0, 0, 1, 1, 1, sa, ma, dtype, 1, mb, dtype, 1, sb, mc, dtype, 1, compute_type, -1],
              [mc], fallback=fallback, value=expected, unchanged=compute_type == 999, typed=recorded)
        check("cublasGemmBatchedEx", [Int] * 5 + [P, P, Int, Int, P, Int, Int, P, P, Int, Int, Int, Int, Int],
              [0, 0, 1, 1, 1, sa, pa, dtype, 1, pb, dtype, 1, sb, pc, dtype, 1, 1, compute_type, -1],
              [mc], fallback=fallback, value=expected, unchanged=compute_type == 999, typed=recorded)
        check("cublasGemmStridedBatchedEx", [Int] * 5 + [P, P, Int, Int, L, P, Int, Int, L, P, P, Int, Int, L, Int, Int, Int],
              [0, 0, 1, 1, 1, sa, ma, dtype, 1, 1, mb, dtype, 1, 1, sb, mc, dtype, 1, 1, 1, compute_type, -1],
              [mc], fallback=fallback, value=expected, unchanged=compute_type == 999, typed=recorded)

    c[0], beta[0] = float("inf"), -0.0
    check("cublasSgemm_v2", [Int] * 5 + [P, P, Int, P, Int, P, P, Int],
          [0, 0, 1, 1, 1, alpha, a, 1, b, 1, beta, c, 1], [c], fallback=skip_gemm, value=6)
    c[0], beta[0] = 123, 0.0
    lt_handle, descriptor, layout = P(), P(), P()
    assert bind("cublasLtCreate", [P])(ct.byref(lt_handle)) == 0
    assert bind("cublasLtMatmulDescCreate", [P, Int, Int])(ct.byref(descriptor), 68, 0) == 0
    assert bind("cublasLtMatrixLayoutCreate", [P, Int, ct.c_uint64, ct.c_uint64, L])(
        ct.byref(layout), 0, 1, 1, 1,
    ) == 0

    def lt_check(fallback, *, reject=False, value=6, unchanged=False, extra_outputs=()):
        check("cublasLtMatmul", [P] * 13 + [ct.c_size_t, P],
              [descriptor, alpha, a, layout, b, layout, beta, c, layout, c, layout, None, None, 0, None],
              [c, *extra_outputs], fallback=fallback, value=value, call_handle=lt_handle,
              reject=reject, unchanged=unchanged)

    lt_check(skip_gemm)
    if os.environ["FAKEGPU_CPU_GEMM_MODE"] == "auto":
        large_a, large_b, large_c = ((ct.c_float * 4)(*([value] * 4)) for value in (2, 3, 123))
        check("cublasSgemm_v2", [Int] * 5 + [P, P, Int, P, Int, P, P, Int],
              [0, 0, 2, 2, 2, alpha, large_a, 2, large_b, 2, beta, large_c, 2], [large_c])
        large_layout = P()
        assert bind("cublasLtMatrixLayoutCreate", [P, Int, ct.c_uint64, ct.c_uint64, L])(
            ct.byref(large_layout), 0, 2, 2, 2,
        ) == 0
        check("cublasLtMatmul", [P] * 13 + [ct.c_size_t, P],
              [descriptor, alpha, large_a, large_layout, large_b, large_layout, beta,
               large_c, large_layout, large_c, large_layout, None, None, 0, None], [large_c],
              call_handle=lt_handle)
        assert bind("cublasLtMatrixLayoutDestroy", [P])(large_layout) == 0
    for api, obj, attr, invalid in (
        ("cublasLtMatrixLayoutSetAttribute", layout, 1, 2),
        ("cublasLtMatrixLayoutSetAttribute", layout, 0, 3),
        ("cublasLtMatmulDescSetAttribute", descriptor, 2, 2),
    ):
        setter = bind(api, [P, Int, P, ct.c_size_t])
        value = Int(invalid)
        assert setter(obj, attr, ct.byref(value), ct.sizeof(value)) == 0
        lt_check(True)
        value.value = 0
        assert setter(obj, attr, ct.byref(value), ct.sizeof(value)) == 0
    # cuBLASLt only supports TRANSC=N, regardless of fallback policy or build.
    setter = bind("cublasLtMatmulDescSetAttribute", [P, Int, P, ct.c_size_t])
    for transc in (1, 2, 99):
        value = Int(transc)
        assert setter(descriptor, 5, ct.byref(value), ct.sizeof(value)) == 0
        lt_check(True, reject=True)
    value = Int(0)
    assert setter(descriptor, 5, ct.byref(value), ct.sizeof(value)) == 0
    lt_check(skip_gemm)

    bias = (ct.c_float * 1)(-2)
    bias_pointer = P(ct.addressof(bias))
    assert setter(descriptor, 8, ct.byref(bias_pointer), ct.sizeof(bias_pointer)) == 0
    # Auxiliary outputs and activation/bias gradients are not implemented.
    # Invalid combinations of otherwise known bits must not run either.
    for epilogue in (130, 134, 160, 164, 136, 152, 192, 208, 256, 512, 0, 3, 34, 999):
        value = Int(epilogue)
        assert setter(descriptor, 7, ct.byref(value), ct.sizeof(value)) == 0
        lt_check(True, unchanged=True, extra_outputs=(bias,))

    def gelu(x):
        return ct.c_float(0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))).value

    a[0], b[0] = 1, 1
    for epilogue, expected in ((1, 1), (2, 1), (4, -1), (6, 0), (32, gelu(1)), (36, gelu(-1))):
        value = Int(epilogue)
        assert setter(descriptor, 7, ct.byref(value), ct.sizeof(value)) == 0
        lt_check(skip_gemm, value=expected)
        assert bias[0] == -2
    a[0], b[0] = 2, 3
    value = Int(1)
    assert setter(descriptor, 7, ct.byref(value), ct.sizeof(value)) == 0
    lt_check(skip_gemm)
    assert bind("cublasLtMatrixLayoutDestroy", [P])(layout) == 0
    assert bind("cublasLtMatmulDescDestroy", [P])(descriptor) == 0
    assert bind("cublasLtDestroy", [P])(lt_handle) == 0
    assert bind("cublasDestroy_v2", [P])(handle) == 0
    return {"unsupported": dict(unsupported), "lt_matmuls": lt_matmuls, "typed_gemms": typed_gemms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--no-cpu-simulation", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    library = build / ("libcublas.dylib" if sys.platform == "darwin" else "libcublas.so.12")
    if args.child:
        print(json.dumps(probe(library, cpu_simulation=not args.no_cpu_simulation)))
        return 0
    if not library.is_file():
        parser.error(f"FakeGPU cuBLAS library not found: {library}")
    modes = ("exact",) if args.no_cpu_simulation else ("exact", "skip", "auto")
    with tempfile.TemporaryDirectory(prefix="fakegpu-cublas-policy-") as tmp:
        for mode in modes:
            for policy in ("allow", "warn", "error"):
                report_path = Path(tmp) / f"{mode}-{policy}.json"
                env = dict(os.environ)
                env.update({
                    "FAKEGPU_UNSUPPORTED_API": policy,
                    "FAKEGPU_CPU_GEMM_MODE": mode,
                    "FAKEGPU_CPU_GEMM_MAX_OPS": "1",  # force the auto threshold below
                    "FAKEGPU_TERMINAL_REPORT": "0",
                    "FAKEGPU_REPORT_PATH": str(report_path),
                    "FAKEGPU_SMI_STATE_DIR": tmp,
                    "FAKEGPU_SMI_STATE_PATH": str(Path(tmp) / "smi.json"),
                })
                command = [sys.executable, str(Path(__file__).resolve()), "--child", "--build-dir", str(build)]
                if args.no_cpu_simulation:
                    command.append("--no-cpu-simulation")
                completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
                assert completed.returncode == 0, (mode, policy, completed.stdout, completed.stderr)
                child = json.loads(completed.stdout)
                expected = child["unsupported"]
                report = json.loads(report_path.read_text(encoding="utf-8"))
                observed: Counter[str] = Counter()
                warnings: Counter[str] = Counter()
                for device in report["devices"]:
                    for event in device.get("unsupported_api_events", []):
                        assert event["policy"] == policy, event
                        observed[event["operation"]] += event["count"]
                        warnings[event["operation"]] += 1
                assert dict(observed) == expected, (mode, policy, observed, expected)
                lt_recorded = sum(device["compute"]["cublaslt_matmul"]["calls"] for device in report["devices"])
                if args.no_cpu_simulation:
                    # Without CPU execution every accepted cuBLASLt matmul is still counted.
                    assert lt_recorded == child["lt_matmuls"], (policy, lt_recorded, child["lt_matmuls"])
                # Every dtype-parameterised GEMM is broken down by dtype, as is every cuBLASLt matmul.
                by_dtype = sum(stats["calls"] for device in report["devices"]
                               for stats in device.get("gemm_by_dtype", {}).values())
                assert by_dtype == child["typed_gemms"] + lt_recorded, (mode, policy, by_dtype, child["typed_gemms"])
                for api in expected:
                    count = completed.stderr.count(f"[FakeGPU] warning: {api} is ")
                    assert count == (warnings[api] if policy == "warn" else 0), (mode, policy, api, count)
    print("cuBLAS fallback policy validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
