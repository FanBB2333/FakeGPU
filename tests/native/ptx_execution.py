#!/usr/bin/env python3
"""Check custom PTX execution through the CUDA Driver ABI, without a toolkit."""

from __future__ import annotations

import argparse
import ctypes as ct
import os
from pathlib import Path
import sys
import tempfile


HEADER = ".version 7.0\n.target sm_80\n.address_size 64\n"


def kernel(name: str, parameters: str, body: str) -> str:
    return HEADER + f".visible .entry {name}({parameters}) {{\n{body}\n}}\n"


def check(result: int, operation: str) -> None:
    assert result == 0, f"{operation} returned CUDA error {result}"


class Driver:
    def __init__(self, library: Path):
        self.api = ct.CDLL(str(library))
        signatures = {
            "cuInit": [ct.c_uint],
            "cuModuleLoadData": [ct.POINTER(ct.c_void_p), ct.c_void_p],
            "cuModuleLoad": [ct.POINTER(ct.c_void_p), ct.c_char_p],
            "cuModuleUnload": [ct.c_void_p],
            "cuModuleGetFunction": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            "cuModuleGetGlobal_v2": [ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_size_t),
                                     ct.c_void_p, ct.c_char_p],
            "cuMemAlloc_v2": [ct.POINTER(ct.c_uint64), ct.c_size_t],
            "cuMemFree_v2": [ct.c_uint64],
            "cuMemcpyHtoD_v2": [ct.c_uint64, ct.c_void_p, ct.c_size_t],
            "cuMemcpyDtoH_v2": [ct.c_void_p, ct.c_uint64, ct.c_size_t],
            "cuLaunchKernel": [ct.c_void_p, *([ct.c_uint] * 7), ct.c_void_p,
                               ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.api, name)
            function.argtypes = arguments
            function.restype = ct.c_int
        check(self.api.cuInit(0), "cuInit")
        self.allocations: list[ct.c_uint64] = []
        self.modules: list[ct.c_void_p] = []

    def upload(self, values: ct.Array) -> ct.c_uint64:
        pointer = ct.c_uint64()
        check(self.api.cuMemAlloc_v2(ct.byref(pointer), ct.sizeof(values)), "cuMemAlloc")
        self.allocations.append(pointer)
        check(self.api.cuMemcpyHtoD_v2(pointer, values, ct.sizeof(values)), "cuMemcpyHtoD")
        return pointer

    def download(self, pointer: ct.c_uint64, element: type, count: int) -> list:
        values = (element * count)()
        check(self.api.cuMemcpyDtoH_v2(values, pointer, ct.sizeof(values)), "cuMemcpyDtoH")
        return list(values)

    def load(self, source: str, name: str, *, from_file: bool = False) -> ct.c_void_p:
        module = ct.c_void_p()
        if from_file:
            with tempfile.TemporaryDirectory(prefix="fakegpu-ptx-") as directory:
                path = Path(directory) / "program.ptx"
                path.write_text(source)
                check(self.api.cuModuleLoad(ct.byref(module), os.fsencode(path)), "cuModuleLoad")
        else:
            check(self.api.cuModuleLoadData(ct.byref(module), source.encode()), "cuModuleLoadData")
        self.modules.append(module)
        function = ct.c_void_p()
        check(self.api.cuModuleGetFunction(ct.byref(function), module, name.encode()),
              "cuModuleGetFunction")
        return function

    def launch(self, function: ct.c_void_p, arguments: list, *, grid=(1, 1, 1),
               block=(1, 1, 1), shared=0) -> int:
        pointers = (ct.c_void_p * len(arguments))(
            *(ct.addressof(argument) for argument in arguments)
        )
        return self.api.cuLaunchKernel(function, *grid, *block, shared, None, pointers, None)

    def global_variable(self, name: str) -> tuple[ct.c_uint64, int]:
        pointer, size = ct.c_uint64(), ct.c_size_t()
        check(self.api.cuModuleGetGlobal_v2(ct.byref(pointer), ct.byref(size),
                                          self.modules[-1], name.encode()), "cuModuleGetGlobal")
        return pointer, size.value

    def close(self) -> None:
        for module in self.modules:
            check(self.api.cuModuleUnload(module), "cuModuleUnload")
        for pointer in self.allocations:
            check(self.api.cuMemFree_v2(pointer), "cuMemFree")


def test_parameterized_programs(driver: Driver) -> None:
    # Different entry names and instruction constants exercise the PTX itself.
    for name, multiplier, bias, from_file in (
        ("entry_74d9", 7, -11, False),
        ("unrelated_entry_a203", -3, 19, True),
    ):
        source = kernel(name, ".param .u64 output, .param .u64 input, "
                        ".param .s32 offset, .param .u32 count", f"""
            .reg .pred %p;
            .reg .b32 %r<8>;
            .reg .b64 %rd<6>;
            ld.param.u64 %rd1, [output];
            ld.param.u64 %rd2, [input];
            ld.param.s32 %r1, [offset];
            ld.param.u32 %r2, [count];
            mov.u32 %r3, %ctaid.x;
            mov.u32 %r4, %ntid.x;
            mov.u32 %r5, %tid.x;
            mad.lo.u32 %r3, %r3, %r4, %r5;
            setp.ge.u32 %p, %r3, %r2;
            @%p bra done;
            mul.wide.u32 %rd3, %r3, 4;
            add.u64 %rd4, %rd2, %rd3;
            ld.global.s32 %r6, [%rd4];
            mad.lo.s32 %r7, %r6, {multiplier}, %r1;
            add.u64 %rd5, %rd1, %rd3;
            st.global.s32 [%rd5], %r7;
        done:
            ret;
        """)
        values = [i * i - 17 for i in range(19)]
        input_pointer = driver.upload((ct.c_int32 * len(values))(*values))
        output = driver.upload((ct.c_int32 * 22)(*([-999] * 22)))
        function = driver.load(source, name, from_file=from_file)
        arguments = [output, input_pointer, ct.c_int32(bias), ct.c_uint32(len(values))]
        check(driver.launch(function, arguments, grid=(3, 1, 1), block=(8, 1, 1)), name)
        assert driver.download(output, ct.c_int32, 22) == [
            value * multiplier + bias for value in values
        ] + [-999] * 3


def test_wide_scalar_and_local_memory(driver: Driver) -> None:
    source = kernel("scalar_64_and_stack", ".param .u64 output, .param .s64 value", """
        .reg .b64 %rd<4>;
        .local .align 8 .b8 slot[8];
        ld.param.u64 %rd1, [output];
        ld.param.s64 %rd2, [value];
        st.local.s64 [slot], %rd2;
        ld.local.s64 %rd3, [slot];
        add.s64 %rd3, %rd3, -17;
        st.global.s64 [%rd1], %rd3;
        ret;
    """)
    output = driver.upload((ct.c_int64 * 1)(0))
    function = driver.load(source, "scalar_64_and_stack")
    value = -(1 << 45) + 13
    check(driver.launch(function, [output, ct.c_int64(value)]), "wide scalar and local memory")
    assert driver.download(output, ct.c_int64, 1) == [value - 17]

    class Packed(ct.Structure):
        _fields_ = [("output", ct.c_uint64), ("value", ct.c_int64)]

    packed = Packed(output.value, value + 100)
    size = ct.c_size_t(ct.sizeof(packed))
    extra = (ct.c_void_p * 5)(1, ct.addressof(packed), 2, ct.addressof(size), 0)
    check(driver.api.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, None, None, extra),
          "packed launch arguments")
    assert driver.download(output, ct.c_int64, 1) == [value + 83]
    size.value = ct.sizeof(packed) - 1
    assert driver.api.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, None, None, extra) == 1
    assert driver.download(output, ct.c_int64, 1) == [value + 83]


def test_globals_constants_and_device_call(driver: Driver) -> None:
    source = HEADER + """
        .global .align 4 .u32 state = 9;
        .const .align 4 .u32 factor = 3;
        .func (.param .b32 result) adjust(.param .b32 input) {
            .reg .b32 %r;
            ld.param.b32 %r, [input];
            add.u32 %r, %r, 7;
            st.param.b32 [result], %r;
            ret;
        }
        .visible .entry update_state() {
            .reg .b32 %r<4>;
            .param .b32 argument;
            .param .b32 returned;
            ld.global.u32 %r1, [state];
            ld.const.u32 %r2, [factor];
            mul.lo.u32 %r3, %r1, %r2;
            st.param.b32 [argument], %r3;
            call.uni (returned), adjust, (argument);
            ld.param.b32 %r3, [returned];
            st.global.u32 [state], %r3;
            ret;
        }
    """
    function = driver.load(source, "update_state")
    state, state_size = driver.global_variable("state")
    factor, factor_size = driver.global_variable("factor")
    assert state_size == factor_size == 4
    assert driver.download(state, ct.c_uint32, 1) == [9]
    assert driver.download(factor, ct.c_uint32, 1) == [3]
    check(driver.launch(function, []), "device function and module variables")
    assert driver.download(state, ct.c_uint32, 1) == [34]
    check(driver.launch(function, []), "persistent module state")
    assert driver.download(state, ct.c_uint32, 1) == [109]
    replacement = ct.c_uint32(5)
    check(driver.api.cuMemcpyHtoD_v2(state, ct.byref(replacement), ct.sizeof(replacement)),
          "write exported module variable")
    check(driver.launch(function, []), "module state after host update")
    assert driver.download(state, ct.c_uint32, 1) == [22]


def test_device_heap_persists_across_launches(driver: Driver) -> None:
    source = HEADER + """
        .global .align 8 .u64 saved_pointer = 0;
        .extern .func (.param .b64 returned) malloc(.param .b64 size);
        .extern .func free(.param .b64 pointer);
        .visible .entry allocate_value() {
            .reg .b64 %rd;
            .param .b64 size;
            .param .b64 returned;
            st.param.b64 [size], 8;
            call.uni (returned), malloc, (size);
            ld.param.u64 %rd, [returned];
            st.global.u64 [saved_pointer], %rd;
            st.global.u32 [%rd], 41;
            ret;
        }
        .visible .entry use_value(.param .u64 output) {
            .reg .b64 %rd<3>;
            .reg .b32 %r;
            ld.param.u64 %rd1, [output];
            ld.global.u64 %rd2, [saved_pointer];
            ld.global.u32 %r, [%rd2];
            add.u32 %r, %r, 7;
            st.global.u32 [%rd2], %r;
            st.global.u32 [%rd1], %r;
            ret;
        }
        .visible .entry release_value() {
            .reg .b64 %rd;
            .param .b64 pointer;
            ld.global.u64 %rd, [saved_pointer];
            st.param.b64 [pointer], %rd;
            call.uni free, (pointer);
            st.global.u64 [saved_pointer], 0;
            ret;
        }
    """
    allocate = driver.load(source, "allocate_value")
    module = driver.modules[-1]
    use, release = ct.c_void_p(), ct.c_void_p()
    check(driver.api.cuModuleGetFunction(ct.byref(use), module, b"use_value"), "lookup heap reader")
    check(driver.api.cuModuleGetFunction(ct.byref(release), module, b"release_value"), "lookup heap release")
    saved_pointer, size = driver.global_variable("saved_pointer")
    assert size == 8
    output = driver.upload((ct.c_uint32 * 1)(0))
    check(driver.launch(allocate, []), "device-side malloc")
    assert driver.download(saved_pointer, ct.c_uint64, 1)[0] != 0
    for expected in (48, 55):
        check(driver.launch(use, [output]), "device heap persists across launches")
        assert driver.download(output, ct.c_uint32, 1) == [expected]
    check(driver.launch(release, []), "device-side free")
    assert driver.download(saved_pointer, ct.c_uint64, 1) == [0]


def test_float_loop_and_2d_grid(driver: Driver) -> None:
    source = kernel("floating_recurrence", ".param .u64 output, .param .f32 bias", """
        .reg .pred %p;
        .reg .b32 %r<10>;
        .reg .b64 %rd<3>;
        .reg .f32 %f<3>;
        ld.param.u64 %rd1, [output];
        ld.param.f32 %f1, [bias];
        mov.u32 %r1, %ctaid.x;
        mov.u32 %r2, %ntid.x;
        mov.u32 %r3, %tid.x;
        mad.lo.u32 %r1, %r1, %r2, %r3;
        mov.u32 %r4, %ctaid.y;
        mov.u32 %r5, %ntid.y;
        mov.u32 %r6, %tid.y;
        mad.lo.u32 %r4, %r4, %r5, %r6;
        mov.u32 %r7, %nctaid.x;
        mul.lo.u32 %r7, %r7, %r2;
        mad.lo.u32 %r8, %r4, %r7, %r1;
        cvt.rn.f32.u32 %f2, %r8;
        mov.u32 %r9, 0;
    loop:
        fma.rn.f32 %f2, %f2, 0f3F000000, %f1;
        add.u32 %r9, %r9, 1;
        setp.lt.u32 %p, %r9, 4;
        @%p bra loop;
        mul.wide.u32 %rd2, %r8, 4;
        add.u64 %rd1, %rd1, %rd2;
        st.global.f32 [%rd1], %f2;
        ret;
    """)
    output = driver.upload((ct.c_float * 24)(*([-999] * 24)))
    function = driver.load(source, "floating_recurrence")
    check(driver.launch(function, [output, ct.c_float(-0.25)],
                        grid=(2, 2, 1), block=(3, 2, 1)), "float loop and 2D grid")
    expected = []
    for i in range(24):
        value = float(i)
        for _ in range(4):
            value = value * 0.5 - 0.25
        expected.append(value)
    assert driver.download(output, ct.c_float, 24) == expected


def test_thread_local_storage(driver: Driver) -> None:
    source = kernel("private_scratch", ".param .u64 output", """
        .reg .b32 %r<3>;
        .reg .b64 %rd<3>;
        .local .align 4 .b8 slot[4];
        ld.param.u64 %rd1, [output];
        mov.u32 %r1, %tid.x;
        st.local.u32 [slot], %r1;
        bar.sync 0;
        ld.local.u32 %r2, [slot];
        mul.wide.u32 %rd2, %r1, 4;
        add.u64 %rd1, %rd1, %rd2;
        st.global.u32 [%rd1], %r2;
        ret;
    """)
    output = driver.upload((ct.c_uint32 * 8)(*([999] * 8)))
    function = driver.load(source, "private_scratch")
    check(driver.launch(function, [output], block=(8, 1, 1)), "per-thread local storage")
    assert driver.download(output, ct.c_uint32, 8) == list(range(8))


def test_shared_barrier_and_atomics(driver: Driver) -> None:
    source = kernel("exchange_then_accumulate", ".param .u64 output", """
        .reg .b32 %r<8>;
        .reg .b64 %rd<3>;
        .shared .align 4 .b8 values[32];
        ld.param.u64 %rd1, [output];
        mov.u32 %r1, %tid.x;
        mul.lo.u32 %r2, %r1, 4;
        mov.u32 %r3, values;
        add.u32 %r3, %r3, %r2;
        add.u32 %r4, %r1, 1;
        st.shared.u32 [%r3], %r4;
        bar.sync 0;
        sub.u32 %r5, 7, %r1;
        mul.lo.u32 %r5, %r5, 4;
        mov.u32 %r6, values;
        add.u32 %r6, %r6, %r5;
        ld.shared.u32 %r7, [%r6];
        atom.global.add.u32 %r4, [%rd1], %r7;
        ret;
    """)
    output = driver.upload((ct.c_uint32 * 1)(0))
    function = driver.load(source, "exchange_then_accumulate")
    check(driver.launch(function, [output], grid=(3, 1, 1), block=(8, 1, 1)),
          "shared barrier and atomic add")
    assert driver.download(output, ct.c_uint32, 1) == [3 * sum(range(1, 9))]


def test_rejected_operations(driver: Driver) -> None:
    source = kernel("unsupported_instruction", ".param .u64 output", """
        .reg .b64 %rd;
        .reg .b32 %r;
        ld.param.u64 %rd, [output];
        fakegpu.unsupported.u32 %r, 7;
        st.global.u32 [%rd], %r;
        ret;
    """)
    output = driver.upload((ct.c_uint32 * 1)(12345))
    module = ct.c_void_p()
    result = driver.api.cuModuleLoadData(ct.byref(module), source.encode())
    if result == 0:
        driver.modules.append(module)
        function = ct.c_void_p()
        check(driver.api.cuModuleGetFunction(ct.byref(function), module, b"unsupported_instruction"),
              "unsupported kernel lookup")
        result = driver.launch(function, [output])
    assert result == 801, f"unsupported instruction returned {result}, expected 801"
    assert driver.download(output, ct.c_uint32, 1) == [12345]

    source = kernel("outside_allocation", ".param .u64 output", """
        .reg .b64 %rd;
        ld.param.u64 %rd, [output];
        add.u64 %rd, %rd, 4;
        st.global.u32 [%rd], 77;
        ret;
    """)
    function = driver.load(source, "outside_allocation")
    result = driver.launch(function, [output])
    assert result == 700, f"out-of-bounds store returned {result}, expected 700"
    assert driver.download(output, ct.c_uint32, 1) == [12345]
    assert driver.launch(function, [output], grid=(0, 1, 1)) == 1


def test_module_lifetime(driver: Driver) -> None:
    function = driver.load(kernel("temporary_entry", "", "ret;"), "temporary_entry")
    module = driver.modules.pop()
    missing = ct.c_void_p()
    assert driver.api.cuModuleGetFunction(ct.byref(missing), module, b"no_such_entry") == 500
    check(driver.api.cuModuleUnload(module), "unload temporary module")
    assert driver.launch(function, []) == 400
    assert driver.api.cuModuleUnload(module) == 400


def test_read_only_and_unmapped_memory(driver: Driver) -> None:
    source = kernel("store_word", ".param .u64 output", """
        .reg .b64 %rd;
        ld.param.u64 %rd, [output];
        st.global.u32 [%rd], 77;
        ret;
    """)
    source = source.replace(HEADER, HEADER + ".const .align 4 .u32 protected_value = 13;\n")
    function = driver.load(source, "store_word")
    constant, size = driver.global_variable("protected_value")
    assert size == 4
    assert driver.launch(function, [constant]) == 719
    assert driver.download(constant, ct.c_uint32, 1) == [13]

    pointer = driver.upload((ct.c_uint32 * 1)(123))
    driver.allocations.pop()
    check(driver.api.cuMemFree_v2(pointer), "free kernel argument storage")
    assert driver.launch(function, [pointer]) == 700
    host_value = ct.c_uint32(456)
    assert driver.launch(function, [ct.c_uint64(ct.addressof(host_value))]) == 700
    assert host_value.value == 456


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--expect-unavailable", action="store_true")
    args = parser.parse_args(argv)
    os.environ["FAKEGPU_MODE"] = "simulate"
    os.environ.setdefault("FAKEGPU_TERMINAL_REPORT", "0")
    os.environ.setdefault("FAKEGPU_REPORT_PATH", os.devnull)
    suffix = "libcuda.dylib" if sys.platform == "darwin" else "libcuda.so.1"
    driver = Driver(args.build_dir.resolve() / suffix)
    try:
        if args.expect_unavailable:
            source = kernel("valid_but_unavailable", "", "ret;")
            module = ct.c_void_p()
            result = driver.api.cuModuleLoadData(ct.byref(module), source.encode())
            assert result == 801, f"unavailable backend returned {result}, expected 801"
            print("Driver PTX: unavailable backend explicitly rejected module execution")
            return 0
        test_parameterized_programs(driver)
        test_wide_scalar_and_local_memory(driver)
        test_globals_constants_and_device_call(driver)
        test_device_heap_persists_across_launches(driver)
        test_float_loop_and_2d_grid(driver)
        test_thread_local_storage(driver)
        test_shared_barrier_and_atomics(driver)
        test_rejected_operations(driver)
        test_module_lifetime(driver)
        test_read_only_and_unmapped_memory(driver)
    finally:
        driver.close()
    print("Driver PTX: custom programs, scalar arguments, memory, barriers, atomics and errors passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
