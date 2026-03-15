#!/usr/bin/env python3

import ctypes
import pathlib


CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_STREAM_NON_BLOCKING = 1
CUDA_CAPTURE_NONE = 0
CUDA_CAPTURE_ACTIVE = 1


ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBCUDART = ROOT / "build" / "libcudart.so.12"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    cudart = ctypes.CDLL(str(LIBCUDART), mode=ctypes.RTLD_GLOBAL)

    cudart.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudart.cudaGetDeviceCount.restype = ctypes.c_int

    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int

    cudart.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cudart.cudaStreamCreateWithFlags.restype = ctypes.c_int

    cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamDestroy.restype = ctypes.c_int

    cudart.cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    cudart.cudaStreamWaitEvent.restype = ctypes.c_int

    cudart.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cudart.cudaStreamBeginCapture.restype = ctypes.c_int

    cudart.cudaStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaStreamEndCapture.restype = ctypes.c_int

    cudart.cudaStreamIsCapturing.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    cudart.cudaStreamIsCapturing.restype = ctypes.c_int

    cudart.cudaStreamGetCaptureInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    cudart.cudaStreamGetCaptureInfo.restype = ctypes.c_int

    cudart.cudaStreamGetCaptureInfo_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    cudart.cudaStreamGetCaptureInfo_v2.restype = ctypes.c_int

    cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaEventCreate.restype = ctypes.c_int

    cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaEventDestroy.restype = ctypes.c_int

    cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaEventRecord.restype = ctypes.c_int

    cudart.cudaEventRecordWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    cudart.cudaEventRecordWithFlags.restype = ctypes.c_int

    cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaEventSynchronize.restype = ctypes.c_int

    cudart.cudaEventQuery.argtypes = [ctypes.c_void_p]
    cudart.cudaEventQuery.restype = ctypes.c_int

    cudart.cudaEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    cudart.cudaEventElapsedTime.restype = ctypes.c_int

    cudart.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaGraphDestroy.restype = ctypes.c_int

    device_count = ctypes.c_int()
    require(cudart.cudaGetDeviceCount(ctypes.byref(device_count)) == CUDA_SUCCESS, "cudaGetDeviceCount should succeed")
    require(device_count.value > 0, "at least one fake device should be visible")
    require(cudart.cudaSetDevice(0) == CUDA_SUCCESS, "cudaSetDevice(0) should succeed")

    stream = ctypes.c_void_p()
    require(
        cudart.cudaStreamCreateWithFlags(ctypes.byref(stream), CUDA_STREAM_NON_BLOCKING) == CUDA_SUCCESS,
        "cudaStreamCreateWithFlags should succeed",
    )
    require(stream.value not in (None, 0), "stream should be non-null")

    event = ctypes.c_void_p()
    require(cudart.cudaEventCreate(ctypes.byref(event)) == CUDA_SUCCESS, "cudaEventCreate should succeed")
    require(event.value not in (None, 0), "event should be non-null")

    require(cudart.cudaEventRecord(event, stream) == CUDA_SUCCESS, "cudaEventRecord should succeed")
    require(cudart.cudaEventRecordWithFlags(event, stream, 0) == CUDA_SUCCESS, "cudaEventRecordWithFlags should succeed")
    require(cudart.cudaEventQuery(event) == CUDA_SUCCESS, "cudaEventQuery should succeed for a live event")
    require(cudart.cudaEventSynchronize(event) == CUDA_SUCCESS, "cudaEventSynchronize should succeed for a live event")
    require(cudart.cudaStreamWaitEvent(stream, event, 0) == CUDA_SUCCESS, "cudaStreamWaitEvent should succeed for live handles")

    elapsed_ms = ctypes.c_float()
    require(
        cudart.cudaEventElapsedTime(ctypes.byref(elapsed_ms), event, event) == CUDA_SUCCESS,
        "cudaEventElapsedTime should succeed for live events",
    )
    require(elapsed_ms.value == 0.0, "simulate-mode elapsed time should remain zero")

    capture_status = ctypes.c_int(-1)
    capture_id = ctypes.c_ulonglong(99)
    require(
        cudart.cudaStreamIsCapturing(stream, ctypes.byref(capture_status)) == CUDA_SUCCESS,
        "cudaStreamIsCapturing should succeed before capture",
    )
    require(capture_status.value == CUDA_CAPTURE_NONE, "stream should not be capturing before begin")
    require(
        cudart.cudaStreamGetCaptureInfo(stream, ctypes.byref(capture_status), ctypes.byref(capture_id)) == CUDA_SUCCESS,
        "cudaStreamGetCaptureInfo should succeed before capture",
    )
    require(capture_status.value == CUDA_CAPTURE_NONE, "capture info should report not capturing before begin")
    require(capture_id.value == 0, "capture id should be zero before begin")

    require(cudart.cudaStreamBeginCapture(stream, 0) == CUDA_SUCCESS, "cudaStreamBeginCapture should succeed")
    require(
        cudart.cudaStreamBeginCapture(stream, 0) == CUDA_ERROR_INVALID_VALUE,
        "nested cudaStreamBeginCapture should fail",
    )
    require(
        cudart.cudaStreamIsCapturing(stream, ctypes.byref(capture_status)) == CUDA_SUCCESS,
        "cudaStreamIsCapturing should succeed during capture",
    )
    require(capture_status.value == CUDA_CAPTURE_ACTIVE, "stream should report capturing after begin")
    require(
        cudart.cudaStreamGetCaptureInfo(stream, ctypes.byref(capture_status), ctypes.byref(capture_id)) == CUDA_SUCCESS,
        "cudaStreamGetCaptureInfo should succeed during capture",
    )
    require(capture_status.value == CUDA_CAPTURE_ACTIVE, "capture info should report active capture")
    require(capture_id.value > 0, "capture id should be assigned during capture")

    capture_id_v2 = ctypes.c_ulonglong(0)
    graph_handle = ctypes.c_void_p()
    deps_ptr = ctypes.c_void_p()
    deps_count = ctypes.c_size_t(123)
    require(
        cudart.cudaStreamGetCaptureInfo_v2(
            stream,
            ctypes.byref(capture_status),
            ctypes.byref(capture_id_v2),
            ctypes.byref(graph_handle),
            ctypes.byref(deps_ptr),
            ctypes.byref(deps_count),
        ) == CUDA_SUCCESS,
        "cudaStreamGetCaptureInfo_v2 should succeed during capture",
    )
    require(capture_status.value == CUDA_CAPTURE_ACTIVE, "v2 capture info should report active capture")
    require(capture_id_v2.value == capture_id.value, "v2 capture id should match v1")
    require(graph_handle.value in (None, 0), "v2 graph handle should remain null in the stub")
    require(deps_count.value == 0, "v2 dependency count should be zero in the stub")

    captured_graph = ctypes.c_void_p()
    require(
        cudart.cudaStreamEndCapture(stream, ctypes.byref(captured_graph)) == CUDA_SUCCESS,
        "cudaStreamEndCapture should succeed",
    )
    require(captured_graph.value not in (None, 0), "cudaStreamEndCapture should produce a graph handle")
    require(cudart.cudaGraphDestroy(captured_graph) == CUDA_SUCCESS, "cudaGraphDestroy should succeed")
    require(
        cudart.cudaStreamIsCapturing(stream, ctypes.byref(capture_status)) == CUDA_SUCCESS,
        "cudaStreamIsCapturing should succeed after capture",
    )
    require(capture_status.value == CUDA_CAPTURE_NONE, "stream should not be capturing after end")
    require(
        cudart.cudaStreamEndCapture(stream, ctypes.byref(captured_graph)) == CUDA_ERROR_INVALID_VALUE,
        "cudaStreamEndCapture without an active capture should fail",
    )

    require(cudart.cudaEventDestroy(event) == CUDA_SUCCESS, "cudaEventDestroy should succeed")
    require(cudart.cudaEventQuery(event) == CUDA_ERROR_INVALID_VALUE, "destroyed event should become invalid")
    require(cudart.cudaEventRecord(event, stream) == CUDA_ERROR_INVALID_VALUE, "destroyed event record should fail")
    require(
        cudart.cudaStreamWaitEvent(stream, event, 0) == CUDA_ERROR_INVALID_VALUE,
        "waiting on a destroyed event should fail",
    )
    require(
        cudart.cudaEventElapsedTime(ctypes.byref(elapsed_ms), event, event) == CUDA_ERROR_INVALID_VALUE,
        "elapsed time on a destroyed event should fail",
    )

    require(cudart.cudaStreamDestroy(stream) == CUDA_SUCCESS, "cudaStreamDestroy should succeed")
    require(
        cudart.cudaStreamIsCapturing(stream, ctypes.byref(capture_status)) == CUDA_ERROR_INVALID_VALUE,
        "capture query should reject a destroyed stream",
    )
    require(
        cudart.cudaStreamBeginCapture(stream, 0) == CUDA_ERROR_INVALID_VALUE,
        "capture begin should reject a destroyed stream",
    )

    print("cudart event and capture test passed")


if __name__ == "__main__":
    main()
