#pragma once

#include "cuda_driver_defs.hpp"
#include "../core/backend_config.hpp"
#include "../core/real_cuda_loader.hpp"
#include "../core/logging.hpp"

namespace fake_gpu {

// Function pointer types for CUDA Driver API
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDriverGetVersion_fn)(int*);
typedef CUresult (*cuDeviceGetCount_fn)(int*);
typedef CUresult (*cuDeviceGet_fn)(CUdevice*, int);
typedef CUresult (*cuDeviceGetName_fn)(char*, int, CUdevice);
typedef CUresult (*cuDeviceGetAttribute_fn)(int*, CUdevice_attribute, CUdevice);
typedef CUresult (*cuDeviceTotalMem_fn)(size_t*, CUdevice);
typedef CUresult (*cuDeviceGetUuid_fn)(CUuuid*, CUdevice);
typedef CUresult (*cuCtxCreate_fn)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_fn)(CUcontext);
typedef CUresult (*cuCtxSetCurrent_fn)(CUcontext);
typedef CUresult (*cuCtxGetCurrent_fn)(CUcontext*);
typedef CUresult (*cuCtxGetDevice_fn)(CUdevice*);
typedef CUresult (*cuCtxSynchronize_fn)(void);
typedef CUresult (*cuMemAlloc_fn)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_fn)(CUdeviceptr);
typedef CUresult (*cuMemcpyDtoH_fn)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyHtoD_fn)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuMemcpyDtoD_fn)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*cuMemAllocManaged_fn)(CUdeviceptr*, size_t, unsigned int);
typedef CUresult (*cuMemAllocHost_fn)(void**, size_t);
typedef CUresult (*cuMemFreeHost_fn)(void*);
typedef CUresult (*cuMemHostAlloc_fn)(void**, size_t, unsigned int);
typedef CUresult (*cuMemHostGetDevicePointer_fn)(CUdeviceptr*, void*, unsigned int);
typedef CUresult (*cuMemGetInfo_fn)(size_t*, size_t*);
typedef CUresult (*cuStreamCreate_fn)(CUstream*, unsigned int);
typedef CUresult (*cuStreamDestroy_fn)(CUstream);
typedef CUresult (*cuStreamSynchronize_fn)(CUstream);
typedef CUresult (*cuStreamQuery_fn)(CUstream);
typedef CUresult (*cuStreamWaitEvent_fn)(CUstream, CUevent, unsigned int);
typedef CUresult (*cuStreamGetFlags_fn)(CUstream, unsigned int*);
typedef CUresult (*cuEventCreate_fn)(CUevent*, unsigned int);
typedef CUresult (*cuEventDestroy_fn)(CUevent);
typedef CUresult (*cuEventRecord_fn)(CUevent, CUstream);
typedef CUresult (*cuEventSynchronize_fn)(CUevent);
typedef CUresult (*cuEventQuery_fn)(CUevent);
typedef CUresult (*cuEventElapsedTime_fn)(float*, CUevent, CUevent);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void**, void**);
typedef CUresult (*cuModuleLoad_fn)(CUmodule*, const char*);
typedef CUresult (*cuModuleLoadData_fn)(CUmodule*, const void*);
typedef CUresult (*cuModuleUnload_fn)(CUmodule);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuDevicePrimaryCtxRetain_fn)(CUcontext*, CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRelease_fn)(CUdevice);
typedef CUresult (*cuDevicePrimaryCtxReset_fn)(CUdevice);
typedef CUresult (*cuGetProcAddress_fn)(const char*, void**, int, unsigned long long);

// Passthrough dispatcher for CUDA Driver API
class CudaDriverPassthrough {
public:
    static CudaDriverPassthrough& instance() {
        static CudaDriverPassthrough pt;
        return pt;
    }

    // Initialize and cache function pointers
    bool initialize() {
        if (initialized_) return available_;
        initialized_ = true;

        RealCudaLoader& loader = RealCudaLoader::instance();
        if (!loader.initialize_driver()) {
            FGPU_LOG("[CudaDriverPassthrough] Real CUDA loader not available\n");
            return false;
        }

        // Cache function pointers
        #define LOAD_FUNC(name) \
            real_##name = (name##_fn)loader.get_cuda_driver_func(#name); \
            if (!real_##name) real_##name = (name##_fn)loader.get_cuda_driver_func(#name "_v2")

        // Several legacy Driver API entry points use 32-bit pointer/size
        // signatures.  On a 64-bit process we must prefer their _v2 symbols;
        // calling the legacy cuMemAlloc, for example, returns
        // CUDA_ERROR_INVALID_CONTEXT even with a valid current context.
        #define LOAD_FUNC_V2(name) \
            real_##name = (name##_fn)loader.get_cuda_driver_func(#name "_v2"); \
            if (!real_##name) real_##name = (name##_fn)loader.get_cuda_driver_func(#name)

        LOAD_FUNC(cuInit);
        LOAD_FUNC(cuDriverGetVersion);
        LOAD_FUNC(cuDeviceGetCount);
        LOAD_FUNC(cuDeviceGet);
        LOAD_FUNC(cuDeviceGetName);
        LOAD_FUNC(cuDeviceGetAttribute);
        LOAD_FUNC_V2(cuDeviceTotalMem);
        LOAD_FUNC(cuDeviceGetUuid);
        LOAD_FUNC_V2(cuCtxCreate);
        LOAD_FUNC_V2(cuCtxDestroy);
        LOAD_FUNC(cuCtxSetCurrent);
        LOAD_FUNC(cuCtxGetCurrent);
        LOAD_FUNC(cuCtxGetDevice);
        LOAD_FUNC(cuCtxSynchronize);
        LOAD_FUNC_V2(cuMemAlloc);
        LOAD_FUNC_V2(cuMemFree);
        LOAD_FUNC_V2(cuMemcpyDtoH);
        LOAD_FUNC_V2(cuMemcpyHtoD);
        LOAD_FUNC_V2(cuMemcpyDtoD);
        LOAD_FUNC(cuMemAllocManaged);
        LOAD_FUNC_V2(cuMemAllocHost);
        LOAD_FUNC(cuMemFreeHost);
        LOAD_FUNC(cuMemHostAlloc);
        LOAD_FUNC_V2(cuMemHostGetDevicePointer);
        LOAD_FUNC_V2(cuMemGetInfo);
        LOAD_FUNC(cuStreamCreate);
        LOAD_FUNC_V2(cuStreamDestroy);
        LOAD_FUNC(cuStreamSynchronize);
        LOAD_FUNC(cuStreamQuery);
        LOAD_FUNC(cuStreamWaitEvent);
        LOAD_FUNC(cuStreamGetFlags);
        LOAD_FUNC(cuEventCreate);
        LOAD_FUNC_V2(cuEventDestroy);
        LOAD_FUNC(cuEventRecord);
        LOAD_FUNC(cuEventSynchronize);
        LOAD_FUNC(cuEventQuery);
        LOAD_FUNC(cuEventElapsedTime);
        LOAD_FUNC(cuLaunchKernel);
        LOAD_FUNC(cuModuleLoad);
        LOAD_FUNC(cuModuleLoadData);
        LOAD_FUNC(cuModuleUnload);
        LOAD_FUNC(cuModuleGetFunction);
        LOAD_FUNC(cuDevicePrimaryCtxRetain);
        LOAD_FUNC(cuDevicePrimaryCtxRelease);
        LOAD_FUNC(cuDevicePrimaryCtxReset);
        LOAD_FUNC(cuGetProcAddress);

        #undef LOAD_FUNC
        #undef LOAD_FUNC_V2

        available_ = (real_cuInit != nullptr);
        if (available_) {
            FGPU_LOG("[CudaDriverPassthrough] Initialized successfully\n");
        }
        return available_;
    }

    // Passthrough functions: CUDA_ERROR_NOT_INITIALIZED when the real entry point is missing.
    CUresult cuInit(unsigned int flags) { return forward(real_cuInit, flags); }

    CUresult cuDriverGetVersion(int* version) { return forward(real_cuDriverGetVersion, version); }

    CUresult cuDeviceGetCount(int* count) { return forward(real_cuDeviceGetCount, count); }

    CUresult cuDeviceGet(CUdevice* device, int ordinal) { return forward(real_cuDeviceGet, device, ordinal); }

    CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
        return forward(real_cuDeviceGetName, name, len, dev);
    }

    CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
        return forward(real_cuDeviceGetAttribute, pi, attrib, dev);
    }

    CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) { return forward(real_cuDeviceTotalMem, bytes, dev); }

    CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) { return forward(real_cuDeviceGetUuid, uuid, dev); }

    CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
        return forward(real_cuCtxCreate, pctx, flags, dev);
    }

    CUresult cuCtxDestroy(CUcontext ctx) { return forward(real_cuCtxDestroy, ctx); }

    CUresult cuCtxSetCurrent(CUcontext ctx) { return forward(real_cuCtxSetCurrent, ctx); }

    CUresult cuCtxGetCurrent(CUcontext* pctx) { return forward(real_cuCtxGetCurrent, pctx); }

    CUresult cuCtxGetDevice(CUdevice* device) { return forward(real_cuCtxGetDevice, device); }

    CUresult cuCtxSynchronize() { return forward(real_cuCtxSynchronize); }

    CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) { return forward(real_cuMemAlloc, dptr, bytesize); }

    CUresult cuMemFree(CUdeviceptr dptr) { return forward(real_cuMemFree, dptr); }

    CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t byteCount) {
        return forward(real_cuMemcpyDtoH, dstHost, srcDevice, byteCount);
    }

    CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t byteCount) {
        return forward(real_cuMemcpyHtoD, dstDevice, srcHost, byteCount);
    }

    CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount) {
        return forward(real_cuMemcpyDtoD, dstDevice, srcDevice, byteCount);
    }

    CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
        return forward(real_cuMemAllocManaged, dptr, bytesize, flags);
    }

    CUresult cuMemAllocHost(void** pp, size_t bytesize) { return forward(real_cuMemAllocHost, pp, bytesize); }

    CUresult cuMemFreeHost(void* p) { return forward(real_cuMemFreeHost, p); }

    CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
        return forward(real_cuMemHostAlloc, pp, bytesize, flags);
    }

    CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int flags) {
        return forward(real_cuMemHostGetDevicePointer, pdptr, p, flags);
    }

    CUresult cuMemGetInfo(size_t* free, size_t* total) { return forward(real_cuMemGetInfo, free, total); }

    CUresult cuStreamCreate(CUstream* phStream, unsigned int flags) {
        return forward(real_cuStreamCreate, phStream, flags);
    }

    CUresult cuStreamDestroy(CUstream hStream) { return forward(real_cuStreamDestroy, hStream); }

    CUresult cuStreamSynchronize(CUstream hStream) { return forward(real_cuStreamSynchronize, hStream); }

    CUresult cuStreamQuery(CUstream hStream) { return forward(real_cuStreamQuery, hStream); }

    CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
        return forward(real_cuStreamWaitEvent, hStream, hEvent, flags);
    }

    CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
        return forward(real_cuStreamGetFlags, hStream, flags);
    }

    CUresult cuEventCreate(CUevent* phEvent, unsigned int flags) { return forward(real_cuEventCreate, phEvent, flags); }

    CUresult cuEventDestroy(CUevent hEvent) { return forward(real_cuEventDestroy, hEvent); }

    CUresult cuEventRecord(CUevent hEvent, CUstream hStream) { return forward(real_cuEventRecord, hEvent, hStream); }

    CUresult cuEventSynchronize(CUevent hEvent) { return forward(real_cuEventSynchronize, hEvent); }

    CUresult cuEventQuery(CUevent hEvent) { return forward(real_cuEventQuery, hEvent); }

    CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
        return forward(real_cuEventElapsedTime, pMilliseconds, hStart, hEnd);
    }

    CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                            unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                            unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
        return forward(real_cuLaunchKernel, f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                       sharedMemBytes, hStream, kernelParams, extra);
    }

    CUresult cuModuleLoad(CUmodule* module, const char* fname) { return forward(real_cuModuleLoad, module, fname); }

    CUresult cuModuleLoadData(CUmodule* module, const void* image) {
        return forward(real_cuModuleLoadData, module, image);
    }

    CUresult cuModuleUnload(CUmodule hmod) { return forward(real_cuModuleUnload, hmod); }

    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
        return forward(real_cuModuleGetFunction, hfunc, hmod, name);
    }

    CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
        return forward(real_cuDevicePrimaryCtxRetain, pctx, dev);
    }

    CUresult cuDevicePrimaryCtxRelease(CUdevice dev) { return forward(real_cuDevicePrimaryCtxRelease, dev); }

    CUresult cuDevicePrimaryCtxReset(CUdevice dev) { return forward(real_cuDevicePrimaryCtxReset, dev); }

    CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, unsigned long long flags) {
        return forward(real_cuGetProcAddress, symbol, pfn, cudaVersion, flags);
    }

    // Get any function by name from real library
    void* getRealFunction(const char* name) {
        return RealCudaLoader::instance().get_cuda_driver_func(name);
    }

private:
    CudaDriverPassthrough() = default;

    template <class Fn, class... Args>
    static CUresult forward(Fn fn, Args... args) {
        return fn ? fn(args...) : CUDA_ERROR_NOT_INITIALIZED;
    }

    bool initialized_ = false;
    bool available_ = false;

    // Cached function pointers
    cuInit_fn real_cuInit = nullptr;
    cuDriverGetVersion_fn real_cuDriverGetVersion = nullptr;
    cuDeviceGetCount_fn real_cuDeviceGetCount = nullptr;
    cuDeviceGet_fn real_cuDeviceGet = nullptr;
    cuDeviceGetName_fn real_cuDeviceGetName = nullptr;
    cuDeviceGetAttribute_fn real_cuDeviceGetAttribute = nullptr;
    cuDeviceTotalMem_fn real_cuDeviceTotalMem = nullptr;
    cuDeviceGetUuid_fn real_cuDeviceGetUuid = nullptr;
    cuCtxCreate_fn real_cuCtxCreate = nullptr;
    cuCtxDestroy_fn real_cuCtxDestroy = nullptr;
    cuCtxSetCurrent_fn real_cuCtxSetCurrent = nullptr;
    cuCtxGetCurrent_fn real_cuCtxGetCurrent = nullptr;
    cuCtxGetDevice_fn real_cuCtxGetDevice = nullptr;
    cuCtxSynchronize_fn real_cuCtxSynchronize = nullptr;
    cuMemAlloc_fn real_cuMemAlloc = nullptr;
    cuMemFree_fn real_cuMemFree = nullptr;
    cuMemcpyDtoH_fn real_cuMemcpyDtoH = nullptr;
    cuMemcpyHtoD_fn real_cuMemcpyHtoD = nullptr;
    cuMemcpyDtoD_fn real_cuMemcpyDtoD = nullptr;
    cuMemAllocManaged_fn real_cuMemAllocManaged = nullptr;
    cuMemAllocHost_fn real_cuMemAllocHost = nullptr;
    cuMemFreeHost_fn real_cuMemFreeHost = nullptr;
    cuMemHostAlloc_fn real_cuMemHostAlloc = nullptr;
    cuMemHostGetDevicePointer_fn real_cuMemHostGetDevicePointer = nullptr;
    cuMemGetInfo_fn real_cuMemGetInfo = nullptr;
    cuStreamCreate_fn real_cuStreamCreate = nullptr;
    cuStreamDestroy_fn real_cuStreamDestroy = nullptr;
    cuStreamSynchronize_fn real_cuStreamSynchronize = nullptr;
    cuStreamQuery_fn real_cuStreamQuery = nullptr;
    cuStreamWaitEvent_fn real_cuStreamWaitEvent = nullptr;
    cuStreamGetFlags_fn real_cuStreamGetFlags = nullptr;
    cuEventCreate_fn real_cuEventCreate = nullptr;
    cuEventDestroy_fn real_cuEventDestroy = nullptr;
    cuEventRecord_fn real_cuEventRecord = nullptr;
    cuEventSynchronize_fn real_cuEventSynchronize = nullptr;
    cuEventQuery_fn real_cuEventQuery = nullptr;
    cuEventElapsedTime_fn real_cuEventElapsedTime = nullptr;
    cuLaunchKernel_fn real_cuLaunchKernel = nullptr;
    cuModuleLoad_fn real_cuModuleLoad = nullptr;
    cuModuleLoadData_fn real_cuModuleLoadData = nullptr;
    cuModuleUnload_fn real_cuModuleUnload = nullptr;
    cuModuleGetFunction_fn real_cuModuleGetFunction = nullptr;
    cuDevicePrimaryCtxRetain_fn real_cuDevicePrimaryCtxRetain = nullptr;
    cuDevicePrimaryCtxRelease_fn real_cuDevicePrimaryCtxRelease = nullptr;
    cuDevicePrimaryCtxReset_fn real_cuDevicePrimaryCtxReset = nullptr;
    cuGetProcAddress_fn real_cuGetProcAddress = nullptr;
};

} // namespace fake_gpu
