#pragma once

#include "cuda_driver_defs.hpp"
#include "../core/backend_config.hpp"
#include "../core/real_cuda_loader.hpp"
#include "../core/hybrid_memory_manager.hpp"
#include "../core/logging.hpp"
#include <cstring>

namespace fake_gpu {

// Function pointer types for CUDA Driver API
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDriverGetVersion_fn)(int*);
typedef CUresult (*cuDeviceGetCount_fn)(int*);
typedef CUresult (*cuDeviceGet_fn)(CUdevice*, int);
typedef CUresult (*cuDeviceGetName_fn)(char*, int, CUdevice);
typedef CUresult (*cuDeviceGetAttribute_fn)(int*, CUdevice_attribute, CUdevice);
typedef CUresult (*cuDeviceTotalMem_fn)(size_t*, CUdevice);
typedef CUresult (*cuDeviceGetUuid_fn)(char*, CUdevice);
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

    bool is_available() const { return available_; }

    // Initialize and cache function pointers
    bool initialize() {
        if (initialized_) return available_;
        initialized_ = true;

        RealCudaLoader& loader = RealCudaLoader::instance();
        if (!loader.initialize()) {
            FGPU_LOG("[CudaDriverPassthrough] Real CUDA loader not available\n");
            return false;
        }

        // Cache function pointers
        #define LOAD_FUNC(name) \
            real_##name = (name##_fn)loader.get_cuda_driver_func(#name); \
            if (!real_##name) real_##name = (name##_fn)loader.get_cuda_driver_func(#name "_v2")

        LOAD_FUNC(cuInit);
        LOAD_FUNC(cuDriverGetVersion);
        LOAD_FUNC(cuDeviceGetCount);
        LOAD_FUNC(cuDeviceGet);
        LOAD_FUNC(cuDeviceGetName);
        LOAD_FUNC(cuDeviceGetAttribute);
        LOAD_FUNC(cuDeviceTotalMem);
        LOAD_FUNC(cuDeviceGetUuid);
        LOAD_FUNC(cuCtxCreate);
        LOAD_FUNC(cuCtxDestroy);
        LOAD_FUNC(cuCtxSetCurrent);
        LOAD_FUNC(cuCtxGetCurrent);
        LOAD_FUNC(cuCtxGetDevice);
        LOAD_FUNC(cuCtxSynchronize);
        LOAD_FUNC(cuMemAlloc);
        LOAD_FUNC(cuMemFree);
        LOAD_FUNC(cuMemcpyDtoH);
        LOAD_FUNC(cuMemcpyHtoD);
        LOAD_FUNC(cuMemcpyDtoD);
        LOAD_FUNC(cuMemAllocManaged);
        LOAD_FUNC(cuMemAllocHost);
        LOAD_FUNC(cuMemFreeHost);
        LOAD_FUNC(cuMemHostAlloc);
        LOAD_FUNC(cuMemHostGetDevicePointer);
        LOAD_FUNC(cuMemGetInfo);
        LOAD_FUNC(cuStreamCreate);
        LOAD_FUNC(cuStreamDestroy);
        LOAD_FUNC(cuStreamSynchronize);
        LOAD_FUNC(cuStreamQuery);
        LOAD_FUNC(cuStreamWaitEvent);
        LOAD_FUNC(cuStreamGetFlags);
        LOAD_FUNC(cuEventCreate);
        LOAD_FUNC(cuEventDestroy);
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

        available_ = (real_cuInit != nullptr);
        if (available_) {
            FGPU_LOG("[CudaDriverPassthrough] Initialized successfully\n");
        }
        return available_;
    }

    // Passthrough functions
    CUresult cuInit(unsigned int flags) {
        if (!real_cuInit) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuInit(flags);
    }

    CUresult cuDriverGetVersion(int* version) {
        if (!real_cuDriverGetVersion) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuDriverGetVersion(version);
    }

    CUresult cuDeviceGetCount(int* count) {
        if (!real_cuDeviceGetCount) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuDeviceGetCount(count);
    }

    CUresult cuDeviceGet(CUdevice* device, int ordinal) {
        if (!real_cuDeviceGet) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuDeviceGet(device, ordinal);
    }

    CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
        if (!real_cuDeviceGetName) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuDeviceGetName(name, len, dev);
    }

    CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
        if (!real_cuDeviceGetAttribute) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuDeviceGetAttribute(pi, attrib, dev);
    }

    CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
        if (!real_cuDeviceTotalMem) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuDeviceTotalMem(bytes, dev);
    }

    CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
        if (!real_cuCtxCreate) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuCtxCreate(pctx, flags, dev);
    }

    CUresult cuCtxDestroy(CUcontext ctx) {
        if (!real_cuCtxDestroy) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuCtxDestroy(ctx);
    }

    CUresult cuCtxSetCurrent(CUcontext ctx) {
        if (!real_cuCtxSetCurrent) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuCtxSetCurrent(ctx);
    }

    CUresult cuCtxGetCurrent(CUcontext* pctx) {
        if (!real_cuCtxGetCurrent) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuCtxGetCurrent(pctx);
    }

    CUresult cuCtxGetDevice(CUdevice* device) {
        if (!real_cuCtxGetDevice) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuCtxGetDevice(device);
    }

    CUresult cuCtxSynchronize() {
        if (!real_cuCtxSynchronize) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuCtxSynchronize();
    }

    CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
        if (!real_cuMemAlloc) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemAlloc(dptr, bytesize);
    }

    CUresult cuMemFree(CUdeviceptr dptr) {
        if (!real_cuMemFree) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemFree(dptr);
    }

    CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t byteCount) {
        if (!real_cuMemcpyDtoH) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemcpyDtoH(dstHost, srcDevice, byteCount);
    }

    CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t byteCount) {
        if (!real_cuMemcpyHtoD) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemcpyHtoD(dstDevice, srcHost, byteCount);
    }

    CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount) {
        if (!real_cuMemcpyDtoD) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemcpyDtoD(dstDevice, srcDevice, byteCount);
    }

    CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
        if (!real_cuMemAllocManaged) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemAllocManaged(dptr, bytesize, flags);
    }

    CUresult cuMemAllocHost(void** pp, size_t bytesize) {
        if (!real_cuMemAllocHost) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemAllocHost(pp, bytesize);
    }

    CUresult cuMemFreeHost(void* p) {
        if (!real_cuMemFreeHost) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemFreeHost(p);
    }

    CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
        if (!real_cuMemHostAlloc) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemHostAlloc(pp, bytesize, flags);
    }

    CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int flags) {
        if (!real_cuMemHostGetDevicePointer) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemHostGetDevicePointer(pdptr, p, flags);
    }

    CUresult cuMemGetInfo(size_t* free, size_t* total) {
        if (!real_cuMemGetInfo) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuMemGetInfo(free, total);
    }

    CUresult cuStreamCreate(CUstream* phStream, unsigned int flags) {
        if (!real_cuStreamCreate) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuStreamCreate(phStream, flags);
    }

    CUresult cuStreamDestroy(CUstream hStream) {
        if (!real_cuStreamDestroy) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuStreamDestroy(hStream);
    }

    CUresult cuStreamSynchronize(CUstream hStream) {
        if (!real_cuStreamSynchronize) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuStreamSynchronize(hStream);
    }

    CUresult cuStreamQuery(CUstream hStream) {
        if (!real_cuStreamQuery) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuStreamQuery(hStream);
    }

    CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
        if (!real_cuStreamWaitEvent) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuStreamWaitEvent(hStream, hEvent, flags);
    }

    CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
        if (!real_cuStreamGetFlags) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuStreamGetFlags(hStream, flags);
    }

    CUresult cuEventCreate(CUevent* phEvent, unsigned int flags) {
        if (!real_cuEventCreate) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuEventCreate(phEvent, flags);
    }

    CUresult cuEventDestroy(CUevent hEvent) {
        if (!real_cuEventDestroy) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuEventDestroy(hEvent);
    }

    CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
        if (!real_cuEventRecord) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuEventRecord(hEvent, hStream);
    }

    CUresult cuEventSynchronize(CUevent hEvent) {
        if (!real_cuEventSynchronize) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuEventSynchronize(hEvent);
    }

    CUresult cuEventQuery(CUevent hEvent) {
        if (!real_cuEventQuery) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuEventQuery(hEvent);
    }

    CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
        if (!real_cuEventElapsedTime) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuEventElapsedTime(pMilliseconds, hStart, hEnd);
    }

    CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                            unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                            unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
        if (!real_cuLaunchKernel) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                   sharedMemBytes, hStream, kernelParams, extra);
    }

    CUresult cuModuleLoad(CUmodule* module, const char* fname) {
        if (!real_cuModuleLoad) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuModuleLoad(module, fname);
    }

    CUresult cuModuleLoadData(CUmodule* module, const void* image) {
        if (!real_cuModuleLoadData) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuModuleLoadData(module, image);
    }

    CUresult cuModuleUnload(CUmodule hmod) {
        if (!real_cuModuleUnload) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuModuleUnload(hmod);
    }

    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
        if (!real_cuModuleGetFunction) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuModuleGetFunction(hfunc, hmod, name);
    }

    CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
        if (!real_cuDevicePrimaryCtxRetain) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuDevicePrimaryCtxRetain(pctx, dev);
    }

    CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
        if (!real_cuDevicePrimaryCtxRelease) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuDevicePrimaryCtxRelease(dev);
    }

    CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
        if (!real_cuDevicePrimaryCtxReset) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuDevicePrimaryCtxReset(dev);
    }

    CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, unsigned long long flags) {
        if (!real_cuGetProcAddress) return CUDA_ERROR_NOT_INITIALIZED;
        return real_cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    }

    // Get any function by name from real library
    void* getRealFunction(const char* name) {
        return RealCudaLoader::instance().get_cuda_driver_func(name);
    }

private:
    CudaDriverPassthrough() = default;

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
