#include "cudart_defs.hpp"
#include "cuda_driver_defs.hpp"
#include "kernel_executor.hpp"
#include "../core/global_state.hpp"
#include "../core/logging.hpp"
#include "../core/unsupported_api.hpp"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Thread-local error tracking. Assigning stores the status of the current call;
// only errors overwrite the value cudaGetLastError/cudaPeekAtLastError report
// (CUDA C++ Programming Guide 12.4, section 3.2.12 "Error Checking").
namespace {
struct LastError {
    cudaError_t result = cudaSuccess;
    cudaError_t recorded = cudaSuccess;

    LastError& operator=(cudaError_t error) {
        result = error;
        if (error != cudaSuccess) recorded = error;
        return *this;
    }
    operator cudaError_t() const { return result; }
};
}  // namespace

static thread_local LastError last_error;

namespace {

struct LoadedModule {
    CUmodule module = nullptr;
    CUresult status = CUDA_SUCCESS;
};

struct RegisteredModule {
    const void* image;
    std::unordered_map<int, LoadedModule> devices;
};

struct RegisteredEntry {
    RegisteredModule* owner;
    std::string name;
};

struct RuntimeRegistry {
    std::mutex mutex;
    std::unordered_map<const void*, RegisteredEntry> functions;
    std::unordered_map<const void*, RegisteredEntry> symbols;
};

RuntimeRegistry& runtimeRegistry() {
    static RuntimeRegistry registry;
    return registry;
}

// The caller holds the registry lock; CUDA globals belong to each device's module.
LoadedModule& currentModule(RegisteredModule& registered) {
    auto& state = fake_gpu::GlobalState::instance();
    state.initialize();
    auto [entry, inserted] = registered.devices.try_emplace(state.get_current_device());
    if (inserted)
        entry->second.status = cuModuleLoadData(&entry->second.module, registered.image);
    return entry->second;
}

CUresult lookupSymbol(const void* symbol, CUdeviceptr& pointer, size_t& size) {
    auto& registry = runtimeRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.symbols.find(symbol);
    if (found == registry.symbols.end()) return CUDA_ERROR_NOT_FOUND;
    const auto& entry = found->second;
    const auto& loaded = currentModule(*entry.owner);
    if (loaded.status != CUDA_SUCCESS) return loaded.status;
    return cuModuleGetGlobal(&pointer, &size, loaded.module, entry.name.c_str());
}

struct CaptureState {
    bool active = false;
    unsigned long long id = 0;
};

std::mutex g_capture_mutex;
std::unordered_map<cudaStream_t, CaptureState> g_capture_states;
std::atomic<unsigned long long> g_next_capture_id{1};

CaptureState lookupCaptureState(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    auto it = g_capture_states.find(stream);
    if (it == g_capture_states.end()) {
        return {};
    }
    return it->second;
}

bool beginCapture(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    CaptureState& state = g_capture_states[stream];
    if (state.active) {
        return false;
    }
    state.active = true;
    state.id = g_next_capture_id.fetch_add(1);
    return true;
}

bool endCapture(cudaStream_t stream, unsigned long long& capture_id) {
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    auto it = g_capture_states.find(stream);
    if (it == g_capture_states.end() || !it->second.active) {
        return false;
    }
    capture_id = it->second.id;
    it->second.active = false;
    it->second.id = 0;
    return true;
}

void clearCaptureState(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    g_capture_states.erase(stream);
}

}  // namespace

static cudaError_t setLastError(cudaError_t error) {
    last_error = error;
    return error;
}

// Helper: Convert CUresult to cudaError_t
static cudaError_t convertDriverError(CUresult result) {
    switch (result) {
        case CUDA_SUCCESS:
            return cudaSuccess;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return cudaErrorMemoryAllocation;
        case CUDA_ERROR_INVALID_DEVICE:
            return cudaErrorInvalidDevice;
        case CUDA_ERROR_NOT_INITIALIZED:
            return cudaErrorInitializationError;
        case CUDA_ERROR_NOT_READY:
            return cudaErrorNotReady;
        case CUDA_ERROR_NOT_SUPPORTED:
            return cudaErrorNotSupported;
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            return cudaErrorPeerAccessUnsupported;
        case CUDA_ERROR_INVALID_IMAGE:
            return cudaErrorInvalidKernelImage;
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return cudaErrorNoKernelImageForDevice;
        case CUDA_ERROR_INVALID_PTX:
            return cudaErrorInvalidPtx;
        case CUDA_ERROR_INVALID_HANDLE:
            return cudaErrorInvalidResourceHandle;
        case CUDA_ERROR_NOT_FOUND:
            return cudaErrorInvalidDeviceFunction;
        case CUDA_ERROR_ILLEGAL_ADDRESS:
            return cudaErrorIllegalAddress;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return cudaErrorLaunchOutOfResources;
        case CUDA_ERROR_LAUNCH_FAILED:
            return cudaErrorLaunchFailure;
        default:
            return cudaErrorUnknown;
    }
}

static cudaError_t convertSymbolError(CUresult result) {
    return result == CUDA_ERROR_NOT_FOUND ? cudaErrorInvalidSymbol : convertDriverError(result);
}

static bool validateStreamArgument(cudaStream_t stream) {
    if (!fake_gpu::cuda::is_stream_handle_live((CUstream)stream)) {
        last_error = cudaErrorInvalidValue;
        return false;
    }
    return true;
}

static bool rejectUnsupportedApi(
    const char* operation,
    const char* behavior = "not_executed") {
    if (fake_gpu::record_unsupported_api(operation, behavior)) {
        last_error = cudaErrorNotSupported;
        return true;
    }
    last_error = cudaSuccess;
    return false;
}

// Stores a malloc'ed opaque handle for simulated graphs, graph nodes, execs and pools.
template <class T>
static cudaError_t newPlaceholder(T* out) {
    if (!out) return setLastError(cudaErrorInvalidValue);
    *out = (T)malloc(sizeof(void*));
    return setLastError(cudaSuccess);
}

struct ErrorDescription {
    int code;
    const char* name;
    const char* text;
};

// Names and descriptions as returned by libcudart.so.12 (CUDA 12.4).
static const ErrorDescription& describeError(cudaError_t error) {
    static const ErrorDescription kErrors[] = {
        {0, "cudaSuccess", "no error"},
        {1, "cudaErrorInvalidValue", "invalid argument"},
        {2, "cudaErrorMemoryAllocation", "out of memory"},
        {3, "cudaErrorInitializationError", "initialization error"},
        {4, "cudaErrorCudartUnloading", "driver shutting down"},
        {5, "cudaErrorProfilerDisabled", "profiler disabled while using external profiling tool"},
        {6, "cudaErrorProfilerNotInitialized", "profiler not initialized: call cudaProfilerInitialize()"},
        {7, "cudaErrorProfilerAlreadyStarted", "profiler already started"},
        {8, "cudaErrorProfilerAlreadyStopped", "profiler already stopped"},
        {9, "cudaErrorInvalidConfiguration", "invalid configuration argument"},
        {12, "cudaErrorInvalidPitchValue", "invalid pitch argument"},
        {13, "cudaErrorInvalidSymbol", "invalid device symbol"},
        {16, "cudaErrorInvalidHostPointer", "invalid host pointer"},
        {17, "cudaErrorInvalidDevicePointer", "invalid device pointer"},
        {18, "cudaErrorInvalidTexture", "invalid texture reference"},
        {19, "cudaErrorInvalidTextureBinding", "texture is not bound to a pointer"},
        {20, "cudaErrorInvalidChannelDescriptor", "invalid channel descriptor"},
        {21, "cudaErrorInvalidMemcpyDirection", "invalid copy direction for memcpy"},
        {22, "cudaErrorAddressOfConstant", "invalid address of constant"},
        {23, "cudaErrorTextureFetchFailed", "fetch from texture failed"},
        {24, "cudaErrorTextureNotBound", "cannot fetch from a texture that is not bound"},
        {25, "cudaErrorSynchronizationError", "incorrect use of __syncthreads()"},
        {26, "cudaErrorInvalidFilterSetting", "linear filtering not supported for non-float type"},
        {27, "cudaErrorInvalidNormSetting", "read as normalized float not supported for 32-bit non float type"},
        {28, "cudaErrorMixedDeviceExecution", "device emulation mode and device execution mode cannot be mixed"},
        {31, "cudaErrorNotYetImplemented", "feature not yet implemented"},
        {32, "cudaErrorMemoryValueTooLarge", "memory size or pointer value too large to fit in 32 bit"},
        {34, "cudaErrorStubLibrary", "CUDA driver is a stub library"},
        {35, "cudaErrorInsufficientDriver", "CUDA driver version is insufficient for CUDA runtime version"},
        {36, "cudaErrorCallRequiresNewerDriver", "API call is not supported in the installed CUDA driver"},
        {37, "cudaErrorInvalidSurface", "invalid surface reference"},
        {43, "cudaErrorDuplicateVariableName", "duplicate global variable looked up by string name"},
        {44, "cudaErrorDuplicateTextureName", "duplicate texture looked up by string name"},
        {45, "cudaErrorDuplicateSurfaceName", "duplicate surface looked up by string name"},
        {46, "cudaErrorDevicesUnavailable", "CUDA-capable device(s) is/are busy or unavailable"},
        {49, "cudaErrorIncompatibleDriverContext", "incompatible driver context"},
        {52, "cudaErrorMissingConfiguration", "__global__ function call is not configured"},
        {53, "cudaErrorPriorLaunchFailure", "unspecified launch failure in prior launch"},
        {65, "cudaErrorLaunchMaxDepthExceeded", "launch would exceed maximum depth of nested launches"},
        {66, "cudaErrorLaunchFileScopedTex", "launch failed because kernel uses unsupported, file-scoped textures (texture objects are supported)"},
        {67, "cudaErrorLaunchFileScopedSurf", "launch failed because kernel uses unsupported, file-scoped surfaces (surface objects are supported)"},
        {68, "cudaErrorSyncDepthExceeded", "cudaDeviceSynchronize failed because caller's grid depth exceeds cudaLimitDevRuntimeSyncDepth"},
        {69, "cudaErrorLaunchPendingCountExceeded", "launch failed because launch would exceed cudaLimitDevRuntimePendingLaunchCount"},
        {98, "cudaErrorInvalidDeviceFunction", "invalid device function"},
        {100, "cudaErrorNoDevice", "no CUDA-capable device is detected"},
        {101, "cudaErrorInvalidDevice", "invalid device ordinal"},
        {102, "cudaErrorDeviceNotLicensed", "device doesn't have valid Grid license"},
        {103, "cudaErrorSoftwareValidityNotEstablished", "integrity checks failed"},
        {127, "cudaErrorStartupFailure", "startup failure in cuda runtime"},
        {200, "cudaErrorInvalidKernelImage", "device kernel image is invalid"},
        {201, "cudaErrorDeviceUninitialized", "invalid device context"},
        {205, "cudaErrorMapBufferObjectFailed", "mapping of buffer object failed"},
        {206, "cudaErrorUnmapBufferObjectFailed", "unmapping of buffer object failed"},
        {207, "cudaErrorArrayIsMapped", "array is mapped"},
        {208, "cudaErrorAlreadyMapped", "resource already mapped"},
        {209, "cudaErrorNoKernelImageForDevice", "no kernel image is available for execution on the device"},
        {210, "cudaErrorAlreadyAcquired", "resource already acquired"},
        {211, "cudaErrorNotMapped", "resource not mapped"},
        {212, "cudaErrorNotMappedAsArray", "resource not mapped as array"},
        {213, "cudaErrorNotMappedAsPointer", "resource not mapped as pointer"},
        {214, "cudaErrorECCUncorrectable", "uncorrectable ECC error encountered"},
        {215, "cudaErrorUnsupportedLimit", "limit is not supported on this architecture"},
        {216, "cudaErrorDeviceAlreadyInUse", "exclusive-thread device already in use by a different thread"},
        {217, "cudaErrorPeerAccessUnsupported", "peer access is not supported between these two devices"},
        {218, "cudaErrorInvalidPtx", "a PTX JIT compilation failed"},
        {219, "cudaErrorInvalidGraphicsContext", "invalid OpenGL or DirectX context"},
        {220, "cudaErrorNvlinkUncorrectable", "uncorrectable NVLink error detected during the execution"},
        {221, "cudaErrorJitCompilerNotFound", "PTX JIT compiler library not found"},
        {222, "cudaErrorUnsupportedPtxVersion", "the provided PTX was compiled with an unsupported toolchain."},
        {223, "cudaErrorJitCompilationDisabled", "PTX JIT compilation was disabled"},
        {224, "cudaErrorUnsupportedExecAffinity", "the provided execution affinity is not supported"},
        {225, "cudaErrorUnsupportedDevSideSync", "the provided PTX contains unsupported call to cudaDeviceSynchronize"},
        {300, "cudaErrorInvalidSource", "device kernel image is invalid"},
        {301, "cudaErrorFileNotFound", "file not found"},
        {302, "cudaErrorSharedObjectSymbolNotFound", "shared object symbol not found"},
        {303, "cudaErrorSharedObjectInitFailed", "shared object initialization failed"},
        {304, "cudaErrorOperatingSystem", "OS call failed or operation not supported on this OS"},
        {400, "cudaErrorInvalidResourceHandle", "invalid resource handle"},
        {401, "cudaErrorIllegalState", "the operation cannot be performed in the present state"},
        {402, "cudaErrorLossyQuery", "attempted introspection would be semantically lossy"},
        {500, "cudaErrorSymbolNotFound", "named symbol not found"},
        {600, "cudaErrorNotReady", "device not ready"},
        {700, "cudaErrorIllegalAddress", "an illegal memory access was encountered"},
        {701, "cudaErrorLaunchOutOfResources", "too many resources requested for launch"},
        {702, "cudaErrorLaunchTimeout", "the launch timed out and was terminated"},
        {703, "cudaErrorLaunchIncompatibleTexturing", "launch uses incompatible texturing mode"},
        {704, "cudaErrorPeerAccessAlreadyEnabled", "peer access is already enabled"},
        {705, "cudaErrorPeerAccessNotEnabled", "peer access has not been enabled"},
        {708, "cudaErrorSetOnActiveProcess", "cannot set while device is active in this process"},
        {709, "cudaErrorContextIsDestroyed", "context is destroyed"},
        {710, "cudaErrorAssert", "device-side assert triggered"},
        {711, "cudaErrorTooManyPeers", "peer mapping resources exhausted"},
        {712, "cudaErrorHostMemoryAlreadyRegistered", "part or all of the requested memory range is already mapped"},
        {713, "cudaErrorHostMemoryNotRegistered", "pointer does not correspond to a registered memory region"},
        {714, "cudaErrorHardwareStackError", "hardware stack error"},
        {715, "cudaErrorIllegalInstruction", "an illegal instruction was encountered"},
        {716, "cudaErrorMisalignedAddress", "misaligned address"},
        {717, "cudaErrorInvalidAddressSpace", "operation not supported on global/shared address space"},
        {718, "cudaErrorInvalidPc", "invalid program counter"},
        {719, "cudaErrorLaunchFailure", "unspecified launch failure"},
        {720, "cudaErrorCooperativeLaunchTooLarge", "too many blocks in cooperative launch"},
        {800, "cudaErrorNotPermitted", "operation not permitted"},
        {801, "cudaErrorNotSupported", "operation not supported"},
        {802, "cudaErrorSystemNotReady", "system not yet initialized"},
        {803, "cudaErrorSystemDriverMismatch", "system has unsupported display driver / cuda driver combination"},
        {804, "cudaErrorCompatNotSupportedOnDevice", "forward compatibility was attempted on non supported HW"},
        {805, "cudaErrorMpsConnectionFailed", "MPS client failed to connect to the MPS control daemon or the MPS server"},
        {806, "cudaErrorMpsRpcFailure", "the remote procedural call between the MPS server and the MPS client failed"},
        {807, "cudaErrorMpsServerNotReady", "MPS server is not ready to accept new MPS client requests"},
        {808, "cudaErrorMpsMaxClientsReached", "the hardware resources required to create MPS client have been exhausted"},
        {809, "cudaErrorMpsMaxConnectionsReached", "the hardware resources required to support device connections have been exhausted"},
        {810, "cudaErrorMpsClientTerminated", "the MPS client has been terminated by the server"},
        {811, "cudaErrorCdpNotSupported", "is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it"},
        {812, "cudaErrorCdpVersionMismatch", "unsupported interaction between different versions of CUDA Dynamic Parallelism"},
        {900, "cudaErrorStreamCaptureUnsupported", "operation not permitted when stream is capturing"},
        {901, "cudaErrorStreamCaptureInvalidated", "operation failed due to a previous error during capture"},
        {902, "cudaErrorStreamCaptureMerge", "operation would result in a merge of separate capture sequences"},
        {903, "cudaErrorStreamCaptureUnmatched", "capture was not ended in the same stream as it began"},
        {904, "cudaErrorStreamCaptureUnjoined", "capturing stream has unjoined work"},
        {905, "cudaErrorStreamCaptureIsolation", "dependency created on uncaptured work in another stream"},
        {906, "cudaErrorStreamCaptureImplicit", "operation would make the legacy stream depend on a capturing blocking stream"},
        {907, "cudaErrorCapturedEvent", "operation not permitted on an event last recorded in a capturing stream"},
        {908, "cudaErrorStreamCaptureWrongThread", "attempt to terminate a thread-local capture sequence from another thread"},
        {909, "cudaErrorTimeout", "wait operation timed out"},
        {910, "cudaErrorGraphExecUpdateFailure", "the graph update was not performed because it included changes which violated constraints specific to instantiated graph update"},
        {911, "cudaErrorExternalDevice", "an async error has occured in external entity outside of CUDA"},
        {912, "cudaErrorInvalidClusterSize", "a kernel launch error has occurred due to cluster misconfiguration"},
        {999, "cudaErrorUnknown", "unknown error"},
        {10000, "cudaErrorApiFailureBase", "api failure base"},
    };
    // cuda_runtime_api.h: both functions return this text for unknown codes.
    static const ErrorDescription kUnrecognized = {-1, "unrecognized error code", "unrecognized error code"};
    for (const ErrorDescription& entry : kErrors) {
        if (entry.code == static_cast<int>(error)) return entry;
    }
    return kUnrecognized;
}

static cudaError_t allocDevice(void** out, size_t size) {
    CUdeviceptr dptr;
    const CUresult result = cuMemAlloc(&dptr, size);
    if (result == CUDA_SUCCESS) *out = (void*)dptr;
    return setLastError(convertDriverError(result));
}

static cudaError_t allocManaged(void** out, size_t size, unsigned int flags) {
    CUdeviceptr dptr;
    const CUresult result = cuMemAllocManaged(&dptr, size, flags);
    if (result == CUDA_SUCCESS) *out = (void*)dptr;
    return setLastError(convertDriverError(result));
}

// Pinned host memory is simulated with regular malloc.
static cudaError_t allocPinnedHost(void** out, size_t size) {
    *out = malloc(size);
    if (!*out) return setLastError(cudaErrorMemoryAllocation);
    fake_gpu::GlobalState::instance().register_host_allocation(
        *out, size, fake_gpu::GlobalState::instance().get_current_device());
    return setLastError(cudaSuccess);
}

// Performs a synchronous copy in the given direction and sets last_error.
// Returns false only for an invalid direction.
static bool copyByKind(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    CUresult result = CUDA_SUCCESS;
    switch (kind) {
        case cudaMemcpyHostToDevice:
            result = cuMemcpyHtoD((CUdeviceptr)dst, src, count);
            break;
        case cudaMemcpyDeviceToHost:
            result = cuMemcpyDtoH(dst, (CUdeviceptr)src, count);
            break;
        case cudaMemcpyDeviceToDevice:
            result = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, count);
            break;
        case cudaMemcpyDefault:
            result = cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count);
            break;
        case cudaMemcpyHostToHost:
            memcpy(dst, src, count);
            fake_gpu::GlobalState::instance().record_memcpy_h2h(count);
            break;
        default:
            last_error = cudaErrorInvalidMemcpyDirection;
            return false;
    }
    last_error = convertDriverError(result);
    return true;
}

static cudaError_t rejectKernelApi(const char* operation) {
    auto& state = fake_gpu::GlobalState::instance();
    state.initialize();
    state.record_unsupported_api(state.get_current_device(), operation, "not_executed", "error");
    return setLastError(cudaErrorNotSupported);
}

static bool submitAsyncStreamOperation(cudaStream_t stream) {
    const CUresult result = fake_gpu::cuda::submit_stream_operation((CUstream)stream);
    if (result != CUDA_SUCCESS) {
        last_error = convertDriverError(result);
        return false;
    }
    return true;
}

template <typename Properties>
static cudaError_t getDeviceProperties(Properties *prop, int device) {
    if (!prop) return setLastError(cudaErrorInvalidValue);

    // Zero out the structure
    memset(prop, 0, sizeof(Properties));

    // Get device name
    char name[256];
    CUresult result = cuDeviceGetName(name, 256, device);
    if (result != CUDA_SUCCESS) return setLastError(convertDriverError(result));
    strncpy(prop->name, name, 256);

    // Get total memory
    size_t totalMem;
    result = cuDeviceTotalMem(&totalMem, device);
    if (result == CUDA_SUCCESS) {
        prop->totalGlobalMem = totalMem;
    }

    // Get compute capability
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    prop->major = major;
    prop->minor = minor;

    // Get other attributes
    cuDeviceGetAttribute(&prop->multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    cuDeviceGetAttribute(&prop->maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    cuDeviceGetAttribute(&prop->maxThreadsDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
    cuDeviceGetAttribute(&prop->maxThreadsDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device);
    cuDeviceGetAttribute(&prop->maxThreadsDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device);
    cuDeviceGetAttribute(&prop->maxGridSize[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
    cuDeviceGetAttribute(&prop->maxGridSize[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
    cuDeviceGetAttribute(&prop->maxGridSize[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);
    cuDeviceGetAttribute(&prop->maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
    cuDeviceGetAttribute(&prop->warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
    cuDeviceGetAttribute(&prop->clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
    cuDeviceGetAttribute(&prop->memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
    cuDeviceGetAttribute(&prop->memoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
    cuDeviceGetAttribute(&prop->l2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device);
    cuDeviceGetAttribute((int*)&prop->sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);
    cuDeviceGetAttribute(&prop->regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device);
    cuDeviceGetAttribute(&prop->pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device);
    cuDeviceGetAttribute(&prop->pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);
    cuDeviceGetAttribute(&prop->pciDomainID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device);

    // The remaining fields mirror the cudaDevAttr values of the same name
    // (cuda_runtime_api.h, cudaGetDeviceProperties and cudaDeviceGetAttribute).
    auto attribute = [device](CUdevice_attribute attrib) {
        int value = 0;
        cuDeviceGetAttribute(&value, attrib, device);
        return value;
    };
    CUuuid uuid{};
    if (cuDeviceGetUuid(&uuid, device) == CUDA_SUCCESS) memcpy(prop->uuid, uuid.bytes, sizeof(prop->uuid));
    prop->totalConstMem = attribute(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
    prop->memPitch = attribute(CU_DEVICE_ATTRIBUTE_MAX_PITCH);
    prop->textureAlignment = attribute(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
    prop->deviceOverlap = attribute(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP);
    prop->kernelExecTimeoutEnabled = attribute(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT);
    prop->integrated = attribute(CU_DEVICE_ATTRIBUTE_INTEGRATED);
    prop->canMapHostMemory = attribute(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
    prop->computeMode = attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
    prop->concurrentKernels = attribute(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS);
    prop->ECCEnabled = attribute(CU_DEVICE_ATTRIBUTE_ECC_ENABLED);
    prop->tccDriver = attribute(CU_DEVICE_ATTRIBUTE_TCC_DRIVER);
    prop->asyncEngineCount = attribute(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
    prop->unifiedAddressing = attribute(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
    prop->persistingL2CacheMaxSize = attribute(CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE);
    prop->streamPrioritiesSupported = attribute(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED);
    prop->globalL1CacheSupported = attribute(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED);
    prop->localL1CacheSupported = attribute(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED);
    prop->sharedMemPerMultiprocessor = attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    prop->regsPerMultiprocessor = attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
    prop->managedMemory = attribute(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
    prop->isMultiGpuBoard = attribute(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD);
    prop->multiGpuBoardGroupID = attribute(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID);
    prop->hostNativeAtomicSupported = attribute(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED);
    prop->singleToDoublePrecisionPerfRatio = attribute(CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO);
    prop->pageableMemoryAccess = attribute(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS);
    prop->concurrentManagedAccess = attribute(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS);
    prop->computePreemptionSupported = attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED);
    prop->canUseHostPointerForRegisteredMem = attribute(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM);
    prop->cooperativeLaunch = attribute(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH);
    prop->cooperativeMultiDeviceLaunch = attribute(CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH);
    prop->sharedMemPerBlockOptin = attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    prop->pageableMemoryAccessUsesHostPageTables =
        attribute(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES);
    prop->directManagedMemAccessFromHost = attribute(CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST);
    prop->maxBlocksPerMultiProcessor = attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR);
    prop->accessPolicyMaxWindowSize = attribute(CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE);
    prop->reservedSharedMemPerBlock = attribute(CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK);

    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGetDeviceProperties(%d)\n", device);
    return last_error;
}

extern "C" {

// ============================================================================
// Device Management
// ============================================================================

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuDeviceGetCount(count);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaGetDeviceCount returning %d\n", *count);
    return last_error;
}

cudaError_t cudaSetDevice(int device) {
    // Get device handle
    CUdevice cuDevice;
    CUresult result = cuDeviceGet(&cuDevice, device);
    if (result != CUDA_SUCCESS) return setLastError(convertDriverError(result));

    // Retain primary context
    CUcontext context;
    result = cuDevicePrimaryCtxRetain(&context, cuDevice);
    if (result != CUDA_SUCCESS) return setLastError(convertDriverError(result));

    // Set as current context
    result = cuCtxSetCurrent(context);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaSetDevice(%d)\n", device);
    return last_error;
}

cudaError_t cudaGetDevice(int *device) {
    if (!device) return setLastError(cudaErrorInvalidValue);

    // Get current context
    CUcontext context;
    CUresult result = cuCtxGetCurrent(&context);
    if (result != CUDA_SUCCESS) return setLastError(convertDriverError(result));

    // Get device from context
    CUdevice cuDevice;
    result = cuCtxGetDevice(&cuDevice);
    if (result != CUDA_SUCCESS) return setLastError(convertDriverError(result));

    *device = cuDevice;
    return setLastError(cudaSuccess);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    return getDeviceProperties(prop, device);
}

cudaError_t cudaGetDeviceProperties_v2(cudaDevicePropV2 *prop, int device) {
    const cudaError_t result = getDeviceProperties(prop, device);
    if (result != cudaSuccess) return result;
    auto attribute = [device](CUdevice_attribute attrib) {
        int value = 0;
        cuDeviceGetAttribute(&value, attrib, device);
        return value;
    };
    prop->hostRegisterSupported = attribute(CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED);
    prop->sparseCudaArraySupported = attribute(CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED);
    prop->hostRegisterReadOnlySupported = attribute(CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED);
    prop->timelineSemaphoreInteropSupported = attribute(CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED);
    prop->memoryPoolsSupported = attribute(CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED);
    prop->gpuDirectRDMASupported = attribute(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED);
    prop->gpuDirectRDMAFlushWritesOptions = attribute(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS);
    prop->gpuDirectRDMAWritesOrdering = attribute(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING);
    prop->memoryPoolSupportedHandleTypes = attribute(CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES);
    prop->deferredMappingCudaArraySupported = attribute(CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED);
    prop->ipcEventSupported = attribute(CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED);
    prop->clusterLaunch = attribute(CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH);
    prop->unifiedFunctionPointers = attribute(CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS);
    return cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
    CUcontext context = nullptr;
    CUresult result = cuCtxGetCurrent(&context);
    if (result == CUDA_SUCCESS && context) {
        CUdevice device;
        result = cuCtxGetDevice(&device);
        if (result == CUDA_SUCCESS) result = cuDevicePrimaryCtxReset(device);
    }
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaDeviceReset()\n");
    return last_error;
}

cudaError_t cudaDeviceSynchronize(void) {
    CUresult result = cuCtxSynchronize();
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaDeviceSynchronize()\n");
    return last_error;
}

// ============================================================================
// Memory Management
// ============================================================================

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) return setLastError(cudaErrorInvalidValue);

    if (allocDevice(devPtr, size) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaMalloc allocated %zu bytes at %p\n", size, *devPtr);
    }
    return last_error;
}

cudaError_t cudaFree(void *devPtr) {
    if (!devPtr) return setLastError(cudaSuccess);

    CUresult result = cuMemFree((CUdeviceptr)devPtr);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaFree(%p)\n", devPtr);
    return last_error;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    if (!devPtr) return setLastError(cudaErrorInvalidValue);
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    if (allocDevice(devPtr, size) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaMallocAsync allocated %zu bytes at %p\n", size, *devPtr);
        submitAsyncStreamOperation(stream);
    }
    return last_error;
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }
    if (!devPtr) return setLastError(cudaSuccess);

    CUresult result = cuMemFree((CUdeviceptr)devPtr);
    last_error = convertDriverError(result);
    if (last_error == cudaSuccess && !submitAsyncStreamOperation(stream)) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaFreeAsync(%p)\n", devPtr);
    return last_error;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    if (!copyByKind(dst, src, count, kind)) return last_error;
    FGPU_LOG("[FakeCUDART] cudaMemcpy count=%zu kind=%d\n", count, kind);
    return last_error;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    // Simplified: execute synchronously after validating the stream handle.
    if (copyByKind(dst, src, count, kind) && last_error == cudaSuccess) {
        submitAsyncStreamOperation(stream);
    }
    return last_error;
}

cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) {
    last_error = convertDriverError(static_cast<CUresult>(fake_gpu::kernel::copy(dst, src, count)));
    if (last_error != cudaSuccess) return last_error;
    fake_gpu::GlobalState::instance().record_memcpy_peer(dstDevice, srcDevice, count);
    FGPU_LOG("[FakeCUDART] cudaMemcpyPeer count=%zu from device %d to device %d\n", count, srcDevice, dstDevice);
    return last_error;
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    last_error = convertDriverError(static_cast<CUresult>(fake_gpu::kernel::copy(dst, src, count)));
    if (last_error != cudaSuccess) return last_error;
    fake_gpu::GlobalState::instance().record_memcpy_peer(dstDevice, srcDevice, count);
    if (!submitAsyncStreamOperation(stream)) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaMemcpyPeerAsync count=%zu from device %d to device %d\n", count, srcDevice, dstDevice);
    return last_error;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    last_error = convertDriverError(cuMemsetD8((CUdeviceptr)devPtr, static_cast<unsigned char>(value), count));
    FGPU_LOG("[FakeCUDART] cudaMemset ptr=%p value=%d count=%zu\n", devPtr, value, count);
    return last_error;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    last_error = convertDriverError(cuMemsetD8((CUdeviceptr)devPtr, static_cast<unsigned char>(value), count));
    if (last_error == cudaSuccess) submitAsyncStreamOperation(stream);
    return last_error;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    CUresult result = cuMemGetInfo(free, total);
    return setLastError(convertDriverError(result));
}

// ============================================================================
// Stream Management
// ============================================================================

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    if (!pStream) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuStreamCreate((CUstream*)pStream, 0);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaStreamCreate\n");
    return last_error;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (lookupCaptureState(stream).active) return setLastError(cudaErrorInvalidValue);
    CUresult result = cuStreamDestroy((CUstream)stream);
    last_error = convertDriverError(result);
    if (last_error == cudaSuccess) {
        clearCaptureState(stream);
    }
    FGPU_LOG("[FakeCUDART] cudaStreamDestroy\n");
    return last_error;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    CUresult result = cuStreamSynchronize((CUstream)stream);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaStreamSynchronize\n");
    return last_error;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    CUresult result = cuStreamQuery((CUstream)stream);
    return setLastError(convertDriverError(result));
}

// ============================================================================
// Event Management
// ============================================================================

cudaError_t cudaEventCreate(cudaEvent_t *event) {
    if (!event) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuEventCreate((CUevent*)event, 0);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaEventCreate\n");
    return last_error;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    CUresult result = cuEventDestroy((CUevent)event);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaEventDestroy\n");
    return last_error;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    CUresult result = cuEventRecord((CUevent)event, (CUstream)stream);
    return setLastError(convertDriverError(result));
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    CUresult result = cuEventSynchronize((CUevent)event);
    return setLastError(convertDriverError(result));
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    CUresult result = cuEventQuery((CUevent)event);
    return setLastError(convertDriverError(result));
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuEventElapsedTime(ms, (CUevent)start, (CUevent)end);
    return setLastError(convertDriverError(result));
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
    CUresult result = cuEventRecord((CUevent)event, (CUstream)stream);
    return setLastError(convertDriverError(result));
}

// ============================================================================
// Error Handling
// ============================================================================

cudaError_t cudaGetLastError(void) {
    const cudaError_t err = last_error.recorded;
    last_error.recorded = cudaSuccess;  // Clear error
    FGPU_LOG("[FakeCUDART] cudaGetLastError returning %d\n", err);
    return err;
}

cudaError_t cudaPeekAtLastError(void) {
    FGPU_LOG("[FakeCUDART] cudaPeekAtLastError returning %d\n", last_error.recorded);
    return last_error.recorded;
}

const char* cudaGetErrorString(cudaError_t error) {
    return describeError(error).text;
}

const char* cudaGetErrorName(cudaError_t error) {
    return describeError(error).name;
}

// ============================================================================
// Version Management
// ============================================================================

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion) return setLastError(cudaErrorInvalidValue);

    *runtimeVersion = 12000;  // CUDA 12.0
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaRuntimeGetVersion returning 12000\n");
    return last_error;
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuDriverGetVersion(driverVersion);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaDriverGetVersion returning %d\n", *driverVersion);
    return last_error;
}

// ============================================================================
// Kernel Launch
// ============================================================================

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }
    if (lookupCaptureState(stream).active) return rejectKernelApi("cudaLaunchKernel.capture");
    auto& registry = runtimeRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.functions.find(func);
    if (found == registry.functions.end()) return last_error = cudaErrorInvalidDeviceFunction;
    const auto& entry = found->second;
    const auto& loaded = currentModule(*entry.owner);
    if (loaded.status != CUDA_SUCCESS) return last_error = convertDriverError(loaded.status);
    CUfunction function = nullptr;
    const CUresult resolved = cuModuleGetFunction(&function, loaded.module, entry.name.c_str());
    if (resolved != CUDA_SUCCESS) return last_error = convertDriverError(resolved);
    if (sharedMem > std::numeric_limits<unsigned>::max()) return last_error = cudaErrorInvalidValue;
    last_error = convertDriverError(cuLaunchKernel(
        function, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
        static_cast<unsigned>(sharedMem), stream, args, nullptr));
    return last_error;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaConfigureCall (stub)\n");
    if (rejectUnsupportedApi(
            "cudaConfigureCall", "launch_configuration_not_materialized")) {
        return last_error;
    }
    submitAsyncStreamOperation(stream);
    return last_error;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    rejectUnsupportedApi("cudaSetupArgument", "launch_argument_not_materialized");
    return last_error;
}

cudaError_t cudaLaunch(const void *func) {
    return rejectKernelApi("cudaLaunch");
}

cudaError_t cudaLaunchKernelExC(const void *config, const void *func, void **args) {
    struct LaunchConfig {
        dim3 grid;
        dim3 block;
        size_t shared_bytes;
        cudaStream_t stream;
        const void* attributes;
        unsigned num_attributes;
    };
    if (!config) return last_error = cudaErrorInvalidValue;
    const auto& launch = *static_cast<const LaunchConfig*>(config);
    if (launch.num_attributes) return rejectKernelApi("cudaLaunchKernelExC.attributes");
    return cudaLaunchKernel(func, launch.grid, launch.block, args, launch.shared_bytes, launch.stream);
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void *userData), void *userData) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    // Simplified: just execute the function directly after validating the stream handle.
    if (fn) {
        fn(userData);
    }
    last_error = cudaSuccess;
    if (!submitAsyncStreamOperation(stream)) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaLaunchHostFunc (executed host function)\n");
    return last_error;
}

// ============================================================================
// Host Memory
// ============================================================================

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    if (!ptr) return setLastError(cudaErrorInvalidValue);

    if (allocPinnedHost(ptr, size) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaMallocHost allocated %zu bytes\n", size);
    }
    return last_error;
}

cudaError_t cudaFreeHost(void *ptr) {
    size_t size = 0;
    int device = 0;
    fake_gpu::GlobalState::instance().release_host_allocation(ptr, size, device);
    free(ptr);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaFreeHost\n");
    return last_error;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    if (!pHost) return setLastError(cudaErrorInvalidValue);

    if (allocPinnedHost(pHost, size) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaHostAlloc allocated %zu bytes\n", size);
    }
    return last_error;
}

cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaHostRegister ptr=%p size=%zu flags=%u\n", ptr, size, flags);
    return last_error;
}

cudaError_t cudaHostUnregister(void *ptr) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaHostUnregister ptr=%p\n", ptr);
    return last_error;
}

// ============================================================================
// Peer Access
// ============================================================================

// Peer access follows the modeled NVLink/PCIe topology (FAKEGPU_SLOTS,
// FAKEGPU_NVLINK_FABRIC); without one every pair of devices has access.
static bool modeled_peer_access(int device, int peerDevice, cudaError_t& error) {
    fake_gpu::GlobalState& state = fake_gpu::GlobalState::instance();
    state.initialize();
    const int count = state.get_device_count();
    if (device < 0 || device >= count || peerDevice < 0 || peerDevice >= count) {
        error = cudaErrorInvalidDevice;
        return false;
    }
    error = cudaSuccess;
    return state.snapshot_device_topology().can_access_peer(device, peerDevice);
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
    if (!canAccessPeer) return setLastError(cudaErrorInvalidValue);
    cudaError_t error = cudaSuccess;
    const bool allowed = modeled_peer_access(device, peerDevice, error);
    if (error != cudaSuccess) return setLastError(error);
    *canAccessPeer = allowed ? 1 : 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    (void)flags;
    cudaError_t error = cudaSuccess;
    const int device = fake_gpu::GlobalState::instance().get_current_device();
    const bool allowed = modeled_peer_access(device, peerDevice, error);
    last_error = error != cudaSuccess ? error : allowed ? cudaSuccess : cudaErrorPeerAccessUnsupported;
    FGPU_LOG("[FakeCUDART] cudaDeviceEnablePeerAccess(%d -> %d) = %d\n", device, peerDevice, last_error.result);
    return last_error;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    // Simplified: just succeed
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDeviceDisablePeerAccess(%d)\n", peerDevice);
    return last_error;
}

// ============================================================================
// Driver Entry Point (CUDA 12+)
// ============================================================================

cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr, unsigned int cudaVersion, unsigned long long flags, int *driverStatus) {
    if (!symbol || !funcPtr) return setLastError(cudaErrorInvalidValue);

    // Use cuGetProcAddress to get the driver function
    CUresult result = cuGetProcAddress(symbol, funcPtr, cudaVersion, flags);

    if (driverStatus) {
        *driverStatus = (result == CUDA_SUCCESS && *funcPtr) ? 0 : 1;
    }

    return setLastError(convertDriverError(result));
}

// CUDA 12 added the optional driverStatus argument (cuda_runtime_api.h).
cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, int *driverStatus) {
    return cudaGetDriverEntryPointByVersion(symbol, funcPtr, 12000, flags, driverStatus);
}

// ============================================================================
// Additional Stream Management
// ============================================================================

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
    if (!pStream) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuStreamCreate((CUstream*)pStream, flags);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaStreamCreateWithFlags flags=%u\n", flags);
    return last_error;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) {
    if (!pStream) return setLastError(cudaErrorInvalidValue);

    // Simplified: ignore priority
    CUresult result = cuStreamCreate((CUstream*)pStream, flags);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaStreamCreateWithPriority flags=%u priority=%d\n", flags, priority);
    return last_error;
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {
    if (!flags) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuStreamGetFlags((CUstream)hStream, flags);
    return setLastError(convertDriverError(result));
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority) {
    if (!priority) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuStreamGetPriority((CUstream)hStream, priority);
    return setLastError(convertDriverError(result));
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    CUresult result = cuStreamWaitEvent((CUstream)stream, (CUevent)event, flags);
    return setLastError(convertDriverError(result));
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, void (*callback)(cudaStream_t stream, cudaError_t status, void *userData), void *userData, unsigned int flags) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    if (rejectUnsupportedApi("cudaStreamAddCallback", "callback_not_registered")) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaStreamAddCallback (stub)\n");
    return last_error;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    if (rejectUnsupportedApi(
            "cudaStreamUpdateCaptureDependencies",
            "capture_dependencies_not_materialized")) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaStreamUpdateCaptureDependencies (stub)\n");
    return last_error;
}

// ============================================================================
// Additional Event Management
// ============================================================================

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    if (!event) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuEventCreate((CUevent*)event, flags);
    last_error = convertDriverError(result);
    FGPU_LOG("[FakeCUDART] cudaEventCreateWithFlags flags=%u\n", flags);
    return last_error;
}

// ============================================================================
// Occupancy Calculation
// ============================================================================

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
    if (!numBlocks) return setLastError(cudaErrorInvalidValue);

    // Simplified: return a reasonable fake value
    *numBlocks = 16;
    return setLastError(cudaSuccess);
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

// ============================================================================
// Device Attributes
// ============================================================================

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
    if (!value) return setLastError(cudaErrorInvalidValue);

    CUresult result = cuDeviceGetAttribute(value, (CUdevice_attribute)attr, device);
    return setLastError(convertDriverError(result));
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, int limit) {
    if (!pValue) return setLastError(cudaErrorInvalidValue);

    // Simplified: return reasonable defaults
    *pValue = 1024 * 1024 * 1024;  // 1GB
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceSetLimit(int limit, size_t value) {
    // Simplified: just succeed
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceGetCacheConfig(int *pCacheConfig) {
    if (!pCacheConfig) return setLastError(cudaErrorInvalidValue);

    *pCacheConfig = 0;  // cudaFuncCachePreferNone
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceSetCacheConfig(int cacheConfig) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceGetSharedMemConfig(int *pConfig) {
    if (!pConfig) return setLastError(cudaErrorInvalidValue);

    *pConfig = 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceSetSharedMemConfig(int config) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    if (!leastPriority || !greatestPriority) return setLastError(cudaErrorInvalidValue);

    *leastPriority = 0;
    *greatestPriority = 0;
    return setLastError(cudaSuccess);
}

// ============================================================================
// Additional Memory Management
// ============================================================================

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
    if (!devPtr || !pitch) return setLastError(cudaErrorInvalidValue);

    // Align pitch to 512 bytes
    *pitch = (width + 511) & ~511;
    size_t size = (*pitch) * height;

    if (allocDevice(devPtr, size) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaMallocPitch allocated %zu bytes (pitch=%zu)\n", size, *pitch);
    }
    return last_error;
}

cudaError_t cudaMalloc3D(void **devPtr, size_t width, size_t height, size_t depth) {
    if (!devPtr) return setLastError(cudaErrorInvalidValue);

    size_t size = width * height * depth;
    if (allocDevice(devPtr, size) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaMalloc3D allocated %zu bytes\n", size);
    }
    return last_error;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    if (!devPtr) return setLastError(cudaErrorInvalidValue);

    if (allocManaged(devPtr, size, flags) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaMallocManaged allocated %zu bytes\n", size);
    }
    return last_error;
}

cudaError_t cudaMallocArray(void **array, const void *desc, size_t width, size_t height, unsigned int flags) {
    if (!array) return setLastError(cudaErrorInvalidValue);

    // Simplified: allocate linear memory
    size_t size = width * height * 4;  // Assume 4 bytes per element
    *array = malloc(size);
    if (!*array) return setLastError(cudaErrorMemoryAllocation);

    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaMallocArray allocated %zu bytes\n", size);
    return last_error;
}

cudaError_t cudaFreeArray(void *array) {
    free(array);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaFreeArray\n");
    return last_error;
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
    // Simplified: copy row by row
    for (size_t i = 0; i < height; i++) {
        const char *srcRow = (const char*)src + i * spitch;
        char *dstRow = (char*)dst + i * dpitch;
        const auto result = cudaMemcpy(dstRow, srcRow, width, kind);
        if (result != cudaSuccess) return result;
    }

    return setLastError(cudaSuccess);
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    // Simplified: do synchronous copy after validating the stream handle.
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
    submitAsyncStreamOperation(stream);
    return last_error;
}

cudaError_t cudaMemcpy3D(const void *p) {
    // Simplified: just succeed
    return setLastError(cudaSuccess);
}

cudaError_t cudaMemcpy3DAsync(const void *p, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    // Simplified: just succeed after validating the stream handle.
    last_error = cudaSuccess;
    submitAsyncStreamOperation(stream);
    return last_error;
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind) {
    if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToDevice && kind != cudaMemcpyDefault)
        return last_error = cudaErrorInvalidMemcpyDirection;
    CUdeviceptr pointer = 0;
    size_t size = 0;
    last_error = convertSymbolError(lookupSymbol(symbol, pointer, size));
    if (last_error != cudaSuccess) return last_error;
    if (offset > size || count > size - offset) return last_error = cudaErrorInvalidValue;
    return cudaMemcpy(reinterpret_cast<void*>(pointer + offset), src, count, kind);
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, cudaMemcpyKind kind) {
    if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice && kind != cudaMemcpyDefault)
        return last_error = cudaErrorInvalidMemcpyDirection;
    CUdeviceptr pointer = 0;
    size_t size = 0;
    last_error = convertSymbolError(lookupSymbol(symbol, pointer, size));
    if (last_error != cudaSuccess) return last_error;
    if (offset > size || count > size - offset) return last_error = cudaErrorInvalidValue;
    return cudaMemcpy(dst, reinterpret_cast<void*>(pointer + offset), count, kind);
}

cudaError_t cudaGetSymbolAddress(void** pointer, const void* symbol) {
    if (!pointer) return last_error = cudaErrorInvalidValue;
    CUdeviceptr address = 0;
    size_t size = 0;
    last_error = convertSymbolError(lookupSymbol(symbol, address, size));
    if (last_error == cudaSuccess) *pointer = reinterpret_cast<void*>(address);
    return last_error;
}

cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol) {
    if (!size) return last_error = cudaErrorInvalidValue;
    CUdeviceptr address = 0;
    last_error = convertSymbolError(lookupSymbol(symbol, address, *size));
    return last_error;
}

cudaError_t cudaMemAdvise(const void *devPtr, size_t count, int advice, int device) {
    // Simplified: just succeed
    return setLastError(cudaSuccess);
}

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    // Simplified: just succeed after validating the stream handle.
    last_error = cudaSuccess;
    submitAsyncStreamOperation(stream);
    return last_error;
}

cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize, int attribute, const void *devPtr, size_t count) {
    // Simplified: just succeed
    return setLastError(cudaSuccess);
}

cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes, int *attributes, size_t numAttributes, const void *devPtr, size_t count) {
    // Simplified: just succeed
    return setLastError(cudaSuccess);
}

// ============================================================================
// Unified Memory
// ============================================================================

cudaError_t cudaMemAttachGlobal(void *devPtr) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaMemAttach(void *devPtr, unsigned int flags) {
    return setLastError(cudaSuccess);
}

// ============================================================================
// Memory Pool Management
// ============================================================================

cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool, const void *poolProps) {
    if (newPlaceholder(memPool) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaMemPoolCreate\n");
    }
    return last_error;
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) {
    free(memPool);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaMemPoolDestroy\n");
    return last_error;
}

cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) {
    if (!ptr) return setLastError(cudaErrorInvalidValue);
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    if (allocDevice(ptr, size) == cudaSuccess) {
        submitAsyncStreamOperation(stream);
    }
    return last_error;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, int attr, void *value) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, int attr, void *value) {
    if (!value) return setLastError(cudaErrorInvalidValue);

    // Attributes 1-3 (the reuse policies) are int; the rest are cuuint64_t.
    if (attr >= 1 && attr <= 3) {
        *(int*)value = 0;
    } else {
        *(uint64_t*)value = 0;
    }
    return setLastError(cudaSuccess);
}

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const void *descList, size_t count) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaMemPoolGetAccess(void *flags, cudaMemPool_t memPool, void *location) {
    if (!flags) return setLastError(cudaErrorInvalidValue);

    *(int*)flags = 1;  // Access granted
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device) {
    if (!memPool) return setLastError(cudaErrorInvalidValue);

    // Return a dummy default pool
    static cudaMemPool_t defaultPool = NULL;
    if (!defaultPool) {
        defaultPool = (cudaMemPool_t)malloc(sizeof(void*));
    }
    *memPool = defaultPool;
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device) {
    return cudaDeviceGetDefaultMemPool(memPool, device);
}

// ============================================================================
// Pointer Attributes
// ============================================================================

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr) {
    if (!attributes || !ptr) return setLastError(cudaErrorInvalidValue);

    memset(attributes, 0, sizeof(cudaPointerAttributes));

    size_t alloc_size = 0;
    int alloc_device = 0;
    fake_gpu::GlobalState::AllocationKind alloc_kind = fake_gpu::GlobalState::AllocationKind::Device;
    bool found = fake_gpu::GlobalState::instance().get_allocation_info_ex(
        const_cast<void*>(ptr), alloc_size, alloc_device, alloc_kind);

    if (!found) {
        attributes->type = cudaMemoryTypeUnregistered;
        attributes->device = 0;
    } else {
        switch (alloc_kind) {
            case fake_gpu::GlobalState::AllocationKind::Device:
                attributes->type = cudaMemoryTypeDevice;
                break;
            case fake_gpu::GlobalState::AllocationKind::Managed:
                attributes->type = cudaMemoryTypeManaged;
                break;
            case fake_gpu::GlobalState::AllocationKind::Host:
                attributes->type = cudaMemoryTypeHost;
                break;
        }
        attributes->device = alloc_device;
    }
    // Each alias is NULL when that side cannot access the memory directly:
    // the host cannot dereference device memory, nor the device unregistered memory.
    if (attributes->type != cudaMemoryTypeUnregistered) {
        attributes->devicePointer = const_cast<void*>(ptr);
    }
    if (attributes->type != cudaMemoryTypeDevice) {
        attributes->hostPointer = const_cast<void*>(ptr);
    }

    return setLastError(cudaSuccess);
}

// ============================================================================
// IPC (Inter-Process Communication)
// ============================================================================

cudaError_t cudaIpcGetMemHandle(void *handle, void *devPtr) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaIpcOpenMemHandle(void **devPtr, void *handle, unsigned int flags) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaIpcGetEventHandle(void *handle, cudaEvent_t event) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, void *handle) {
    return setLastError(cudaSuccess);
}

// ============================================================================
// CUDA Graph API
// ============================================================================

cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags) {
    if (newPlaceholder(pGraph) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphCreate\n");
    }
    return last_error;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    free(graph);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphDestroy\n");
    return last_error;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pNodeParams) {
    if (newPlaceholder(pGraphNode) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphAddKernelNode\n");
    }
    return last_error;
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pCopyParams) {
    if (newPlaceholder(pGraphNode) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphAddMemcpyNode\n");
    }
    return last_error;
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pMemsetParams) {
    if (newPlaceholder(pGraphNode) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphAddMemsetNode\n");
    }
    return last_error;
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pNodeParams) {
    if (newPlaceholder(pGraphNode) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphAddHostNode\n");
    }
    return last_error;
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph) {
    if (newPlaceholder(pGraphNode) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphAddChildGraphNode\n");
    }
    return last_error;
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies) {
    if (newPlaceholder(pGraphNode) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphAddEmptyNode\n");
    }
    return last_error;
}

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) {
    if (newPlaceholder(pGraphNode) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphAddEventRecordNode\n");
    }
    return last_error;
}

cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) {
    if (newPlaceholder(pGraphNode) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphAddEventWaitNode\n");
    }
    return last_error;
}

cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph) {
    if (newPlaceholder(pGraphClone) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphClone\n");
    }
    return last_error;
}

cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) {
    if (!pNode) return setLastError(cudaErrorInvalidValue);

    *pNode = originalNode;
    return setLastError(cudaSuccess);
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType *pType) {
    if (!pType) return setLastError(cudaErrorInvalidValue);

    *pType = cudaGraphNodeTypeKernel;
    return setLastError(cudaSuccess);
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes) {
    if (!numNodes) return setLastError(cudaErrorInvalidValue);

    *numNodes = 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes) {
    if (!pNumRootNodes) return setLastError(cudaErrorInvalidValue);

    *pNumRootNodes = 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges) {
    if (!numEdges) return setLastError(cudaErrorInvalidValue);

    *numEdges = 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies) {
    if (!pNumDependencies) return setLastError(cudaErrorInvalidValue);

    *pNumDependencies = 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes) {
    if (!pNumDependentNodes) return setLastError(cudaErrorInvalidValue);

    *pNumDependentNodes = 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize) {
    if (newPlaceholder(pGraphExec) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphInstantiate\n");
    }
    return last_error;
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags) {
    if (newPlaceholder(pGraphExec) == cudaSuccess) {
        FGPU_LOG("[FakeCUDART] cudaGraphInstantiateWithFlags flags=%llu\n", flags);
    }
    return last_error;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    free(graphExec);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaGraphExecDestroy\n");
    return last_error;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    if (!graphExec) return setLastError(cudaErrorInvalidValue);
    if (!validateStreamArgument(stream)) {
        return last_error;
    }
    return rejectKernelApi("cudaGraphLaunch");
}

cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) {
    if (!graphExec) return setLastError(cudaErrorInvalidValue);
    if (!validateStreamArgument(stream)) {
        return last_error;
    }
    if (rejectUnsupportedApi("cudaGraphUpload", "state_not_materialized")) {
        return last_error;
    }
    if (!submitAsyncStreamOperation(stream)) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaGraphUpload (stub)\n");
    return last_error;
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags) {
    rejectUnsupportedApi("cudaGraphDebugDotPrint", "output_not_written");
    FGPU_LOG("[FakeCUDART] cudaGraphDebugDotPrint path=%s flags=%u (stub)\n", path ? path : "NULL", flags);
    return last_error;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, int mode) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }
    if (!beginCapture(stream)) return setLastError(cudaErrorInvalidValue);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaStreamBeginCapture mode=%d\n", mode);
    return last_error;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) {
    if (!pGraph) return setLastError(cudaErrorInvalidValue);
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    unsigned long long capture_id = 0;
    if (!endCapture(stream, capture_id)) return setLastError(cudaErrorInvalidValue);

    *pGraph = (cudaGraph_t)malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaStreamEndCapture capture_id=%llu\n", capture_id);
    return last_error;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, int *pCaptureStatus) {
    if (!pCaptureStatus) return setLastError(cudaErrorInvalidValue);
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    const CaptureState state = lookupCaptureState(stream);
    *pCaptureStatus = state.active ? 1 : 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, int *captureStatus, unsigned long long *id) {
    if (!captureStatus) return setLastError(cudaErrorInvalidValue);
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    const CaptureState state = lookupCaptureState(stream);
    *captureStatus = state.active ? 1 : 0;
    if (id) *id = state.active ? state.id : 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaThreadExchangeStreamCaptureMode(int *mode) {
    if (!mode) return setLastError(cudaErrorInvalidValue);

    // Return current mode (simplified: always relaxed mode)
    int oldMode = *mode;
    *mode = 0;  // cudaStreamCaptureModeGlobal
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaThreadExchangeStreamCaptureMode old=%d new=0\n", oldMode);
    return last_error;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, int *captureStatus, unsigned long long *id, cudaGraph_t *graph, const cudaGraphNode_t **dependencies, size_t *numDependencies) {
    if (!captureStatus) return setLastError(cudaErrorInvalidValue);
    if (!validateStreamArgument(stream)) {
        return last_error;
    }

    const CaptureState state = lookupCaptureState(stream);
    *captureStatus = state.active ? 1 : 0;
    if (id) *id = state.active ? state.id : 0;
    if (graph) *graph = NULL;
    if (numDependencies) *numDependencies = 0;
    return setLastError(cudaSuccess);
}

// ============================================================================
// Texture and Surface Object API
// ============================================================================

cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const cudaResourceDesc *pResDesc, const cudaTextureDesc *pTexDesc, const cudaResourceViewDesc *pResViewDesc) {
    if (!pTexObject) return setLastError(cudaErrorInvalidValue);

    static cudaTextureObject_t nextTexId = 1;
    *pTexObject = nextTexId++;
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaCreateTextureObject id=%llu\n", *pTexObject);
    return last_error;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDestroyTextureObject id=%llu\n", texObject);
    return last_error;
}

cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc *pResDesc, cudaTextureObject_t texObject) {
    if (!pResDesc) return setLastError(cudaErrorInvalidValue);

    memset(pResDesc, 0, sizeof(cudaResourceDesc));
    return setLastError(cudaSuccess);
}

cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject) {
    if (!pTexDesc) return setLastError(cudaErrorInvalidValue);

    memset(pTexDesc, 0, sizeof(cudaTextureDesc));
    return setLastError(cudaSuccess);
}

cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject) {
    if (!pResViewDesc) return setLastError(cudaErrorInvalidValue);

    memset(pResViewDesc, 0, sizeof(cudaResourceViewDesc));
    return setLastError(cudaSuccess);
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const cudaResourceDesc *pResDesc) {
    if (!pSurfObject) return setLastError(cudaErrorInvalidValue);

    static cudaSurfaceObject_t nextSurfId = 1;
    *pSurfObject = nextSurfId++;
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaCreateSurfaceObject id=%llu\n", *pSurfObject);
    return last_error;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDestroySurfaceObject id=%llu\n", surfObject);
    return last_error;
}

cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) {
    if (!pResDesc) return setLastError(cudaErrorInvalidValue);

    memset(pResDesc, 0, sizeof(cudaResourceDesc));
    return setLastError(cudaSuccess);
}

// ============================================================================
// Cooperative Groups
// ============================================================================

cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
    return rejectKernelApi("cudaLaunchCooperativeKernel");
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(void *launchParamsList, unsigned int numDevices, unsigned int flags) {
    return rejectKernelApi("cudaLaunchCooperativeKernelMultiDevice");
}

// ============================================================================
// Thread Management (Deprecated but still used)
// ============================================================================

cudaError_t cudaThreadSynchronize(void) {
    // Deprecated, maps to cudaDeviceSynchronize
    return cudaDeviceSynchronize();
}

cudaError_t cudaThreadExit(void) {
    // Deprecated, maps to cudaDeviceReset
    return cudaDeviceReset();
}

// ============================================================================
// Function Attributes
// ============================================================================

cudaError_t cudaFuncGetAttributes(void *attr, const void *func) {
    if (!attr) return setLastError(cudaErrorInvalidValue);

    // Simplified: zero out attributes
    memset(attr, 0, sizeof(cudaFuncAttributes));
    return setLastError(cudaSuccess);
}

cudaError_t cudaFuncSetCacheConfig(const void *func, int cacheConfig) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaFuncSetSharedMemConfig(const void *func, int config) {
    return setLastError(cudaSuccess);
}

cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value) {
    return setLastError(cudaSuccess);
}

// ============================================================================
// Launch Bounds
// ============================================================================

cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
    if (!device || !pciBusId) return setLastError(cudaErrorInvalidValue);

    // Simplified: always return device 0
    *device = 0;
    return setLastError(cudaSuccess);
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    if (!pciBusId || len < 13) return setLastError(cudaErrorInvalidValue);

    snprintf(pciBusId, len, "0000:00:00.0");
    return setLastError(cudaSuccess);
}

// ============================================================================
// External Resource Interop
// ============================================================================

cudaError_t cudaImportExternalMemory(void **extMem_out, const void *memHandleDesc) {
    if (!extMem_out) return setLastError(cudaErrorInvalidValue);

    if (rejectUnsupportedApi(
            "cudaImportExternalMemory", "synthetic_external_memory_handle")) {
        return last_error;
    }
    *extMem_out = malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaImportExternalMemory (stub)\n");
    return last_error;
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void **devPtr, void *extMem, const void *bufferDesc) {
    if (!devPtr) return setLastError(cudaErrorInvalidValue);

    if (rejectUnsupportedApi(
            "cudaExternalMemoryGetMappedBuffer", "synthetic_fixed_size_buffer")) {
        return last_error;
    }
    *devPtr = malloc(1024);  // Dummy buffer
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaExternalMemoryGetMappedBuffer (stub)\n");
    return last_error;
}

cudaError_t cudaDestroyExternalMemory(void *extMem) {
    if (rejectUnsupportedApi(
            "cudaDestroyExternalMemory", "synthetic_external_memory_handle")) {
        return last_error;
    }
    free(extMem);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDestroyExternalMemory (stub)\n");
    return last_error;
}

cudaError_t cudaImportExternalSemaphore(void **extSem_out, const void *semHandleDesc) {
    if (!extSem_out) return setLastError(cudaErrorInvalidValue);

    if (rejectUnsupportedApi(
            "cudaImportExternalSemaphore", "synthetic_external_semaphore_handle")) {
        return last_error;
    }
    *extSem_out = malloc(sizeof(void*));
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaImportExternalSemaphore (stub)\n");
    return last_error;
}

cudaError_t cudaSignalExternalSemaphoresAsync(const void **extSemArray, const void **paramsArray, unsigned int numExtSems, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }
    if (rejectUnsupportedApi(
            "cudaSignalExternalSemaphoresAsync",
            "external_semaphore_signal_not_performed")) {
        return last_error;
    }
    if (!submitAsyncStreamOperation(stream)) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaSignalExternalSemaphoresAsync (stub)\n");
    return last_error;
}

cudaError_t cudaWaitExternalSemaphoresAsync(const void **extSemArray, const void **paramsArray, unsigned int numExtSems, cudaStream_t stream) {
    if (!validateStreamArgument(stream)) {
        return last_error;
    }
    if (rejectUnsupportedApi(
            "cudaWaitExternalSemaphoresAsync",
            "external_semaphore_wait_not_performed")) {
        return last_error;
    }
    if (!submitAsyncStreamOperation(stream)) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaWaitExternalSemaphoresAsync (stub)\n");
    return last_error;
}

cudaError_t cudaDestroyExternalSemaphore(void *extSem) {
    if (rejectUnsupportedApi(
            "cudaDestroyExternalSemaphore", "synthetic_external_semaphore_handle")) {
        return last_error;
    }
    free(extSem);
    last_error = cudaSuccess;
    FGPU_LOG("[FakeCUDART] cudaDestroyExternalSemaphore (stub)\n");
    return last_error;
}

// ============================================================================
// Profiling
// ============================================================================

cudaError_t cudaProfilerStart(void) {
    if (rejectUnsupportedApi("cudaProfilerStart", "profiler_not_started")) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaProfilerStart (stub)\n");
    return last_error;
}

cudaError_t cudaProfilerStop(void) {
    if (rejectUnsupportedApi("cudaProfilerStop", "profiler_not_started")) {
        return last_error;
    }
    FGPU_LOG("[FakeCUDART] cudaProfilerStop (stub)\n");
    return last_error;
}

// ============================================================================
// Internal CUDA Runtime Functions (Module/Variable Registration)
// ============================================================================

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, void *tid, void *bid, void *bDim, void *gDim, int *wSize) {
    auto& registry = runtimeRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.functions[hostFun] = {reinterpret_cast<RegisteredModule*>(fatCubinHandle), deviceName};
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global) {
    auto& registry = runtimeRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.symbols[hostVar] = {reinterpret_cast<RegisteredModule*>(fatCubinHandle), deviceName};
}

void** __cudaRegisterFatBinary(void *fatCubin) {
    auto* module = new RegisteredModule{fatCubin, {}};
    return reinterpret_cast<void**>(module);
}

char __cudaInitModule(void **fatCubinHandle) {
    // Device code is loaded lazily for the selected device's context.
    return 1;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    FGPU_LOG("[FakeCUDART] __cudaRegisterFatBinaryEnd\n");
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    auto* module = reinterpret_cast<RegisteredModule*>(fatCubinHandle);
    auto& registry = runtimeRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (auto it = registry.functions.begin(); it != registry.functions.end();) {
        if (it->second.owner == module) it = registry.functions.erase(it);
        else ++it;
    }
    for (auto it = registry.symbols.begin(); it != registry.symbols.end();) {
        if (it->second.owner == module) it = registry.symbols.erase(it);
        else ++it;
    }
    for (const auto& [device, loaded] : module->devices)
        if (loaded.status == CUDA_SUCCESS) cuModuleUnload(loaded.module);
    delete module;
}

void** __cudaRegisterFatBinaryEnd_v2(void **fatCubinHandle) {
    FGPU_LOG("[FakeCUDART] __cudaRegisterFatBinaryEnd_v2\n");
    return fatCubinHandle;
}

struct LaunchConfiguration {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    void* stream;
};

static thread_local std::vector<LaunchConfiguration> launch_configurations;

cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, void *stream) {
    launch_configurations.push_back({gridDim, blockDim, sharedMem, stream});
    return cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void **stream) {
    if (launch_configurations.empty()) return cudaErrorInvalidValue;
    const LaunchConfiguration config = launch_configurations.back();
    launch_configurations.pop_back();
    if (gridDim) *gridDim = config.grid;
    if (blockDim) *blockDim = config.block;
    if (sharedMem) *sharedMem = config.shared_mem;
    if (stream) *stream = config.stream;
    return cudaSuccess;
}

} // extern "C"
