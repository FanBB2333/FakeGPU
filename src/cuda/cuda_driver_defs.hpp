#pragma once
#include <cstddef>
#include <cstdint>

// CUDA Driver API definitions
// These use 'cu' prefix instead of 'cuda' prefix

#ifdef __cplusplus
extern "C" {
#endif

// Result codes
typedef enum CUresult_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

// Opaque types
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUmod_st *CUmodule;
typedef struct CUmemPoolHandle_st *CUmemoryPool;
typedef struct CUfunc_st *CUfunction;
typedef unsigned long long CUdeviceptr;

// UUID structure
typedef struct CUuuid_st {
    char bytes[16];
} CUuuid;

// Limit types
typedef enum CUlimit_enum {
    CU_LIMIT_STACK_SIZE = 0x00,
    CU_LIMIT_PRINTF_FIFO_SIZE = 0x01,
    CU_LIMIT_MALLOC_HEAP_SIZE = 0x02,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05,
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06,
    CU_LIMIT_SHMEM_SIZE = 0x07,
    CU_LIMIT_CIG_ENABLED = 0x08,
    CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED = 0x09
} CUlimit;

// Stream flags
typedef enum CUstream_flags_enum {
    CU_STREAM_DEFAULT = 0x0,
    CU_STREAM_NON_BLOCKING = 0x1
} CUstream_flags;

// Event flags
typedef enum CUevent_flags_enum {
    CU_EVENT_DEFAULT = 0x0,
    CU_EVENT_BLOCKING_SYNC = 0x1,
    CU_EVENT_DISABLE_TIMING = 0x2,
    CU_EVENT_INTERPROCESS = 0x4
} CUevent_flags;

// Memory types
typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 1,
    CU_MEMORYTYPE_DEVICE = 2,
    CU_MEMORYTYPE_ARRAY = 3,
    CU_MEMORYTYPE_UNIFIED = 4
} CUmemorytype;

// Pointer attributes
typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20
} CUpointer_attribute;

// Device attributes (extended list)
typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129
} CUdevice_attribute;

// Context creation flags
typedef enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO = 0x00,
    CU_CTX_SCHED_SPIN = 0x01,
    CU_CTX_SCHED_YIELD = 0x02,
    CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
    CU_CTX_MAP_HOST = 0x08,
    CU_CTX_LMEM_RESIZE_TO_MAX = 0x10
} CUctx_flags;

// Virtual memory management (cuda.h, CUDA 12.4).
typedef unsigned long long CUmemGenericAllocationHandle;

typedef enum CUmemAllocationHandleType_enum {
    CU_MEM_HANDLE_TYPE_NONE = 0x0,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1,
    CU_MEM_HANDLE_TYPE_WIN32 = 0x2,
    CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4,
    CU_MEM_HANDLE_TYPE_FABRIC = 0x8,
    CU_MEM_HANDLE_TYPE_MAX = 0x7FFFFFFF
} CUmemAllocationHandleType;

typedef enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0x0,
    CU_MEM_LOCATION_TYPE_DEVICE = 0x1,
    CU_MEM_LOCATION_TYPE_HOST = 0x2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA = 0x3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 0x4,
    CU_MEM_LOCATION_TYPE_MAX = 0x7FFFFFFF
} CUmemLocationType;

typedef enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 0x1,
    CU_MEM_ALLOCATION_TYPE_MAX = 0x7FFFFFFF
} CUmemAllocationType;

typedef enum CUmemAllocationGranularity_flags_enum {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
} CUmemAllocationGranularity_flags;

typedef enum CUmemAccess_flags_enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
    CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
    CU_MEM_ACCESS_FLAGS_PROT_MAX = 0x7FFFFFFF
} CUmemAccess_flags;

typedef struct CUmemLocation_st {
    CUmemLocationType type;
    int id;
} CUmemLocation;

typedef struct CUmemAllocationProp_st {
    CUmemAllocationType type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation location;
    void *win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp;

typedef struct CUmemAccessDesc_st {
    CUmemLocation location;
    CUmemAccess_flags flags;
} CUmemAccessDesc;

// Interprocess handles are passed by value (cuda.h: CU_IPC_HANDLE_SIZE 64).
typedef struct CUipcMemHandle_st {
    char reserved[64];
} CUipcMemHandle;

typedef struct CUipcEventHandle_st {
    char reserved[64];
} CUipcEventHandle;

// Stream attributes share the launch attribute ids and 64-byte value union.
typedef enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10
} CUlaunchAttributeID;

typedef struct CUaccessPolicyWindow_st {
    void *base_ptr;
    size_t num_bytes;
    float hitRatio;
    int hitProp;
    int missProp;
} CUaccessPolicyWindow;

typedef union CUlaunchAttributeValue_union {
    char pad[64];
    CUaccessPolicyWindow accessPolicyWindow;
    int cooperative;
    int syncPolicy;
    int priority;
    struct {
        unsigned char default_;
        unsigned char remote;
    } memSyncDomainMap;
    int memSyncDomain;
} CUlaunchAttributeValue;

// Function declarations
CUresult cuInit(unsigned int Flags);
CUresult cuDriverGetVersion(int *driverVersion);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);
CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId);

// Context management
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxSynchronize(void);

// Memory management
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char value, size_t count);
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
CUresult cuMemAllocHost(void **pp, size_t bytesize);
CUresult cuMemFreeHost(void *p);
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);
CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
CUresult cuMemHostUnregister(void *p);
CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags);
CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p);
CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr);

// Primary context management
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUresult cuDevicePrimaryCtxReset(CUdevice dev);

// Context stack management
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext *pctx);

// Error handling
CUresult cuGetErrorString(CUresult error, const char **pStr);
CUresult cuGetErrorName(CUresult error, const char **pStr);

// Stream management
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult cuStreamDestroy(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuStreamQuery(CUstream hStream);
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
CUresult cuStreamGetPriority(CUstream hStream, int *priority);
CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags);
CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId);
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx);

// Event management
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult cuEventDestroy(CUevent hEvent);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventQuery(CUevent hEvent);
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);

// Context info
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuCtxGetFlags(unsigned int *flags);
CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit);
CUresult cuCtxSetLimit(CUlimit limit, size_t value);
CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);

// Memory info
CUresult cuMemGetInfo(size_t *free, size_t *total);

// Device UUID (v2)
CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);

// Additional context functions
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxCreate_v3(CUcontext *pctx, void *paramsArray, int numParams, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxPushCurrent_v2(CUcontext ctx);
CUresult cuCtxPopCurrent_v2(CUcontext *pctx);
CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags);

// Function to get proc address (critical for CUDA runtime)
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, unsigned long long flags);
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, unsigned long long flags, void *symbolStatus);

// Optional FakeGPU integration hook. Libraries such as fake NCCL resolve this
// symbol dynamically so they do not acquire a hard dependency on libcuda.
CUresult fakegpuCudaMarkStreamHostFunctionComplete(CUstream stream);

// PTX module loading and execution.
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleUnload(CUmodule module);
CUresult cuModuleGetFunction(CUfunction* function, CUmodule module, const char* name);
CUresult cuModuleGetGlobal(CUdeviceptr* pointer, size_t* bytes, CUmodule module, const char* name);
CUresult cuLaunchKernel(CUfunction function, unsigned gridX, unsigned gridY, unsigned gridZ,
                       unsigned blockX, unsigned blockY, unsigned blockZ,
                       unsigned sharedBytes, CUstream stream, void** arguments, void** extra);

// Occupancy, stream attributes and stream memory operations.
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);
CUresult cuStreamGetAttribute(CUstream hStream, CUlaunchAttributeID attr, CUlaunchAttributeValue *value_out);
CUresult cuStreamSetAttribute(CUstream hStream, CUlaunchAttributeID attr, const CUlaunchAttributeValue *value);
CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, uint32_t value, unsigned int flags);
CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, uint32_t value, unsigned int flags);
CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);

// Virtual memory management.
CUresult cuMemGetAllocationGranularity(
    size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop,
                     unsigned long long flags);
CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr,
                             unsigned long long flags);
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
                  unsigned long long flags);
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr);
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr);
CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle);
CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle,
                                      CUmemAllocationHandleType handleType, unsigned long long flags);
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle,
                                        CUmemAllocationHandleType shHandleType);

// Interprocess communication.
CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr);
CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags);
CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags);
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);
CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event);
CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle);

#ifdef __cplusplus
}

namespace fake_gpu::cuda {

bool is_stream_handle_live(CUstream stream);
CUresult submit_stream_operation(CUstream stream);
CUresult mark_stream_synchronized(CUstream stream);
CUresult mark_stream_host_function_complete(CUstream stream);

// Real driver entry point when a real driver backs the current mode, else nullptr.
void* real_driver_entry(const char* name);

// Releases a simulated allocation the CUDA APIs handed out. Returns
// CUDA_ERROR_INVALID_VALUE for addresses that are known but not freeable
// with this call, and applies FAKEGPU_UNSUPPORTED_API to unknown ones.
CUresult free_device_allocation(void* ptr);

// Modeled occupancy shared by the Driver and Runtime entry points.
CUresult model_occupancy(int* num_blocks, int device, int block_size, size_t dynamic_smem, unsigned int flags);

}  // namespace fake_gpu::cuda
#endif
