#pragma once
#include <cstddef>
#include <cstdint>

// CUDA Runtime API definitions
// These use 'cuda' prefix instead of 'cu' prefix

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum cudaError_enum {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorInvalidDevice = 101,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorUnknown = 999
} cudaError_t;

// Memory copy kinds
typedef enum cudaMemcpyKind_enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

// Stream type
typedef struct CUstream_st *cudaStream_t;

// Event type
typedef struct CUevent_st *cudaEvent_t;

// Device properties
typedef struct cudaDeviceProp {
    char name[256];
    char uuid[16];
    char luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
} cudaDeviceProp;

// Dim3 structure for kernel launch
typedef struct dim3 {
    unsigned int x, y, z;
} dim3;

// Function declarations - Device Management
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaDeviceSynchronize(void);

// Memory Management
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);

// Stream Management
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);

// Event Management
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

// Error Handling
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);

// Version Management
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
cudaError_t cudaDriverGetVersion(int *driverVersion);

// Kernel Launch
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaLaunchKernelExC(const void *config, const void *func, void **args);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void *userData), void *userData);
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
cudaError_t cudaLaunch(const void *func);

// Host Memory
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void *ptr);

// Peer Access
cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);

// Driver Entry Point (CUDA 12+)
cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr, unsigned int cudaVersion, unsigned long long flags, int *driverStatus);
cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags);

// Additional Stream Management
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority);
cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaStreamAddCallback(cudaStream_t stream, void (*callback)(cudaStream_t stream, cudaError_t status, void *userData), void *userData, unsigned int flags);

// Additional Event Management
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags);

// Occupancy Calculation
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);

// Device Attributes
cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device);
cudaError_t cudaDeviceGetLimit(size_t *pValue, int limit);
cudaError_t cudaDeviceSetLimit(int limit, size_t value);
cudaError_t cudaDeviceGetCacheConfig(int *pCacheConfig);
cudaError_t cudaDeviceSetCacheConfig(int cacheConfig);
cudaError_t cudaDeviceGetSharedMemConfig(int *pConfig);
cudaError_t cudaDeviceSetSharedMemConfig(int config);
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

// Additional Memory Management
cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
cudaError_t cudaMalloc3D(void **devPtr, size_t width, size_t height, size_t depth);
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
cudaError_t cudaMallocArray(void **array, const void *desc, size_t width, size_t height, unsigned int flags);
cudaError_t cudaFreeArray(void *array);
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy3D(const void *p);
cudaError_t cudaMemcpy3DAsync(const void *p, cudaStream_t stream);
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, cudaMemcpyKind kind);
cudaError_t cudaMemAdvise(const void *devPtr, size_t count, int advice, int device);
cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream);
cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize, int attribute, const void *devPtr, size_t count);
cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes, int *attributes, size_t numAttributes, const void *devPtr, size_t count);

// Unified Memory
cudaError_t cudaMemAttachGlobal(void *devPtr);
cudaError_t cudaMemAttach(void *devPtr, unsigned int flags);

// Memory Pool Management
typedef struct CUmemoryPool_st *cudaMemPool_t;

cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool, const void *poolProps);
cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool);
cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream);
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep);
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, int attr, void *value);
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, int attr, void *value);
cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const void *descList, size_t count);
cudaError_t cudaMemPoolGetAccess(void *flags, cudaMemPool_t memPool, void *location);
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device);
cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool);
cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device);

// Pointer Attributes
cudaError_t cudaPointerGetAttributes(void *attributes, const void *ptr);

// IPC (Inter-Process Communication)
cudaError_t cudaIpcGetMemHandle(void *handle, void *devPtr);
cudaError_t cudaIpcOpenMemHandle(void **devPtr, void *handle, unsigned int flags);
cudaError_t cudaIpcCloseMemHandle(void *devPtr);
cudaError_t cudaIpcGetEventHandle(void *handle, cudaEvent_t event);
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, void *handle);

// CUDA Graph API
typedef struct CUgraph_st *cudaGraph_t;
typedef struct CUgraphNode_st *cudaGraphNode_t;
typedef struct CUgraphExec_st *cudaGraphExec_t;

typedef enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7
} cudaGraphNodeType;

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags);

cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pNodeParams);
cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pCopyParams);
cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pMemsetParams);
cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *pNodeParams);
cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph);
cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies);
cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph);
cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType *pType);
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes);
cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes);
cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges);
cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies);
cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes);
cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);
cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize);
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags);
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, int mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, int *pCaptureStatus);
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, int *captureStatus, unsigned long long *id);
cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, int *captureStatus, unsigned long long *id, cudaGraph_t *graph, const cudaGraphNode_t **dependencies, size_t *numDependencies);
cudaError_t cudaThreadExchangeStreamCaptureMode(int *mode);

// Device Properties v2 (for newer CUDA versions)
cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp *prop, int device);

// Texture and Surface Object API
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long cudaSurfaceObject_t;

typedef enum cudaTextureAddressMode {
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3
} cudaTextureAddressMode;

typedef enum cudaTextureFilterMode {
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1
} cudaTextureFilterMode;

typedef enum cudaTextureReadMode {
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1
} cudaTextureReadMode;

typedef struct cudaResourceDesc {
    int resType;
    union {
        struct {
            void *array;
        } array;
        struct {
            void *mipmap;
        } mipmap;
        struct {
            void *devPtr;
            void *desc;
            size_t sizeInBytes;
        } linear;
        struct {
            void *devPtr;
            void *desc;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
    } res;
} cudaResourceDesc;

typedef struct cudaTextureDesc {
    cudaTextureAddressMode addressMode[3];
    cudaTextureFilterMode filterMode;
    cudaTextureReadMode readMode;
    int sRGB;
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    int mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    int disableTrilinearOptimization;
} cudaTextureDesc;

typedef struct cudaResourceViewDesc {
    int format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
} cudaResourceViewDesc;

cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const cudaResourceDesc *pResDesc, const cudaTextureDesc *pTexDesc, const cudaResourceViewDesc *pResViewDesc);
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc *pResDesc, cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject);
cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject);

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const cudaResourceDesc *pResDesc);
cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject);

// Cooperative Groups
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaLaunchCooperativeKernelMultiDevice(void *launchParamsList, unsigned int numDevices, unsigned int flags);

// Thread Management (Deprecated but still used)
cudaError_t cudaThreadSynchronize(void);
cudaError_t cudaThreadExit(void);

// Function Attributes
cudaError_t cudaFuncGetAttributes(void *attr, const void *func);
cudaError_t cudaFuncSetCacheConfig(const void *func, int cacheConfig);
cudaError_t cudaFuncSetSharedMemConfig(const void *func, int config);
cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value);

// Launch Bounds
cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);
cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);

// External Resource Interop
cudaError_t cudaImportExternalMemory(void **extMem_out, const void *memHandleDesc);
cudaError_t cudaExternalMemoryGetMappedBuffer(void **devPtr, void *extMem, const void *bufferDesc);
cudaError_t cudaDestroyExternalMemory(void *extMem);

cudaError_t cudaImportExternalSemaphore(void **extSem_out, const void *semHandleDesc);
cudaError_t cudaSignalExternalSemaphoresAsync(const void **extSemArray, const void **paramsArray, unsigned int numExtSems, cudaStream_t stream);
cudaError_t cudaWaitExternalSemaphoresAsync(const void **extSemArray, const void **paramsArray, unsigned int numExtSems, cudaStream_t stream);
cudaError_t cudaDestroyExternalSemaphore(void *extSem);

// Profiling
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);

// Internal CUDA Runtime functions (for module/variable registration)
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, void *tid, void *bid, void *bDim, void *gDim, int *wSize);
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global);
void __cudaRegisterFatBinary(void *fatCubin);
void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
void __cudaUnregisterFatBinary(void **fatCubinHandle);
void** __cudaRegisterFatBinaryEnd_v2(void **fatCubinHandle);

cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, void *stream);
cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void **stream);

#ifdef __cplusplus
}
#endif
