#pragma once
#include <cstddef>
#include <cstdint>

// Struct definitions (must be visible to C++, and compatible layout for C)
typedef struct uint3 {
    unsigned int x, y, z;
} uint3;

typedef struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    dim3(uint3 grid) : x(grid.x), y(grid.y), z(grid.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif
} dim3;

typedef struct cudaDeviceProp_st {
    char name[256];
    char uuid[16];                  // UUID for device (16 bytes)
    char luid[8];                   // LUID for device (8 bytes)
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
    int hostNativeAtomicSupported;
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

// API Declarations
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidValue = 11,
    cudaErrorInvalidDevice = 10,
    cudaErrorNoDevice = 100,
    cudaErrorUnknown = 30
} cudaError_t;

typedef enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef struct CUstream_st *cudaStream_t;

// Function signatures
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
cudaError_t cudaDriverGetVersion(int *driverVersion);
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);

#ifdef __cplusplus
}
#endif
