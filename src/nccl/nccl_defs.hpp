#pragma once

#include <cstddef>
#include <cstdint>
#include <limits.h>

#define NCCL_MAJOR 2
#define NCCL_MINOR 27
#define NCCL_PATCH 5
#define NCCL_VERSION_CODE 22705
#define NCCL_VERSION(X, Y, Z) \
    ((((X) <= 2 && (Y) <= 8) ? ((X) * 1000 + (Y) * 100 + (Z)) : ((X) * 10000 + (Y) * 100 + (Z))))

#define NCCL_UNIQUE_ID_BYTES 128
#define NCCL_CONFIG_UNDEF_INT INT_MIN
#define NCCL_CONFIG_UNDEF_PTR NULL
#define NCCL_UNDEF_FLOAT -1.0f
#define NCCL_SPLIT_NOCOLOR -1

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __FAKEGPU_CUDA_STREAM_T_DEFINED
#define __FAKEGPU_CUDA_STREAM_T_DEFINED
typedef struct CUstream_st* cudaStream_t;
#endif

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8,
} ncclResult_t;

typedef struct ncclComm* ncclComm_t;
typedef struct ncclWindow* ncclWindow_t;

typedef struct {
    char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

typedef struct ncclConfig_v22700 {
    std::size_t size;
    unsigned int magic;
    unsigned int version;
    int blocking;
    int cgaClusterSize;
    int minCTAs;
    int maxCTAs;
    const char* netName;
    int splitShare;
    int trafficClass;
    const char* commName;
    int collnetEnable;
    int CTAPolicy;
    int shrinkShare;
    int nvlsCTAs;
} ncclConfig_t;

typedef struct ncclSimInfo_v22200 {
    std::size_t size;
    unsigned int magic;
    unsigned int version;
    float estimatedTime;
} ncclSimInfo_t;

typedef enum {
    ncclNumOps_dummy = 5,
} ncclRedOp_dummy_t;

typedef enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
    ncclNumOps = 5,
    ncclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(ncclRedOp_dummy_t)),
} ncclRedOp_t;

typedef enum {
    ncclInt8 = 0,
    ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclHalf = 6,
    ncclFloat32 = 7,
    ncclFloat = 7,
    ncclFloat64 = 8,
    ncclDouble = 8,
    ncclBfloat16 = 9,
    ncclFloat8e4m3 = 10,
    ncclFloat8e5m2 = 11,
    ncclNumTypes = 12,
} ncclDataType_t;

typedef enum {
    ncclScalarDevice = 0,
    ncclScalarHostImmediate = 1,
} ncclScalarResidence_t;

ncclResult_t ncclMemAlloc(void** ptr, std::size_t size);
ncclResult_t ncclMemFree(void* ptr);

ncclResult_t ncclGetVersion(int* version);
ncclResult_t ncclGetUniqueId(ncclUniqueId* unique_id);

ncclResult_t ncclCommInitRankConfig(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId comm_id,
    int rank,
    ncclConfig_t* config);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId comm_id, int rank);
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
ncclResult_t ncclCommInitRankScalable(
    ncclComm_t* comm,
    int nranks,
    int myrank,
    int n_id,
    ncclUniqueId* comm_ids,
    ncclConfig_t* config);

ncclResult_t ncclCommFinalize(ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* async_error);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, std::size_t size, void** handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle);
ncclResult_t ncclCommWindowRegister(
    ncclComm_t comm,
    void* buff,
    std::size_t size,
    ncclWindow_t* window,
    int window_flags);
ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t window);

const char* ncclGetErrorString(ncclResult_t result);
const char* ncclGetLastError(ncclComm_t comm);

ncclResult_t ncclRedOpCreatePreMulSum(
    ncclRedOp_t* op,
    void* scalar,
    ncclDataType_t datatype,
    ncclScalarResidence_t residence,
    ncclComm_t comm);
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);

ncclResult_t ncclReduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclBcast(
    void* buff,
    std::size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclBroadcast(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclAllReduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    std::size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclAllGather(
    const void* sendbuff,
    void* recvbuff,
    std::size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclSend(
    const void* sendbuff,
    std::size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t ncclRecv(
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    cudaStream_t stream);

ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* sim_info);

#ifdef __cplusplus
}
#endif
