#pragma once

#include <cstddef>
#include <cstdint>

#define NCCL_UNIQUE_ID_BYTES 128

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __FAKEGPU_CUDA_STREAM_T_DEFINED
#define __FAKEGPU_CUDA_STREAM_T_DEFINED
typedef struct CUstream_st *cudaStream_t;
#endif

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
} ncclResult_t;

typedef struct ncclComm* ncclComm_t;

typedef struct {
    char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

typedef enum {
    ncclInt32 = 0,
    ncclFloat32 = 1,
    ncclInt = ncclInt32,
    ncclFloat = ncclFloat32,
} ncclDataType_t;

typedef enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
} ncclRedOp_t;

ncclResult_t ncclGetVersion(int* version);
ncclResult_t ncclGetUniqueId(ncclUniqueId* unique_id);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId comm_id, int rank);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclAllReduce(
    const void* sendbuff,
    void* recvbuff,
    std::size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
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
const char* ncclGetErrorString(ncclResult_t result);

#ifdef __cplusplus
}
#endif
