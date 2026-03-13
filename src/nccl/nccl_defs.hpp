#pragma once

#include <cstdint>

#define NCCL_UNIQUE_ID_BYTES 128

#ifdef __cplusplus
extern "C" {
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

ncclResult_t ncclGetVersion(int* version);
ncclResult_t ncclGetUniqueId(ncclUniqueId* unique_id);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId comm_id, int rank);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
const char* ncclGetErrorString(ncclResult_t result);

#ifdef __cplusplus
}
#endif
