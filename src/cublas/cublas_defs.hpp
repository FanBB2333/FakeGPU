#pragma once
#include <cstddef>
#include <cstdint>

// Define __half type if not already defined (simplified stub - not actual FP16)
#ifndef __CUDA_FP16_H__
typedef struct __half {
    unsigned short x;
} __half;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// cuBLAS status codes
typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

// cuBLAS handle
typedef struct cublasContext *cublasHandle_t;

// cuBLAS operation types
typedef enum {
    CUBLAS_OP_N = 0,  // Non-transpose
    CUBLAS_OP_T = 1,  // Transpose
    CUBLAS_OP_C = 2   // Conjugate transpose
} cublasOperation_t;

// cuBLAS fill mode
typedef enum {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL = 2
} cublasFillMode_t;

// cuBLAS diagonal type
typedef enum {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1
} cublasDiagType_t;

// cuBLAS side mode
typedef enum {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1
} cublasSideMode_t;

// cuBLAS pointer mode
typedef enum {
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

// cuBLAS atomics mode
typedef enum {
    CUBLAS_ATOMICS_NOT_ALLOWED = 0,
    CUBLAS_ATOMICS_ALLOWED = 1
} cublasAtomicsMode_t;

// cuBLAS math mode
typedef enum {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
} cublasMath_t;

// cuBLAS compute type
typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73
} cublasComputeType_t;

// cuBLAS GEMM algorithm
typedef enum {
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_DFALT = 101,
    CUBLAS_GEMM_DFALT_TENSOR_OP = 102
} cublasGemmAlgo_t;

// Handle management
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version);
cublasStatus_t cublasGetProperty(int type, int *value);

// Stream management
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void *streamId);
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, void **streamId);

// Pointer mode
cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode);

// Math mode
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode);

// Atomics mode
cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);
cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode);

// BLAS Level 1: Vector operations
cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result);
cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result);
cublasStatus_t cublasIsamin_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result);
cublasStatus_t cublasIdamin_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result);

cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result);
cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result);

cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy);
cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy);

cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy);
cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy);

cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result);
cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result);
cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n, const void *x, int incx, const void *y, int incy, void *result);
cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n, const void *x, int incx, const void *y, int incy, void *result);
cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n, const void *x, int incx, const void *y, int incy, void *result);
cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n, const void *x, int incx, const void *y, int incy, void *result);

cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x, int xType, int incx, const void *y, int yType, int incy, void *result, int resultType, int executionType);

cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result);
cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result);

cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x, int incx);
cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha, double *x, int incx);

// BLAS Level 2: Matrix-vector operations
cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy);
cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const void *alpha, const void *A, int lda, const void *x, int incx, const void *beta, void *y, int incy);
cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const void *alpha, const void *A, int lda, const void *x, int incx, const void *beta, void *y, int incy);

cublasStatus_t cublasSger_v2(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);
cublasStatus_t cublasDger_v2(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda);

// BLAS Level 3: Matrix-matrix operations (GEMM)
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);
cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int lda, const void *B, int ldb, const void *beta, void *C, int ldc);
cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int lda, const void *B, int ldb, const void *beta, void *C, int ldc);
cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc);

// GEMM with stride (for batched operations)
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount);
cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int lda, long long int strideA, const void *B, int ldb, long long int strideB, const void *beta, void *C, int ldc, long long int strideC, int batchCount);
cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int lda, long long int strideA, const void *B, int ldb, long long int strideB, const void *beta, void *C, int ldc, long long int strideC, int batchCount);
cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, long long int strideA, const __half *B, int ldb, long long int strideB, const __half *beta, __half *C, int ldc, long long int strideC, int batchCount);

// Batched GEMM
cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *const Aarray[], int lda, const float *const Barray[], int ldb, const float *beta, float *const Carray[], int ldc, int batchCount);
cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *const Aarray[], int lda, const double *const Barray[], int ldb, const double *beta, double *const Carray[], int ldc, int batchCount);

// GEMMEx (mixed precision GEMM)
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int Atype, int lda, const void *B, int Btype, int ldb, const void *beta, void *C, int Ctype, int ldc, int computeType, cublasGemmAlgo_t algo);
cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const void *A, int Atype, int lda, const void *B, int Btype, int ldb, const float *beta, void *C, int Ctype, int ldc);
cublasStatus_t cublasGemmEx_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int Atype, int lda, const void *B, int Btype, int ldb, const void *beta, void *C, int Ctype, int ldc, int computeType, cublasGemmAlgo_t algo);

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int Atype, int lda, long long int strideA, const void *B, int Btype, int ldb, long long int strideB, const void *beta, void *C, int Ctype, int ldc, long long int strideC, int batchCount, int computeType, cublasGemmAlgo_t algo);

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *const Aarray[], int Atype, int lda, const void *const Barray[], int Btype, int ldb, const void *beta, void *const Carray[], int Ctype, int ldc, int batchCount, int computeType, cublasGemmAlgo_t algo);

// TRSM (Triangular solve)
cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb);
cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb);
cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const void *alpha, const void *A, int lda, void *B, int ldb);
cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const void *alpha, const void *A, int lda, void *B, int ldb);

// TRSM batched
cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *const A[], int lda, float *const B[], int ldb, int batchCount);
cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *const A[], int lda, double *const B[], int ldb, int batchCount);
cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const void *alpha, const void *const A[], int lda, void *const B[], int ldb, int batchCount);
cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const void *alpha, const void *const A[], int lda, void *const B[], int ldb, int batchCount);

// GETRF batched (LU factorization)
cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);
cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);
cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, void *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);
cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, void *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);

// GETRS batched (Solve with LU)
cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchSize);
cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *const Aarray[], int lda, const int *devIpiv, double *const Barray[], int ldb, int *info, int batchSize);
cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const void *const Aarray[], int lda, const int *devIpiv, void *const Barray[], int ldb, int *info, int batchSize);
cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const void *const Aarray[], int lda, const int *devIpiv, void *const Barray[], int ldb, int *info, int batchSize);

// GEQRF batched (QR factorization)
cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float *const Aarray[], int lda, float *const TauArray[], int *info, int batchSize);
cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double *const Aarray[], int lda, double *const TauArray[], int *info, int batchSize);
cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, void *const Aarray[], int lda, void *const TauArray[], int *info, int batchSize);
cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, void *const Aarray[], int lda, void *const TauArray[], int *info, int batchSize);

// GELS batched (Least squares)
cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float *const Aarray[], int lda, float *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize);
cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double *const Aarray[], int lda, double *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize);
cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, void *const Aarray[], int lda, void *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize);
cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, void *const Aarray[], int lda, void *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize);

// SYMM/HEMM (Symmetric/Hermitian matrix multiplication)
cublasStatus_t cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
cublasStatus_t cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);

// SYRK (Symmetric rank-k update)
cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc);
cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc);

// SYR2K (Symmetric rank-2k update)
cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);

// SYRKX (Symmetric rank-k update with different input matrices)
cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);

// TRMM (Triangular matrix multiplication)
cublasStatus_t cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, double *C, int ldc);

// Legacy function names (without _v2 suffix)
cublasStatus_t cublasCreate(cublasHandle_t *handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasSetStream(cublasHandle_t handle, void *streamId);
cublasStatus_t cublasGetStream(cublasHandle_t handle, void **streamId);

// Workspace management
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes);
cublasStatus_t cublasGetWorkspace_v2(cublasHandle_t handle, void **workspace, size_t *workspaceSizeInBytes);

// ============================================================================
// cuBLASLt (Lightweight BLAS) API - Modern GEMM interface used by PyTorch
// ============================================================================

typedef struct cublasLtContext *cublasLtHandle_t;
typedef struct cublasLtMatmulDescOpaque *cublasLtMatmulDesc_t;
typedef struct cublasLtMatrixLayoutOpaque *cublasLtMatrixLayout_t;
typedef struct cublasLtMatmulPreferenceOpaque *cublasLtMatmulPreference_t;

// Algorithm heuristic structures
typedef struct {
    int algo;
    size_t workspaceSize;
    int state;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

// cuBLASLt handle management
cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle);

// Descriptor creation/destruction
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, int computeType, int scaleType);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, int attr, const void *buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, int attr, void *buf, size_t sizeInBytes, size_t *sizeWritten);

// Matrix layout creation/destruction
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, int type, uint64_t rows, uint64_t cols, int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, int attr, const void *buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout, int attr, void *buf, size_t sizeInBytes, size_t *sizeWritten);

// Matmul preference
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, int attr, const void *buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref, int attr, void *buf, size_t sizeInBytes, size_t *sizeWritten);

// Algorithm selection
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference,
    int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount
);

// Actual matmul execution
cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void *alpha,
    const void *A,
    cublasLtMatrixLayout_t Adesc,
    const void *B,
    cublasLtMatrixLayout_t Bdesc,
    const void *beta,
    const void *C,
    cublasLtMatrixLayout_t Cdesc,
    void *D,
    cublasLtMatrixLayout_t Ddesc,
    const void *algo,
    void *workspace,
    size_t workspaceSizeInBytes,
    void *stream
);

#ifdef __cplusplus
}
#endif
