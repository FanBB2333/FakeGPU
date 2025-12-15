#include "cublas_defs.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <map>
#include <mutex>

// Global state for cuBLAS handles
struct CublasHandle {
    void *stream;
    cublasPointerMode_t pointerMode;
    cublasMath_t mathMode;
    cublasAtomicsMode_t atomicsMode;

    CublasHandle() : stream(nullptr),
                     pointerMode(CUBLAS_POINTER_MODE_HOST),
                     mathMode(CUBLAS_DEFAULT_MATH),
                     atomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED) {}
};

static std::map<cublasHandle_t, CublasHandle*> g_handles;
static std::mutex g_mutex;
static std::mt19937 g_rng(std::random_device{}());

// Helper: Fill output buffer with random values
template<typename T>
static void fillRandom(T *data, size_t count) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; i++) {
        data[i] = static_cast<T>(dist(g_rng));
    }
}

// Helper: Get alpha/beta values (handles both host and device pointer modes)
template<typename T>
static T getScalar(const T *ptr, cublasPointerMode_t mode) {
    if (mode == CUBLAS_POINTER_MODE_HOST) {
        return *ptr;
    } else {
        // Device mode: just dereference (we're using system RAM anyway)
        return *ptr;
    }
}

extern "C" {

// ============================================================================
// Handle Management
// ============================================================================

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    if (!handle) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    CublasHandle *h = new CublasHandle();
    *handle = reinterpret_cast<cublasHandle_t>(h);
    g_handles[*handle] = h;

    printf("[FakeCUBLAS] cublasCreate_v2 handle=%p\n", *handle);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCreate(cublasHandle_t *handle) {
    return cublasCreate_v2(handle);
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    delete it->second;
    g_handles.erase(it);

    printf("[FakeCUBLAS] cublasDestroy_v2 handle=%p\n", handle);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return cublasDestroy_v2(handle);
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
    if (!version) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    *version = 12000;  // cuBLAS 12.0
    printf("[FakeCUBLAS] cublasGetVersion_v2 returning 12000\n");
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetProperty(int type, int *value) {
    if (!value) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    *value = 12000;  // Default value
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Stream Management
// ============================================================================

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void *streamId) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    it->second->stream = streamId;
    printf("[FakeCUBLAS] cublasSetStream_v2 stream=%p\n", streamId);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, void *streamId) {
    return cublasSetStream_v2(handle, streamId);
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, void **streamId) {
    if (!streamId) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    *streamId = it->second->stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, void **streamId) {
    return cublasGetStream_v2(handle, streamId);
}

// ============================================================================
// Pointer Mode
// ============================================================================

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    it->second->pointerMode = mode;
    printf("[FakeCUBLAS] cublasSetPointerMode_v2 mode=%d\n", mode);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode) {
    if (!mode) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    *mode = it->second->pointerMode;
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Math Mode
// ============================================================================

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    it->second->mathMode = mode;
    printf("[FakeCUBLAS] cublasSetMathMode mode=%d\n", mode);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    if (!mode) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    *mode = it->second->mathMode;
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Atomics Mode
// ============================================================================

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    it->second->atomicsMode = mode;
    printf("[FakeCUBLAS] cublasSetAtomicsMode mode=%d\n", mode);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
    if (!mode) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    *mode = it->second->atomicsMode;
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// BLAS Level 1: Vector Operations (Stubs - return random results)
// ============================================================================

cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_int_distribution<int>(0, n-1)(g_rng);
    printf("[FakeCUBLAS] cublasIsamax_v2 n=%d result=%d\n", n, *result);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_int_distribution<int>(0, n-1)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamin_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_int_distribution<int>(0, n-1)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamin_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_int_distribution<int>(0, n-1)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_real_distribution<float>(0.0f, static_cast<float>(n))(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_real_distribution<double>(0.0, static_cast<double>(n))(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
    if (n <= 0 || !alpha || !x || !y) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    // y = alpha*x + y - just fill with random values
    fillRandom(y, n);
    printf("[FakeCUBLAS] cublasSaxpy_v2 n=%d\n", n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy) {
    if (n <= 0 || !alpha || !x || !y) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    fillRandom(y, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy) {
    if (n <= 0 || !x || !y) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    fillRandom(y, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy) {
    if (n <= 0 || !x || !y) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    fillRandom(y, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_real_distribution<float>(-10.0f, 10.0f)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_real_distribution<double>(-10.0, 10.0)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_real_distribution<float>(0.0f, 10.0f)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *result = std::uniform_real_distribution<double>(0.0, 10.0)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
    if (n <= 0 || !alpha || !x) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    fillRandom(x, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
    if (n <= 0 || !alpha || !x) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    fillRandom(x, n);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// BLAS Level 2: Matrix-Vector Operations
// ============================================================================

cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
    if (!A || !x || !y || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    int output_size = (trans == CUBLAS_OP_N) ? m : n;
    fillRandom(y, output_size);

    printf("[FakeCUBLAS] cublasSgemv_v2 trans=%d m=%d n=%d\n", trans, m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
    if (!A || !x || !y || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    int output_size = (trans == CUBLAS_OP_N) ? m : n;
    fillRandom(y, output_size);

    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger_v2(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
    if (!A || !x || !y || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(A, m * n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger_v2(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
    if (!A || !x || !y || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(A, m * n);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// BLAS Level 3: Matrix-Matrix Operations (GEMM)
// ============================================================================

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    // C is m x n - fill with random values
    size_t total_elements = static_cast<size_t>(m) * n;
    fillRandom(C, total_elements);

    printf("[FakeCUBLAS] cublasSgemm_v2 m=%d n=%d k=%d (output %zu elements)\n", m, n, k, total_elements);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t total_elements = static_cast<size_t>(m) * n;
    fillRandom(C, total_elements);

    printf("[FakeCUBLAS] cublasDgemm_v2 m=%d n=%d k=%d (output %zu elements)\n", m, n, k, total_elements);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    // Fill with random float16 values (simplified - just set random bits)
    size_t total_elements = static_cast<size_t>(m) * n;
    std::uniform_int_distribution<unsigned short> dist(0, 0xFFFF);
    for (size_t i = 0; i < total_elements; i++) {
        C[i].x = dist(g_rng);
    }

    printf("[FakeCUBLAS] cublasHgemm m=%d n=%d k=%d\n", m, n, k);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Strided Batched GEMM
// ============================================================================

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t total_elements = static_cast<size_t>(m) * n * batchCount;
    fillRandom(C, total_elements);

    printf("[FakeCUBLAS] cublasSgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t total_elements = static_cast<size_t>(m) * n * batchCount;
    fillRandom(C, total_elements);

    printf("[FakeCUBLAS] cublasDgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, long long int strideA, const __half *B, int ldb, long long int strideB, const __half *beta, __half *C, int ldc, long long int strideC, int batchCount) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t total_elements = static_cast<size_t>(m) * n * batchCount;
    std::uniform_int_distribution<unsigned short> dist(0, 0xFFFF);
    for (size_t i = 0; i < total_elements; i++) {
        C[i].x = dist(g_rng);
    }

    printf("[FakeCUBLAS] cublasHgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Batched GEMM
// ============================================================================

cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *const Aarray[], int lda, const float *const Barray[], int ldb, const float *beta, float *const Carray[], int ldc, int batchCount) {
    if (!Aarray || !Barray || !Carray || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t matrix_elements = static_cast<size_t>(m) * n;
    for (int i = 0; i < batchCount; i++) {
        if (Carray[i]) {
            fillRandom(Carray[i], matrix_elements);
        }
    }

    printf("[FakeCUBLAS] cublasSgemmBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *const Aarray[], int lda, const double *const Barray[], int ldb, const double *beta, double *const Carray[], int ldc, int batchCount) {
    if (!Aarray || !Barray || !Carray || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t matrix_elements = static_cast<size_t>(m) * n;
    for (int i = 0; i < batchCount; i++) {
        if (Carray[i]) {
            fillRandom(Carray[i], matrix_elements);
        }
    }

    printf("[FakeCUBLAS] cublasDgemmBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// GEMMEx (Mixed Precision GEMM)
// ============================================================================

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int Atype, int lda, const void *B, int Btype, int ldb, const void *beta, void *C, int Ctype, int ldc, int computeType, cublasGemmAlgo_t algo) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    // Determine element size based on Ctype
    size_t element_size = 4;  // Default to float
    if (Ctype == 0) element_size = 2;  // half
    if (Ctype == 2) element_size = 8;  // double

    size_t total_bytes = static_cast<size_t>(m) * n * element_size;

    // Fill with random bytes
    unsigned char *ptr = static_cast<unsigned char*>(C);
    for (size_t i = 0; i < total_bytes; i++) {
        ptr[i] = static_cast<unsigned char>(std::uniform_int_distribution<int>(0, 255)(g_rng));
    }

    printf("[FakeCUBLAS] cublasGemmEx m=%d n=%d k=%d Atype=%d Btype=%d Ctype=%d computeType=%d\n",
           m, n, k, Atype, Btype, Ctype, computeType);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int Atype, int lda, long long int strideA, const void *B, int Btype, int ldb, long long int strideB, const void *beta, void *C, int Ctype, int ldc, long long int strideC, int batchCount, int computeType, cublasGemmAlgo_t algo) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t element_size = 4;
    if (Ctype == 0) element_size = 2;
    if (Ctype == 2) element_size = 8;

    size_t total_bytes = static_cast<size_t>(m) * n * batchCount * element_size;

    unsigned char *ptr = static_cast<unsigned char*>(C);
    for (size_t i = 0; i < total_bytes; i++) {
        ptr[i] = static_cast<unsigned char>(std::uniform_int_distribution<int>(0, 255)(g_rng));
    }

    printf("[FakeCUBLAS] cublasGemmStridedBatchedEx m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *const Aarray[], int Atype, int lda, const void *const Barray[], int Btype, int ldb, const void *beta, void *const Carray[], int Ctype, int ldc, int batchCount, int computeType, cublasGemmAlgo_t algo) {
    if (!Aarray || !Barray || !Carray || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t element_size = 4;
    if (Ctype == 0) element_size = 2;
    if (Ctype == 2) element_size = 8;

    size_t matrix_bytes = static_cast<size_t>(m) * n * element_size;

    for (int i = 0; i < batchCount; i++) {
        if (Carray[i]) {
            unsigned char *ptr = static_cast<unsigned char*>(Carray[i]);
            for (size_t j = 0; j < matrix_bytes; j++) {
                ptr[j] = static_cast<unsigned char>(std::uniform_int_distribution<int>(0, 255)(g_rng));
            }
        }
    }

    printf("[FakeCUBLAS] cublasGemmBatchedEx m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// TRSM (Triangular Solve)
// ============================================================================

cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb) {
    if (!A || !B || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(B, m * n);
    printf("[FakeCUBLAS] cublasStrsm_v2 m=%d n=%d\n", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb) {
    if (!A || !B || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(B, m * n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *const A[], int lda, float *const B[], int ldb, int batchCount) {
    if (!A || !B || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t matrix_elements = static_cast<size_t>(m) * n;
    for (int i = 0; i < batchCount; i++) {
        if (B[i]) {
            fillRandom(B[i], matrix_elements);
        }
    }

    printf("[FakeCUBLAS] cublasStrsmBatched m=%d n=%d batchCount=%d\n", m, n, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *const A[], int lda, double *const B[], int ldb, int batchCount) {
    if (!A || !B || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    size_t matrix_elements = static_cast<size_t>(m) * n;
    for (int i = 0; i < batchCount; i++) {
        if (B[i]) {
            fillRandom(B[i], matrix_elements);
        }
    }

    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// SYMM (Symmetric Matrix Multiplication)
// ============================================================================

cublasStatus_t cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, m * n);
    printf("[FakeCUBLAS] cublasSsymm_v2 m=%d n=%d\n", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, m * n);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// SYRK (Symmetric Rank-K Update)
// ============================================================================

cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc) {
    if (!A || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, n * n);
    printf("[FakeCUBLAS] cublasSsyrk_v2 n=%d k=%d\n", n, k);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc) {
    if (!A || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, n * n);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// SYR2K (Symmetric Rank-2K Update)
// ============================================================================

cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, n * n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, n * n);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// SYRKX (Symmetric Rank-K Update with Different Input Matrices)
// ============================================================================

cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, n * n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, n * n);
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// TRMM (Triangular Matrix Multiplication)
// ============================================================================

cublasStatus_t cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    if (!A || !B || !C || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, m * n);
    printf("[FakeCUBLAS] cublasStrmm_v2 m=%d n=%d\n", m, n);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, double *C, int ldc) {
    if (!A || !B || !C || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(C, m * n);
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
// Complex dot products
cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n, const void *x, int incx, const void *y, int incy, void *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    float *res = (float*)result;
    res[0] = std::uniform_real_distribution<float>(-10.0f, 10.0f)(g_rng);
    res[1] = std::uniform_real_distribution<float>(-10.0f, 10.0f)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n, const void *x, int incx, const void *y, int incy, void *result) {
    return cublasCdotu_v2(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n, const void *x, int incx, const void *y, int incy, void *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    double *res = (double*)result;
    res[0] = std::uniform_real_distribution<double>(-10.0, 10.0)(g_rng);
    res[1] = std::uniform_real_distribution<double>(-10.0, 10.0)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n, const void *x, int incx, const void *y, int incy, void *result) {
    return cublasZdotu_v2(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x, int xType, int incx, const void *y, int yType, int incy, void *result, int resultType, int executionType) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    float *res = (float*)result;
    *res = std::uniform_real_distribution<float>(-10.0f, 10.0f)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
}

// Complex GEMV
cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const void *alpha, const void *A, int lda, const void *x, int incx, const void *beta, void *y, int incy) {
    if (!A || !x || !y || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    int output_size = (trans == CUBLAS_OP_N) ? m : n;
    unsigned char *ptr = (unsigned char*)y;
    for (int i = 0; i < output_size * 8; i++) {
        ptr[i] = std::uniform_int_distribution<int>(0, 255)(g_rng);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const void *alpha, const void *A, int lda, const void *x, int incx, const void *beta, void *y, int incy) {
    if (!A || !x || !y || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    int output_size = (trans == CUBLAS_OP_N) ? m : n;
    unsigned char *ptr = (unsigned char*)y;
    for (int i = 0; i < output_size * 16; i++) {
        ptr[i] = std::uniform_int_distribution<int>(0, 255)(g_rng);
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Complex GEMM
cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int lda, const void *B, int ldb, const void *beta, void *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t total_bytes = static_cast<size_t>(m) * n * 8;  // complex float = 8 bytes
    unsigned char *ptr = (unsigned char*)C;
    for (size_t i = 0; i < total_bytes; i++) {
        ptr[i] = std::uniform_int_distribution<int>(0, 255)(g_rng);
    }
    printf("[FakeCUBLAS] cublasCgemm_v2 m=%d n=%d k=%d\n", m, n, k);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int lda, const void *B, int ldb, const void *beta, void *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t total_bytes = static_cast<size_t>(m) * n * 16;  // complex double = 16 bytes
    unsigned char *ptr = (unsigned char*)C;
    for (size_t i = 0; i < total_bytes; i++) {
        ptr[i] = std::uniform_int_distribution<int>(0, 255)(g_rng);
    }
    printf("[FakeCUBLAS] cublasZgemm_v2 m=%d n=%d k=%d\n", m, n, k);
    return CUBLAS_STATUS_SUCCESS;
}

// Complex strided batched GEMM
cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int lda, long long int strideA, const void *B, int ldb, long long int strideB, const void *beta, void *C, int ldc, long long int strideC, int batchCount) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t total_bytes = static_cast<size_t>(m) * n * batchCount * 8;
    unsigned char *ptr = (unsigned char*)C;
    for (size_t i = 0; i < total_bytes; i++) {
        ptr[i] = std::uniform_int_distribution<int>(0, 255)(g_rng);
    }
    printf("[FakeCUBLAS] cublasCgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int lda, long long int strideA, const void *B, int ldb, long long int strideB, const void *beta, void *C, int ldc, long long int strideC, int batchCount) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t total_bytes = static_cast<size_t>(m) * n * batchCount * 16;
    unsigned char *ptr = (unsigned char*)C;
    for (size_t i = 0; i < total_bytes; i++) {
        ptr[i] = std::uniform_int_distribution<int>(0, 255)(g_rng);
    }
    printf("[FakeCUBLAS] cublasZgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

// Complex TRSM
cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const void *alpha, const void *A, int lda, void *B, int ldb) {
    if (!A || !B || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t total_bytes = static_cast<size_t>(m) * n * 8;
    unsigned char *ptr = (unsigned char*)B;
    for (size_t i = 0; i < total_bytes; i++) {
        ptr[i] = std::uniform_int_distribution<int>(0, 255)(g_rng);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const void *alpha, const void *A, int lda, void *B, int ldb) {
    if (!A || !B || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t total_bytes = static_cast<size_t>(m) * n * 16;
    unsigned char *ptr = (unsigned char*)B;
    for (size_t i = 0; i < total_bytes; i++) {
        ptr[i] = std::uniform_int_distribution<int>(0, 255)(g_rng);
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Complex batched TRSM
cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const void *alpha, const void *const A[], int lda, void *const B[], int ldb, int batchCount) {
    if (!A || !B || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t matrix_bytes = static_cast<size_t>(m) * n * 8;
    for (int i = 0; i < batchCount; i++) {
        if (B[i]) {
            unsigned char *ptr = (unsigned char*)B[i];
            for (size_t j = 0; j < matrix_bytes; j++) {
                ptr[j] = std::uniform_int_distribution<int>(0, 255)(g_rng);
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const void *alpha, const void *const A[], int lda, void *const B[], int ldb, int batchCount) {
    if (!A || !B || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t matrix_bytes = static_cast<size_t>(m) * n * 16;
    for (int i = 0; i < batchCount; i++) {
        if (B[i]) {
            unsigned char *ptr = (unsigned char*)B[i];
            for (size_t j = 0; j < matrix_bytes; j++) {
                ptr[j] = std::uniform_int_distribution<int>(0, 255)(g_rng);
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Batched factorization functions
cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
    if (!Aarray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Aarray[i]) fillRandom(Aarray[i], n * n);
        if (infoArray) infoArray[i] = 0;
    }
    printf("[FakeCUBLAS] cublasSgetrfBatched n=%d batchSize=%d\n", n, batchSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
    if (!Aarray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Aarray[i]) fillRandom(Aarray[i], n * n);
        if (infoArray) infoArray[i] = 0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, void *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
    if (!Aarray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Aarray[i] && infoArray) infoArray[i] = 0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, void *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
    if (!Aarray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Aarray[i] && infoArray) infoArray[i] = 0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Batched solve functions
cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchSize) {
    if (!Aarray || !Barray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Barray[i]) fillRandom(Barray[i], n * nrhs);
    }
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *const Aarray[], int lda, const int *devIpiv, double *const Barray[], int ldb, int *info, int batchSize) {
    if (!Aarray || !Barray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Barray[i]) fillRandom(Barray[i], n * nrhs);
    }
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const void *const Aarray[], int lda, const int *devIpiv, void *const Barray[], int ldb, int *info, int batchSize) {
    if (!Aarray || !Barray) return CUBLAS_STATUS_INVALID_VALUE;
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const void *const Aarray[], int lda, const int *devIpiv, void *const Barray[], int ldb, int *info, int batchSize) {
    if (!Aarray || !Barray) return CUBLAS_STATUS_INVALID_VALUE;
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

// Batched QR factorization
cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float *const Aarray[], int lda, float *const TauArray[], int *info, int batchSize) {
    if (!Aarray || !TauArray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Aarray[i]) fillRandom(Aarray[i], m * n);
        if (TauArray[i]) fillRandom(TauArray[i], std::min(m, n));
    }
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double *const Aarray[], int lda, double *const TauArray[], int *info, int batchSize) {
    if (!Aarray || !TauArray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Aarray[i]) fillRandom(Aarray[i], m * n);
        if (TauArray[i]) fillRandom(TauArray[i], std::min(m, n));
    }
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, void *const Aarray[], int lda, void *const TauArray[], int *info, int batchSize) {
    if (!Aarray || !TauArray) return CUBLAS_STATUS_INVALID_VALUE;
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, void *const Aarray[], int lda, void *const TauArray[], int *info, int batchSize) {
    if (!Aarray || !TauArray) return CUBLAS_STATUS_INVALID_VALUE;
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

// Batched least squares
cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float *const Aarray[], int lda, float *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    if (!Aarray || !Carray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Aarray[i]) fillRandom(Aarray[i], m * n);
        if (Carray[i]) fillRandom(Carray[i], std::max(m, n) * nrhs);
        if (devInfoArray) devInfoArray[i] = 0;
    }
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double *const Aarray[], int lda, double *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    if (!Aarray || !Carray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (Aarray[i]) fillRandom(Aarray[i], m * n);
        if (Carray[i]) fillRandom(Carray[i], std::max(m, n) * nrhs);
        if (devInfoArray) devInfoArray[i] = 0;
    }
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, void *const Aarray[], int lda, void *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    if (!Aarray || !Carray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (devInfoArray) devInfoArray[i] = 0;
    }
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, void *const Aarray[], int lda, void *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    if (!Aarray || !Carray) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < batchSize; i++) {
        if (devInfoArray) devInfoArray[i] = 0;
    }
    if (info) *info = 0;
    return CUBLAS_STATUS_SUCCESS;
}


// Workspace management
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes) {
    printf("[FakeCUBLAS] cublasSetWorkspace_v2 workspaceSize=%zu\n", workspaceSizeInBytes);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetWorkspace_v2(cublasHandle_t handle, void **workspace, size_t *workspaceSizeInBytes) {
    if (!workspace || !workspaceSizeInBytes) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *workspace = nullptr;
    *workspaceSizeInBytes = 0;
    return CUBLAS_STATUS_SUCCESS;
}

// GEMMEx variants for specific types
cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const void *A, int Atype, int lda, const void *B, int Btype, int ldb, const float *beta, void *C, int Ctype, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t element_size = 4;  // Default float
    if (Ctype == 0) element_size = 2;
    if (Ctype == 2) element_size = 8;
    
    size_t total_bytes = static_cast<size_t>(m) * n * element_size;
    unsigned char *ptr = static_cast<unsigned char*>(C);
    for (size_t i = 0; i < total_bytes; i++) {
        ptr[i] = std::uniform_int_distribution<int>(0, 255)(g_rng);
    }
    printf("[FakeCUBLAS] cublasSgemmEx m=%d n=%d k=%d\n", m, n, k);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int Atype, int lda, const void *B, int Btype, int ldb, const void *beta, void *C, int Ctype, int ldc, int computeType, cublasGemmAlgo_t algo) {
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}

// ============================================================================
// cuBLASLt (Lightweight BLAS) Implementation
// ============================================================================

// Global map to track cuBLASLt handles and descriptors
static std::map<cublasLtHandle_t, void*> g_lt_handles;
static std::map<cublasLtMatmulDesc_t, void*> g_lt_matmul_descs;
static std::map<cublasLtMatrixLayout_t, void*> g_lt_matrix_layouts;
static std::map<cublasLtMatmulPreference_t, void*> g_lt_preferences;

// Handle management
cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle) {
    if (!lightHandle) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    void *handle = malloc(16);  // Dummy allocation
    *lightHandle = reinterpret_cast<cublasLtHandle_t>(handle);
    g_lt_handles[*lightHandle] = handle;

    printf("[FakeCUBLASLt] cublasLtCreate handle=%p\n", *lightHandle);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_lt_handles.find(lightHandle);
    if (it == g_lt_handles.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    free(it->second);
    g_lt_handles.erase(it);

    printf("[FakeCUBLASLt] cublasLtDestroy handle=%p\n", lightHandle);
    return CUBLAS_STATUS_SUCCESS;
}

// Matmul descriptor management
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, int computeType, int scaleType) {
    if (!matmulDesc) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    void *desc = malloc(64);  // Dummy allocation
    *matmulDesc = reinterpret_cast<cublasLtMatmulDesc_t>(desc);
    g_lt_matmul_descs[*matmulDesc] = desc;

    printf("[FakeCUBLASLt] cublasLtMatmulDescCreate desc=%p computeType=%d scaleType=%d\n", *matmulDesc, computeType, scaleType);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_lt_matmul_descs.find(matmulDesc);
    if (it == g_lt_matmul_descs.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    free(it->second);
    g_lt_matmul_descs.erase(it);

    printf("[FakeCUBLASLt] cublasLtMatmulDescDestroy desc=%p\n", matmulDesc);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, int attr, const void *buf, size_t sizeInBytes) {
    // Just return success - we don't actually use the attributes
    printf("[FakeCUBLASLt] cublasLtMatmulDescSetAttribute desc=%p attr=%d size=%zu\n", matmulDesc, attr, sizeInBytes);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, int attr, void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    if (sizeWritten) *sizeWritten = 0;
    return CUBLAS_STATUS_SUCCESS;
}

// Matrix layout management
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, int type, uint64_t rows, uint64_t cols, int64_t ld) {
    if (!matLayout) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    void *layout = malloc(64);  // Dummy allocation
    *matLayout = reinterpret_cast<cublasLtMatrixLayout_t>(layout);
    g_lt_matrix_layouts[*matLayout] = layout;

    printf("[FakeCUBLASLt] cublasLtMatrixLayoutCreate layout=%p type=%d rows=%lu cols=%lu ld=%ld\n",
           *matLayout, type, rows, cols, ld);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_lt_matrix_layouts.find(matLayout);
    if (it == g_lt_matrix_layouts.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    free(it->second);
    g_lt_matrix_layouts.erase(it);

    printf("[FakeCUBLASLt] cublasLtMatrixLayoutDestroy layout=%p\n", matLayout);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, int attr, const void *buf, size_t sizeInBytes) {
    printf("[FakeCUBLASLt] cublasLtMatrixLayoutSetAttribute layout=%p attr=%d size=%zu\n", matLayout, attr, sizeInBytes);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout, int attr, void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    if (sizeWritten) *sizeWritten = 0;
    return CUBLAS_STATUS_SUCCESS;
}

// Matmul preference management
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
    if (!pref) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    void *preference = malloc(64);  // Dummy allocation
    *pref = reinterpret_cast<cublasLtMatmulPreference_t>(preference);
    g_lt_preferences[*pref] = preference;

    printf("[FakeCUBLASLt] cublasLtMatmulPreferenceCreate pref=%p\n", *pref);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_lt_preferences.find(pref);
    if (it == g_lt_preferences.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    free(it->second);
    g_lt_preferences.erase(it);

    printf("[FakeCUBLASLt] cublasLtMatmulPreferenceDestroy pref=%p\n", pref);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, int attr, const void *buf, size_t sizeInBytes) {
    printf("[FakeCUBLASLt] cublasLtMatmulPreferenceSetAttribute pref=%p attr=%d size=%zu\n", pref, attr, sizeInBytes);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref, int attr, void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    if (sizeWritten) *sizeWritten = 0;
    return CUBLAS_STATUS_SUCCESS;
}

// Algorithm heuristic - THIS IS THE KEY FUNCTION THAT WAS FAILING
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
) {
    if (!returnAlgoCount || !heuristicResultsArray) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    // Return at least one algorithm result
    int algoCount = (requestedAlgoCount > 0) ? 1 : 0;
    if (algoCount > 0) {
        // Fill in a default algorithm
        heuristicResultsArray[0].algo = 0;  // Default algorithm
        heuristicResultsArray[0].workspaceSize = 0;  // No workspace needed
        heuristicResultsArray[0].state = 0;
        heuristicResultsArray[0].wavesCount = 1.0f;
        for (int i = 0; i < 4; i++) {
            heuristicResultsArray[0].reserved[i] = 0;
        }
    }

    *returnAlgoCount = algoCount;

    printf("[FakeCUBLASLt] cublasLtMatmulAlgoGetHeuristic requested=%d returned=%d\n",
           requestedAlgoCount, algoCount);
    return CUBLAS_STATUS_SUCCESS;
}

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
) {
    if (!A || !B || !D || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    // For now, just fill D with random values
    // In a real implementation, we'd extract matrix dimensions from the descriptors
    // and perform actual computation

    printf("[FakeCUBLASLt] cublasLtMatmul A=%p B=%p D=%p workspace=%zu\n",
           A, B, D, workspaceSizeInBytes);

    // We can't actually compute without knowing the dimensions,
    // but we return success to let PyTorch continue
    return CUBLAS_STATUS_SUCCESS;
}

