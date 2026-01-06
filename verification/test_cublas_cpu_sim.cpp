#include "../src/cublas/cublas_defs.hpp"
#include "../src/cuda/cudart_defs.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure (%s): %d\n", what, static_cast<int>(err));
    std::exit(2);
}

void check_cublas(cublasStatus_t st, const char* what) {
    if (st == CUBLAS_STATUS_SUCCESS) return;
    std::fprintf(stderr, "cuBLAS failure (%s): %d\n", what, static_cast<int>(st));
    std::exit(3);
}

float rand_f32(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(rng);
}

size_t idx_col_major(int row, int col, int ld) {
    return static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(ld);
}

size_t idx_row_major(int row, int col, int ld) {
    return static_cast<size_t>(row) * static_cast<size_t>(ld) + static_cast<size_t>(col);
}

void gemm_ref_col_major(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float beta,
    const float* C_in,
    int ldc,
    float* C_out
) {
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double acc = 0.0;
            for (int p = 0; p < k; ++p) {
                const int a_row = (transa == CUBLAS_OP_N) ? row : p;
                const int a_col = (transa == CUBLAS_OP_N) ? p : row;
                const int b_row = (transb == CUBLAS_OP_N) ? p : col;
                const int b_col = (transb == CUBLAS_OP_N) ? col : p;
                acc += static_cast<double>(A[idx_col_major(a_row, a_col, lda)]) *
                       static_cast<double>(B[idx_col_major(b_row, b_col, ldb)]);
            }
            const size_t c_idx = idx_col_major(row, col, ldc);
            const double prev = static_cast<double>(C_in[c_idx]);
            C_out[c_idx] = static_cast<float>(static_cast<double>(alpha) * acc + static_cast<double>(beta) * prev);
        }
    }
}

void matmul_ref_row_major_bias_relu(
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float beta,
    const float* C,
    int ldc,
    const float* bias, // length m
    float* D,
    int ldd
) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double acc = 0.0;
            for (int p = 0; p < k; ++p) {
                acc += static_cast<double>(A[idx_row_major(row, p, lda)]) *
                       static_cast<double>(B[idx_row_major(p, col, ldb)]);
            }
            double out = static_cast<double>(alpha) * acc + static_cast<double>(beta) * static_cast<double>(C[idx_row_major(row, col, ldc)]);
            out += static_cast<double>(bias[row]);
            if (out < 0.0) out = 0.0;
            D[idx_row_major(row, col, ldd)] = static_cast<float>(out);
        }
    }
}

bool allclose(const std::vector<float>& a, const std::vector<float>& b, float atol, float rtol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float tol = atol + rtol * std::fabs(b[i]);
        if (diff > tol || std::isnan(diff)) {
            std::fprintf(stderr, "Mismatch at %zu: got=%g expected=%g diff=%g tol=%g\n", i, a[i], b[i], diff, tol);
            return false;
        }
    }
    return true;
}

bool allclose_scalar(float got, float expected, float atol, float rtol, const char* what) {
    const float diff = std::fabs(got - expected);
    const float tol = atol + rtol * std::fabs(expected);
    if (diff > tol || std::isnan(diff)) {
        std::fprintf(stderr, "%s mismatch: got=%g expected=%g diff=%g tol=%g\n", what, got, expected, diff, tol);
        return false;
    }
    return true;
}

} // namespace

int main() {
    std::mt19937 rng(12345);

    // -------------------------
    // Test 1: cublasSgemm_v2
    // -------------------------
    {
        const int m = 7;
        const int n = 5;
        const int k = 6;
        const int lda = 9;  // padded
        const int ldb = 8;  // padded
        const int ldc = 10; // padded
        const float alpha = 1.25f;
        const float beta = -0.5f;

        std::vector<float> hA(static_cast<size_t>(lda) * k);
        std::vector<float> hB(static_cast<size_t>(ldb) * n);
        std::vector<float> hC(static_cast<size_t>(ldc) * n);

        for (auto& v : hA) v = rand_f32(rng);
        for (auto& v : hB) v = rand_f32(rng);
        for (auto& v : hC) v = rand_f32(rng);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dA), hA.size() * sizeof(float)), "cudaMalloc A");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dB), hB.size() * sizeof(float)), "cudaMalloc B");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dC), hC.size() * sizeof(float)), "cudaMalloc C");

        check_cuda(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
        check_cuda(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
        check_cuda(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D C");

        cublasHandle_t handle = nullptr;
        check_cublas(cublasCreate_v2(&handle), "cublasCreate_v2");
        check_cublas(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode_v2");

        check_cublas(
            cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc),
            "cublasSgemm_v2"
        );

        std::vector<float> got(hC.size());
        check_cuda(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

        std::vector<float> expected = hC;
        gemm_ref_col_major(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, hA.data(), lda, hB.data(), ldb, beta, hC.data(), ldc, expected.data());

        if (!allclose(got, expected, 1e-5f, 1e-5f)) {
            std::fprintf(stderr, "Test1 failed: cublasSgemm_v2 mismatch\n");
            return 1;
        }

        check_cublas(cublasDestroy_v2(handle), "cublasDestroy_v2");
        check_cuda(cudaFree(dA), "cudaFree A");
        check_cuda(cudaFree(dB), "cudaFree B");
        check_cuda(cudaFree(dC), "cudaFree C");
        std::printf("Test1 OK: cublasSgemm_v2\n");
    }

    // -------------------------
    // Test 2: cublasLtMatmul (row-major + bias + relu)
    // -------------------------
    {
        // Keep constants local to avoid pulling CUDA headers.
        constexpr int32_t CUBLASLT_MATRIX_LAYOUT_ORDER = 1;
        constexpr int32_t CUBLASLT_ORDER_ROW = 1;
        constexpr int32_t CUBLASLT_MATMUL_DESC_EPILOGUE = 7;
        constexpr int32_t CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8;
        constexpr uint32_t CUBLASLT_EPILOGUE_RELU_BIAS = 6; // RELU|BIAS

        constexpr int32_t CUDA_R_32F = 0;

        const int m = 6;
        const int n = 4;
        const int k = 5;
        const int lda = 7;
        const int ldb = 6;
        const int ldc = 5;
        const int ldd = 8;

        const float alpha = 1.0f;
        const float beta = 0.25f;

        std::vector<float> hA(static_cast<size_t>(lda) * m);
        std::vector<float> hB(static_cast<size_t>(ldb) * k);
        std::vector<float> hC(static_cast<size_t>(ldc) * m);
        std::vector<float> hBias(static_cast<size_t>(m));

        for (auto& v : hA) v = rand_f32(rng);
        for (auto& v : hB) v = rand_f32(rng);
        for (auto& v : hC) v = rand_f32(rng);
        for (auto& v : hBias) v = rand_f32(rng);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dBias = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dA), hA.size() * sizeof(float)), "cudaMalloc A");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dB), hB.size() * sizeof(float)), "cudaMalloc B");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dC), hC.size() * sizeof(float)), "cudaMalloc C");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dD), static_cast<size_t>(ldd) * m * sizeof(float)), "cudaMalloc D");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dBias), hBias.size() * sizeof(float)), "cudaMalloc Bias");

        check_cuda(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
        check_cuda(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
        check_cuda(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D C");
        check_cuda(cudaMemcpy(dBias, hBias.data(), hBias.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D Bias");
        check_cuda(cudaMemset(dD, 0, static_cast<size_t>(ldd) * m * sizeof(float)), "cudaMemset D");

        cublasLtHandle_t lt = nullptr;
        check_cublas(cublasLtCreate(&lt), "cublasLtCreate");

        cublasLtMatmulDesc_t op = nullptr;
        check_cublas(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");

        check_cublas(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &CUBLASLT_EPILOGUE_RELU_BIAS, sizeof(CUBLASLT_EPILOGUE_RELU_BIAS)),
                     "cublasLtMatmulDescSetAttribute epilogue");
        const void* bias_ptr = dBias;
        check_cublas(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)),
                     "cublasLtMatmulDescSetAttribute bias");

        cublasLtMatrixLayout_t a_desc = nullptr;
        cublasLtMatrixLayout_t b_desc = nullptr;
        cublasLtMatrixLayout_t c_desc = nullptr;
        cublasLtMatrixLayout_t d_desc = nullptr;

        // A is m x k in row-major
        check_cublas(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, static_cast<uint64_t>(m), static_cast<uint64_t>(k), lda),
                     "cublasLtMatrixLayoutCreate A");
        check_cublas(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &CUBLASLT_ORDER_ROW, sizeof(CUBLASLT_ORDER_ROW)),
                     "cublasLtMatrixLayoutSetAttribute A order");

        // B is k x n in row-major
        check_cublas(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, static_cast<uint64_t>(k), static_cast<uint64_t>(n), ldb),
                     "cublasLtMatrixLayoutCreate B");
        check_cublas(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &CUBLASLT_ORDER_ROW, sizeof(CUBLASLT_ORDER_ROW)),
                     "cublasLtMatrixLayoutSetAttribute B order");

        // C is m x n in row-major
        check_cublas(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldc),
                     "cublasLtMatrixLayoutCreate C");
        check_cublas(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &CUBLASLT_ORDER_ROW, sizeof(CUBLASLT_ORDER_ROW)),
                     "cublasLtMatrixLayoutSetAttribute C order");

        // D is m x n in row-major
        check_cublas(cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldd),
                     "cublasLtMatrixLayoutCreate D");
        check_cublas(cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &CUBLASLT_ORDER_ROW, sizeof(CUBLASLT_ORDER_ROW)),
                     "cublasLtMatrixLayoutSetAttribute D order");

        check_cublas(
            cublasLtMatmul(
                lt, op, &alpha, dA, a_desc, dB, b_desc, &beta, dC, c_desc, dD, d_desc, nullptr, nullptr, 0, nullptr
            ),
            "cublasLtMatmul"
        );

        std::vector<float> got(static_cast<size_t>(ldd) * m);
        check_cuda(cudaMemcpy(got.data(), dD, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H D");

        std::vector<float> expected(got.size(), 0.0f);
        matmul_ref_row_major_bias_relu(m, n, k, alpha, hA.data(), lda, hB.data(), ldb, beta, hC.data(), ldc, hBias.data(), expected.data(), ldd);

        if (!allclose(got, expected, 1e-5f, 1e-5f)) {
            std::fprintf(stderr, "Test2 failed: cublasLtMatmul mismatch\n");
            return 1;
        }

        check_cublas(cublasLtMatrixLayoutDestroy(a_desc), "cublasLtMatrixLayoutDestroy A");
        check_cublas(cublasLtMatrixLayoutDestroy(b_desc), "cublasLtMatrixLayoutDestroy B");
        check_cublas(cublasLtMatrixLayoutDestroy(c_desc), "cublasLtMatrixLayoutDestroy C");
        check_cublas(cublasLtMatrixLayoutDestroy(d_desc), "cublasLtMatrixLayoutDestroy D");
        check_cublas(cublasLtMatmulDescDestroy(op), "cublasLtMatmulDescDestroy");
        check_cublas(cublasLtDestroy(lt), "cublasLtDestroy");

        check_cuda(cudaFree(dA), "cudaFree A");
        check_cuda(cudaFree(dB), "cudaFree B");
        check_cuda(cudaFree(dC), "cudaFree C");
        check_cuda(cudaFree(dD), "cudaFree D");
        check_cuda(cudaFree(dBias), "cudaFree Bias");
        std::printf("Test2 OK: cublasLtMatmul (row-major + bias + relu)\n");
    }

    // -------------------------
    // Test 3: cublasSgemm_v2 (transpose + device pointer mode)
    // -------------------------
    {
        const int m = 5;
        const int n = 4;
        const int k = 3;
        const int lda = 6; // padded
        const int ldb = 5; // padded
        const int ldc = 7; // padded

        const float alpha = 0.9f;
        const float beta = 0.1f;

        // With transa=T, A is (k x m) in column-major (lda x m).
        std::vector<float> hA(static_cast<size_t>(lda) * m);
        std::vector<float> hB(static_cast<size_t>(ldb) * n);
        std::vector<float> hC(static_cast<size_t>(ldc) * n);

        for (auto& v : hA) v = rand_f32(rng);
        for (auto& v : hB) v = rand_f32(rng);
        for (auto& v : hC) v = rand_f32(rng);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        float *dAlpha = nullptr, *dBeta = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dA), hA.size() * sizeof(float)), "cudaMalloc A");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dB), hB.size() * sizeof(float)), "cudaMalloc B");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dC), hC.size() * sizeof(float)), "cudaMalloc C");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dAlpha), sizeof(float)), "cudaMalloc alpha");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dBeta), sizeof(float)), "cudaMalloc beta");

        check_cuda(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
        check_cuda(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
        check_cuda(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D C");
        check_cuda(cudaMemcpy(dAlpha, &alpha, sizeof(float), cudaMemcpyHostToDevice), "H2D alpha");
        check_cuda(cudaMemcpy(dBeta, &beta, sizeof(float), cudaMemcpyHostToDevice), "H2D beta");

        cublasHandle_t handle = nullptr;
        check_cublas(cublasCreate_v2(&handle), "cublasCreate_v2");
        check_cublas(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode_v2");

        check_cublas(
            cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, dAlpha, dA, lda, dB, ldb, dBeta, dC, ldc),
            "cublasSgemm_v2"
        );

        std::vector<float> got(hC.size());
        check_cuda(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

        std::vector<float> expected = hC;
        gemm_ref_col_major(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, hA.data(), lda, hB.data(), ldb, beta, hC.data(), ldc, expected.data());

        if (!allclose(got, expected, 1e-5f, 1e-5f)) {
            std::fprintf(stderr, "Test3 failed: cublasSgemm_v2 mismatch (transpose + device pointer mode)\n");
            return 1;
        }

        check_cublas(cublasDestroy_v2(handle), "cublasDestroy_v2");
        check_cuda(cudaFree(dA), "cudaFree A");
        check_cuda(cudaFree(dB), "cudaFree B");
        check_cuda(cudaFree(dC), "cudaFree C");
        check_cuda(cudaFree(dAlpha), "cudaFree alpha");
        check_cuda(cudaFree(dBeta), "cudaFree beta");
        std::printf("Test3 OK: cublasSgemm_v2 (transpose + device pointer mode)\n");
    }

    // -------------------------
    // Test 4: cublasSgemmStridedBatched
    // -------------------------
    {
        const int m = 4;
        const int n = 3;
        const int k = 5;
        const int lda = 6; // padded
        const int ldb = 4; // padded
        const int ldc = 5; // padded
        const int batchCount = 3;

        const long long int strideA = static_cast<long long>(lda) * k + 13;
        const long long int strideB = static_cast<long long>(ldb) * k + 9;   // transb=T => B is (n x k), stored as (ldb x k)
        const long long int strideC = static_cast<long long>(ldc) * n + 7;

        const float alpha = -0.75f;
        const float beta = 0.5f;

        std::vector<float> hA(static_cast<size_t>(strideA) * batchCount);
        std::vector<float> hB(static_cast<size_t>(strideB) * batchCount);
        std::vector<float> hC(static_cast<size_t>(strideC) * batchCount);

        for (auto& v : hA) v = rand_f32(rng);
        for (auto& v : hB) v = rand_f32(rng);
        for (auto& v : hC) v = rand_f32(rng);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dA), hA.size() * sizeof(float)), "cudaMalloc A");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dB), hB.size() * sizeof(float)), "cudaMalloc B");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dC), hC.size() * sizeof(float)), "cudaMalloc C");

        check_cuda(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
        check_cuda(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
        check_cuda(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D C");

        cublasHandle_t handle = nullptr;
        check_cublas(cublasCreate_v2(&handle), "cublasCreate_v2");
        check_cublas(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode_v2");

        check_cublas(
            cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                m,
                n,
                k,
                &alpha,
                dA,
                lda,
                strideA,
                dB,
                ldb,
                strideB,
                &beta,
                dC,
                ldc,
                strideC,
                batchCount
            ),
            "cublasSgemmStridedBatched"
        );

        std::vector<float> got(hC.size());
        check_cuda(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

        std::vector<float> expected = hC;
        for (int batch = 0; batch < batchCount; ++batch) {
            const float* Ab = hA.data() + static_cast<size_t>(batch) * static_cast<size_t>(strideA);
            const float* Bb = hB.data() + static_cast<size_t>(batch) * static_cast<size_t>(strideB);
            const float* Cin = hC.data() + static_cast<size_t>(batch) * static_cast<size_t>(strideC);
            float* Cout = expected.data() + static_cast<size_t>(batch) * static_cast<size_t>(strideC);
            gemm_ref_col_major(CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, Ab, lda, Bb, ldb, beta, Cin, ldc, Cout);
        }

        if (!allclose(got, expected, 1e-5f, 1e-5f)) {
            std::fprintf(stderr, "Test4 failed: cublasSgemmStridedBatched mismatch\n");
            return 1;
        }

        check_cublas(cublasDestroy_v2(handle), "cublasDestroy_v2");
        check_cuda(cudaFree(dA), "cudaFree A");
        check_cuda(cudaFree(dB), "cudaFree B");
        check_cuda(cudaFree(dC), "cudaFree C");
        std::printf("Test4 OK: cublasSgemmStridedBatched\n");
    }

    // -------------------------
    // Test 5: cublasSgemmBatched
    // -------------------------
    {
        const int m = 3;
        const int n = 4;
        const int k = 2;
        const int lda = 5; // padded
        const int ldb = 4; // padded
        const int ldc = 6; // padded
        const int batchCount = 2;

        const float alpha = 1.0f;
        const float beta = -0.25f;

        std::vector<std::vector<float>> hA(batchCount);
        std::vector<std::vector<float>> hB(batchCount);
        std::vector<std::vector<float>> hC(batchCount);
        std::vector<std::vector<float>> expected(batchCount);

        for (int batch = 0; batch < batchCount; ++batch) {
            hA[batch].resize(static_cast<size_t>(lda) * k);
            hB[batch].resize(static_cast<size_t>(ldb) * n);
            hC[batch].resize(static_cast<size_t>(ldc) * n);

            for (auto& v : hA[batch]) v = rand_f32(rng);
            for (auto& v : hB[batch]) v = rand_f32(rng);
            for (auto& v : hC[batch]) v = rand_f32(rng);

            expected[batch] = hC[batch];
            gemm_ref_col_major(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, hA[batch].data(), lda, hB[batch].data(), ldb, beta, hC[batch].data(), ldc, expected[batch].data());
        }

        std::vector<float*> dA(batchCount);
        std::vector<float*> dB(batchCount);
        std::vector<float*> dC(batchCount);
        for (int batch = 0; batch < batchCount; ++batch) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&dA[batch]), hA[batch].size() * sizeof(float)), "cudaMalloc A");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&dB[batch]), hB[batch].size() * sizeof(float)), "cudaMalloc B");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&dC[batch]), hC[batch].size() * sizeof(float)), "cudaMalloc C");

            check_cuda(cudaMemcpy(dA[batch], hA[batch].data(), hA[batch].size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
            check_cuda(cudaMemcpy(dB[batch], hB[batch].data(), hB[batch].size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
            check_cuda(cudaMemcpy(dC[batch], hC[batch].data(), hC[batch].size() * sizeof(float), cudaMemcpyHostToDevice), "H2D C");
        }

        std::vector<const float*> Aarray(batchCount);
        std::vector<const float*> Barray(batchCount);
        std::vector<float*> Carray(batchCount);
        for (int batch = 0; batch < batchCount; ++batch) {
            Aarray[batch] = dA[batch];
            Barray[batch] = dB[batch];
            Carray[batch] = dC[batch];
        }

        cublasHandle_t handle = nullptr;
        check_cublas(cublasCreate_v2(&handle), "cublasCreate_v2");
        check_cublas(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode_v2");

        check_cublas(
            cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, Aarray.data(), lda, Barray.data(), ldb, &beta, Carray.data(), ldc, batchCount),
            "cublasSgemmBatched"
        );

        for (int batch = 0; batch < batchCount; ++batch) {
            std::vector<float> got(hC[batch].size());
            check_cuda(cudaMemcpy(got.data(), dC[batch], got.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");
            if (!allclose(got, expected[batch], 1e-5f, 1e-5f)) {
                std::fprintf(stderr, "Test5 failed: cublasSgemmBatched mismatch at batch=%d\n", batch);
                return 1;
            }
        }

        check_cublas(cublasDestroy_v2(handle), "cublasDestroy_v2");
        for (int batch = 0; batch < batchCount; ++batch) {
            check_cuda(cudaFree(dA[batch]), "cudaFree A");
            check_cuda(cudaFree(dB[batch]), "cudaFree B");
            check_cuda(cudaFree(dC[batch]), "cudaFree C");
        }
        std::printf("Test5 OK: cublasSgemmBatched\n");
    }

    // -------------------------
    // Test 6: cuBLAS BLAS1 ops (axpy, dot, nrm2, asum, isamax)
    // -------------------------
    {
        const int n = 9;
        std::vector<float> hx(static_cast<size_t>(n));
        std::vector<float> hy(static_cast<size_t>(n));
        for (auto& v : hx) v = rand_f32(rng);
        for (auto& v : hy) v = rand_f32(rng);

        float* dX = nullptr;
        float* dY = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dX), hx.size() * sizeof(float)), "cudaMalloc X");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dY), hy.size() * sizeof(float)), "cudaMalloc Y");
        check_cuda(cudaMemcpy(dX, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D X");
        check_cuda(cudaMemcpy(dY, hy.data(), hy.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D Y");

        cublasHandle_t handle = nullptr;
        check_cublas(cublasCreate_v2(&handle), "cublasCreate_v2");
        check_cublas(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode_v2");

        // saxpy: y[i] = a * x[n-1-i] + y[i] (incx=-1, incy=1)
        const float a = 1.5f;
        {
            std::vector<float> expected = hy;
            for (int i = 0; i < n; ++i) {
                expected[static_cast<size_t>(i)] = a * hx[static_cast<size_t>(n - 1 - i)] + expected[static_cast<size_t>(i)];
            }
            check_cublas(cublasSaxpy_v2(handle, n, &a, dX, -1, dY, 1), "cublasSaxpy_v2");

            std::vector<float> got(hy.size());
            check_cuda(cudaMemcpy(got.data(), dY, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H Y");
            if (!allclose(got, expected, 1e-5f, 1e-5f)) {
                std::fprintf(stderr, "Test6 failed: cublasSaxpy_v2 mismatch\n");
                return 1;
            }
        }

        // dot: sum x[i] * y[n-1-i] (incy=-1)
        {
            // Reset Y so this test is independent of the previous axpy.
            check_cuda(cudaMemcpy(dY, hy.data(), hy.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D Y (dot)");

            float expected = 0.0f;
            for (int i = 0; i < n; ++i) {
                expected += hx[static_cast<size_t>(i)] * hy[static_cast<size_t>(n - 1 - i)];
            }
            float got = 0.0f;
            check_cublas(cublasSdot_v2(handle, n, dX, 1, dY, -1, &got), "cublasSdot_v2");
            if (!allclose_scalar(got, expected, 1e-5f, 1e-5f, "cublasSdot_v2")) {
                std::fprintf(stderr, "Test6 failed: cublasSdot_v2 mismatch\n");
                return 1;
            }
        }

        // nrm2: sqrt(sum x[i]^2) (incx=-1)
        {
            float expected = 0.0f;
            for (int i = 0; i < n; ++i) {
                expected += hx[static_cast<size_t>(i)] * hx[static_cast<size_t>(i)];
            }
            expected = std::sqrt(expected);
            float got = 0.0f;
            check_cublas(cublasSnrm2_v2(handle, n, dX, -1, &got), "cublasSnrm2_v2");
            if (!allclose_scalar(got, expected, 1e-5f, 1e-5f, "cublasSnrm2_v2")) {
                std::fprintf(stderr, "Test6 failed: cublasSnrm2_v2 mismatch\n");
                return 1;
            }
        }

        // asum: sum abs(x[i]) (incx=-1)
        {
            float expected = 0.0f;
            for (int i = 0; i < n; ++i) {
                expected += std::fabs(hx[static_cast<size_t>(i)]);
            }
            float got = 0.0f;
            check_cublas(cublasSasum_v2(handle, n, dX, -1, &got), "cublasSasum_v2");
            if (!allclose_scalar(got, expected, 1e-5f, 1e-5f, "cublasSasum_v2")) {
                std::fprintf(stderr, "Test6 failed: cublasSasum_v2 mismatch\n");
                return 1;
            }
        }

        // isamax: 1-based index of max abs element
        {
            std::vector<float> hx2 = hx;
            hx2[2] = 100.0f; // ensure unique max abs
            check_cuda(cudaMemcpy(dX, hx2.data(), hx2.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D X (isamax)");

            int got = 0;
            check_cublas(cublasIsamax_v2(handle, n, dX, 1, &got), "cublasIsamax_v2");
            const int expected = 3; // 1-based index
            if (got != expected) {
                std::fprintf(stderr, "Test6 failed: cublasIsamax_v2 mismatch: got=%d expected=%d\n", got, expected);
                return 1;
            }
        }

        check_cublas(cublasDestroy_v2(handle), "cublasDestroy_v2");
        check_cuda(cudaFree(dX), "cudaFree X");
        check_cuda(cudaFree(dY), "cudaFree Y");
        std::printf("Test6 OK: BLAS1 ops (axpy/dot/nrm2/asum/isamax)\n");
    }

    std::printf("All FakeGPU CPU simulation tests passed.\n");
    return 0;
}
