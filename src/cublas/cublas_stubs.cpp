#include "cublas_defs.hpp"
#include "../core/global_state.hpp"
#include "../core/logging.hpp"
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <new>
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

#ifndef FAKEGPU_CPU_SIMULATION
#define FAKEGPU_CPU_SIMULATION 0
#endif

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

// Map cudaDataType_t enum values to element sizes (in bytes)
static size_t getDataTypeSize(int dataType) {
    switch (dataType) {
        case 0: return 4;   // CUDA_R_32F
        case 1: return 8;   // CUDA_R_64F
        case 2: return 2;   // CUDA_R_16F
        case 3: return 1;   // CUDA_R_8I
        case 4: return 8;   // CUDA_C_32F (2x float)
        case 5: return 16;  // CUDA_C_64F (2x double)
        case 6: return 4;   // CUDA_C_16F (2x half)
        case 7: return 2;   // CUDA_C_8I  (2x int8)
        case 8: return 1;   // CUDA_R_8U
        case 9: return 2;   // CUDA_C_8U  (2x uint8)
        case 10: return 4;  // CUDA_R_32I
        case 11: return 8;  // CUDA_C_32I
        case 12: return 4;  // CUDA_R_32U
        case 13: return 8;  // CUDA_C_32U
        case 14: return 2;  // CUDA_R_16BF
        case 15: return 4;  // CUDA_C_16BF
        case 16: case 17:   // 4-bit types - treat as 1 byte per element
        case 18: case 19:
            return 1;
        case 20: return 2;  // CUDA_R_16I
        case 21: return 4;  // CUDA_C_16I
        case 22: return 2;  // CUDA_R_16U
        case 23: return 4;  // CUDA_C_16U
        case 24: return 8;  // CUDA_R_64I
        case 25: return 16; // CUDA_C_64I
        case 26: return 8;  // CUDA_R_64U
        case 27: return 16; // CUDA_C_64U
        case 28: case 29: case 30: // FP8/FP6/FP4 variants
        case 31: case 32: case 33:
            return 1;
        default:
            return 4; // fallback to float size
    }
}

namespace {

uint64_t clamp_u128_to_u64(__int128 value) {
    if (value <= 0) return 0;
    const __int128 max_u64 = static_cast<__int128>(std::numeric_limits<uint64_t>::max());
    if (value >= max_u64) return std::numeric_limits<uint64_t>::max();
    return static_cast<uint64_t>(value);
}

uint64_t gemm_flops_u64(uint64_t m, uint64_t n, uint64_t k, uint64_t batch_count, bool complex = false) {
    if (m == 0 || n == 0 || k == 0 || batch_count == 0) return 0;
    const uint64_t factor = complex ? 8ull : 2ull;
    __int128 flops = static_cast<__int128>(m);
    flops *= static_cast<__int128>(n);
    flops *= static_cast<__int128>(k);
    flops *= static_cast<__int128>(batch_count);
    flops *= static_cast<__int128>(factor);
    return clamp_u128_to_u64(flops);
}

bool ascii_iequals(const char* a, const char* b) {
    if (!a || !b) return false;
    while (*a && *b) {
        const unsigned char ca = static_cast<unsigned char>(*a);
        const unsigned char cb = static_cast<unsigned char>(*b);
        if (std::tolower(ca) != std::tolower(cb)) return false;
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

enum class CpuGemmSimMode : uint8_t {
    kAuto = 0,
    kExact = 1,
    kSkip = 2,
};

CpuGemmSimMode cpu_gemm_sim_mode() {
    static CpuGemmSimMode mode = [] {
        const char* env = std::getenv("FAKEGPU_CPU_GEMM_MODE");
        if (!env || !*env) return CpuGemmSimMode::kAuto;
        if (ascii_iequals(env, "exact")) return CpuGemmSimMode::kExact;
        if (ascii_iequals(env, "skip") || ascii_iequals(env, "fast")) return CpuGemmSimMode::kSkip;
        return CpuGemmSimMode::kAuto;
    }();
    return mode;
}

uint64_t cpu_gemm_max_ops() {
    static uint64_t max_ops = [] {
        constexpr uint64_t kDefaultMaxOps = 5'000'000ull;
        const char* env = std::getenv("FAKEGPU_CPU_GEMM_MAX_OPS");
        if (!env || !*env) return kDefaultMaxOps;
        errno = 0;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (errno != 0 || end == env || !end || *end != '\0') return kDefaultMaxOps;
        return static_cast<uint64_t>(parsed);
    }();
    return max_ops;
}

bool should_skip_cpu_gemm(uint64_t m, uint64_t n, uint64_t k, uint64_t batch_count) {
    const CpuGemmSimMode mode = cpu_gemm_sim_mode();
    if (mode == CpuGemmSimMode::kExact) return false;
    if (mode == CpuGemmSimMode::kSkip) return true;

    const uint64_t max_ops = cpu_gemm_max_ops();
    if (max_ops == 0) return false;

    __int128 ops = static_cast<__int128>(m);
    ops *= static_cast<__int128>(n);
    ops *= static_cast<__int128>(k);
    ops *= static_cast<__int128>(batch_count);
    return ops > static_cast<__int128>(max_ops);
}

const void* first_nonnull_ptr(const void* const* ptrs, int count) {
    if (!ptrs || count <= 0) return nullptr;
    for (int i = 0; i < count; ++i) {
        if (ptrs[i]) return ptrs[i];
    }
    return nullptr;
}

// Best-effort bounds checking against FakeGPU's tracked allocations.
bool ensure_allocation_at_least(const void* ptr, size_t required_bytes) {
    if (!ptr || required_bytes == 0) return true;
    size_t size = 0;
    int device = -1;
    if (!fake_gpu::GlobalState::instance().get_allocation_info(const_cast<void*>(ptr), size, device)) {
        return true;
    }
    return size >= required_bytes;
}

float half_to_float(__half h) {
    const uint16_t x = h.x;
    const uint16_t sign = (x >> 15) & 0x1;
    const uint16_t exp = (x >> 10) & 0x1F;
    const uint16_t mant = x & 0x3FF;

    uint32_t out_sign = static_cast<uint32_t>(sign) << 31;
    uint32_t out_exp = 0;
    uint32_t out_mant = 0;

    if (exp == 0) {
        if (mant == 0) {
            out_exp = 0;
            out_mant = 0;
        } else {
            // Subnormal half -> normalized float
            int shift = 0;
            uint16_t m = mant;
            while ((m & 0x400) == 0) {
                m <<= 1;
                shift++;
            }
            m &= 0x3FF;
            const int32_t exp32 = (127 - 15) - shift;
            out_exp = static_cast<uint32_t>(exp32) << 23;
            out_mant = static_cast<uint32_t>(m) << 13;
        }
    } else if (exp == 0x1F) {
        // Inf/NaN
        out_exp = 0xFFu << 23;
        out_mant = static_cast<uint32_t>(mant) << 13;
        if (out_mant == 0) {
            // Inf
        } else {
            // Ensure it's a quiet NaN
            out_mant |= 0x400000u;
        }
    } else {
        const int32_t exp32 = static_cast<int32_t>(exp) + (127 - 15);
        out_exp = static_cast<uint32_t>(exp32) << 23;
        out_mant = static_cast<uint32_t>(mant) << 13;
    }

    const uint32_t bits = out_sign | out_exp | out_mant;
    float out;
    static_assert(sizeof(out) == sizeof(bits), "float size unexpected");
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

__half float_to_half(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF);
    uint32_t mant = bits & 0x7FFFFF;

    __half out{};
    if (exp == 0xFF) {
        // Inf/NaN
        uint16_t out_exp = 0x1F;
        uint16_t out_mant = (mant != 0) ? 0x200 : 0;
        out.x = static_cast<uint16_t>((sign << 15) | (out_exp << 10) | out_mant);
        return out;
    }

    exp = exp - 127 + 15;
    if (exp >= 0x1F) {
        // Overflow -> Inf
        out.x = static_cast<uint16_t>((sign << 15) | (0x1F << 10));
        return out;
    }
    if (exp <= 0) {
        // Underflow -> subnormal/zero
        if (exp < -10) {
            out.x = static_cast<uint16_t>(sign << 15);
            return out;
        }
        mant |= 0x800000; // implicit leading 1
        const int shift = 1 - exp;
        uint32_t mant_half = mant >> (shift + 13);
        // Round to nearest even
        const uint32_t round_bit = (mant >> (shift + 12)) & 0x1;
        const uint32_t sticky = mant & ((1u << (shift + 12)) - 1u);
        if (round_bit && (sticky || (mant_half & 0x1))) mant_half++;
        out.x = static_cast<uint16_t>((sign << 15) | static_cast<uint16_t>(mant_half));
        return out;
    }

    // Normal
    uint32_t mant_half = mant >> 13;
    const uint32_t round_bit = (mant >> 12) & 0x1;
    const uint32_t sticky = mant & 0xFFF;
    if (round_bit && (sticky || (mant_half & 0x1))) mant_half++;
    if (mant_half == 0x400) {
        // mantissa overflow
        mant_half = 0;
        exp++;
        if (exp >= 0x1F) {
            out.x = static_cast<uint16_t>((sign << 15) | (0x1F << 10));
            return out;
        }
    }
    out.x = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | (mant_half & 0x3FF));
    return out;
}

float bf16_to_float(uint16_t bf16) {
    const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t float_to_bf16(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    // Round-to-nearest-even on truncation.
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

double read_scalar_typed(const void* ptr, int dataType) {
    if (!ptr) return 0.0;
    switch (dataType) {
        case 0:  // CUDA_R_32F
            return static_cast<double>(*reinterpret_cast<const float*>(ptr));
        case 1:  // CUDA_R_64F
            return *reinterpret_cast<const double*>(ptr);
        case 2: { // CUDA_R_16F
            const __half h = *reinterpret_cast<const __half*>(ptr);
            return static_cast<double>(half_to_float(h));
        }
        case 14: { // CUDA_R_16BF
            const uint16_t v = *reinterpret_cast<const uint16_t*>(ptr);
            return static_cast<double>(bf16_to_float(v));
        }
        default:
            return static_cast<double>(*reinterpret_cast<const float*>(ptr));
    }
}

double read_elem(const void* base, int dataType, size_t idx) {
    const char* p = reinterpret_cast<const char*>(base) + idx * getDataTypeSize(dataType);
    return read_scalar_typed(p, dataType);
}

void write_elem(void* base, int dataType, size_t idx, double value) {
    char* p = reinterpret_cast<char*>(base) + idx * getDataTypeSize(dataType);
    switch (dataType) {
        case 0: { // CUDA_R_32F
            float out = static_cast<float>(value);
            std::memcpy(p, &out, sizeof(out));
            return;
        }
        case 1: { // CUDA_R_64F
            double out = value;
            std::memcpy(p, &out, sizeof(out));
            return;
        }
        case 2: { // CUDA_R_16F
            __half out = float_to_half(static_cast<float>(value));
            std::memcpy(p, &out, sizeof(out));
            return;
        }
        case 14: { // CUDA_R_16BF
            const uint16_t out = float_to_bf16(static_cast<float>(value));
            std::memcpy(p, &out, sizeof(out));
            return;
        }
        default: {
            float out = static_cast<float>(value);
            std::memcpy(p, &out, sizeof(out));
            return;
        }
    }
}

// Column-major GEMM kernel (best-effort correctness; no SIMD/BLAS).
// Computes: C = alpha * op(A) * op(B) + beta * C
// A,B,C use column-major with leading dimensions lda/ldb/ldc.
void gemm_col_major(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    double alpha,
    const void* A,
    int Atype,
    int lda,
    const void* B,
    int Btype,
    int ldb,
    double beta,
    void* C,
    int Ctype,
    int ldc
) {
    if (m <= 0 || n <= 0) return;
    if (should_skip_cpu_gemm(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1)) {
        const size_t elem_size = getDataTypeSize(Ctype);
        const size_t c_max_idx = static_cast<size_t>(m - 1) + static_cast<size_t>(n - 1) * static_cast<size_t>(ldc);
        std::memset(C, 0, (c_max_idx + 1) * elem_size);
        return;
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double acc = 0.0;
            for (int p = 0; p < k; ++p) {
                const int a_row = (transa == CUBLAS_OP_N) ? row : p;
                const int a_col = (transa == CUBLAS_OP_N) ? p : row;
                const int b_row = (transb == CUBLAS_OP_N) ? p : col;
                const int b_col = (transb == CUBLAS_OP_N) ? col : p;

                const size_t a_idx = static_cast<size_t>(a_row) + static_cast<size_t>(a_col) * static_cast<size_t>(lda);
                const size_t b_idx = static_cast<size_t>(b_row) + static_cast<size_t>(b_col) * static_cast<size_t>(ldb);
                acc += read_elem(A, Atype, a_idx) * read_elem(B, Btype, b_idx);
            }
            const size_t c_idx = static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(ldc);
            const double c_prev = read_elem(C, Ctype, c_idx);
            const double out = alpha * acc + beta * c_prev;
            write_elem(C, Ctype, c_idx, out);
        }
    }
}

double gelu(double x) {
    // Exact GELU (erf-based) to prioritize correctness over speed.
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

bool is_supported_gemm_datatype(int dataType) {
    switch (dataType) {
        case 0:  // CUDA_R_32F
        case 1:  // CUDA_R_64F
        case 2:  // CUDA_R_16F
        case 14: // CUDA_R_16BF
            return true;
        default:
            return false;
    }
}

} // namespace

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

    FGPU_LOG("[FakeCUBLAS] cublasCreate_v2 handle=%p\n", *handle);
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

    FGPU_LOG("[FakeCUBLAS] cublasDestroy_v2 handle=%p\n", handle);
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
    FGPU_LOG("[FakeCUBLAS] cublasGetVersion_v2 returning 12000\n");
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
    FGPU_LOG("[FakeCUBLAS] cublasSetStream_v2 stream=%p\n", streamId);
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
    FGPU_LOG("[FakeCUBLAS] cublasSetPointerMode_v2 mode=%d\n", mode);
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
    FGPU_LOG("[FakeCUBLAS] cublasSetMathMode mode=%d\n", mode);
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
    FGPU_LOG("[FakeCUBLAS] cublasSetAtomicsMode mode=%d\n", mode);
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
#if FAKEGPU_CPU_SIMULATION
    if (!x || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float* x0 = x;
    if (incx < 0) {
        x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    }
    int best = 0;
    float best_val = std::fabs(x0[0]);
    for (int i = 1; i < n; ++i) {
        const float v = std::fabs(x0[static_cast<ptrdiff_t>(i) * incx]);
        if (v > best_val) {
            best_val = v;
            best = i;
        }
    }
    // cuBLAS returns 1-based index.
    *result = best + 1;
    FGPU_LOG("[FakeCUBLAS] cublasIsamax_v2 n=%d result=%d (cpu)\n", n, *result);
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_int_distribution<int>(0, n-1)(g_rng);
    FGPU_LOG("[FakeCUBLAS] cublasIsamax_v2 n=%d result=%d\n", n, *result);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!x || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double* x0 = x;
    if (incx < 0) {
        x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    }
    int best = 0;
    double best_val = std::fabs(x0[0]);
    for (int i = 1; i < n; ++i) {
        const double v = std::fabs(x0[static_cast<ptrdiff_t>(i) * incx]);
        if (v > best_val) {
            best_val = v;
            best = i;
        }
    }
    *result = best + 1;
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_int_distribution<int>(0, n-1)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasIsamin_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!x || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float* x0 = x;
    if (incx < 0) {
        x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    }
    int best = 0;
    float best_val = std::fabs(x0[0]);
    for (int i = 1; i < n; ++i) {
        const float v = std::fabs(x0[static_cast<ptrdiff_t>(i) * incx]);
        if (v < best_val) {
            best_val = v;
            best = i;
        }
    }
    *result = best + 1;
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_int_distribution<int>(0, n-1)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasIdamin_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!x || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double* x0 = x;
    if (incx < 0) {
        x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    }
    int best = 0;
    double best_val = std::fabs(x0[0]);
    for (int i = 1; i < n; ++i) {
        const double v = std::fabs(x0[static_cast<ptrdiff_t>(i) * incx]);
        if (v < best_val) {
            best_val = v;
            best = i;
        }
    }
    *result = best + 1;
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_int_distribution<int>(0, n-1)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!x || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float* x0 = x;
    if (incx < 0) {
        x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::fabs(static_cast<double>(x0[static_cast<ptrdiff_t>(i) * incx]));
    }
    *result = static_cast<float>(sum);
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_real_distribution<float>(0.0f, static_cast<float>(n))(g_rng);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!x || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double* x0 = x;
    if (incx < 0) {
        x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::fabs(x0[static_cast<ptrdiff_t>(i) * incx]);
    }
    *result = sum;
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_real_distribution<double>(0.0, static_cast<double>(n))(g_rng);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
    if (n <= 0 || !alpha || !x || !y) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_handles.find(handle);
        if (it != g_handles.end()) {
            mode = it->second->pointerMode;
        }
    }
    const float a = getScalar(alpha, mode);
    const float* x0 = x;
    float* y0 = y;
    if (incx < 0) x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    if (incy < 0) y0 = y + static_cast<ptrdiff_t>((1 - n) * incy);
    for (int i = 0; i < n; ++i) {
        y0[static_cast<ptrdiff_t>(i) * incy] = a * x0[static_cast<ptrdiff_t>(i) * incx] + y0[static_cast<ptrdiff_t>(i) * incy];
    }
    FGPU_LOG("[FakeCUBLAS] cublasSaxpy_v2 n=%d (cpu)\n", n);
    return CUBLAS_STATUS_SUCCESS;
#else
    // y = alpha*x + y - just fill with random values
    fillRandom(y, n);
    FGPU_LOG("[FakeCUBLAS] cublasSaxpy_v2 n=%d\n", n);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy) {
    if (n <= 0 || !alpha || !x || !y) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_handles.find(handle);
        if (it != g_handles.end()) {
            mode = it->second->pointerMode;
        }
    }
    const double a = getScalar(alpha, mode);
    const double* x0 = x;
    double* y0 = y;
    if (incx < 0) x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    if (incy < 0) y0 = y + static_cast<ptrdiff_t>((1 - n) * incy);
    for (int i = 0; i < n; ++i) {
        y0[static_cast<ptrdiff_t>(i) * incy] = a * x0[static_cast<ptrdiff_t>(i) * incx] + y0[static_cast<ptrdiff_t>(i) * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    fillRandom(y, n);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy) {
    if (n <= 0 || !x || !y) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float* x0 = x;
    float* y0 = y;
    if (incx < 0) x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    if (incy < 0) y0 = y + static_cast<ptrdiff_t>((1 - n) * incy);
    for (int i = 0; i < n; ++i) {
        y0[static_cast<ptrdiff_t>(i) * incy] = x0[static_cast<ptrdiff_t>(i) * incx];
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    fillRandom(y, n);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy) {
    if (n <= 0 || !x || !y) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double* x0 = x;
    double* y0 = y;
    if (incx < 0) x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    if (incy < 0) y0 = y + static_cast<ptrdiff_t>((1 - n) * incy);
    for (int i = 0; i < n; ++i) {
        y0[static_cast<ptrdiff_t>(i) * incy] = x0[static_cast<ptrdiff_t>(i) * incx];
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    fillRandom(y, n);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!x || !y || incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float* x0 = x;
    const float* y0 = y;
    if (incx < 0) x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    if (incy < 0) y0 = y + static_cast<ptrdiff_t>((1 - n) * incy);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += static_cast<double>(x0[static_cast<ptrdiff_t>(i) * incx]) * static_cast<double>(y0[static_cast<ptrdiff_t>(i) * incy]);
    }
    *result = static_cast<float>(acc);
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_real_distribution<float>(-10.0f, 10.0f)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!x || !y || incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double* x0 = x;
    const double* y0 = y;
    if (incx < 0) x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    if (incy < 0) y0 = y + static_cast<ptrdiff_t>((1 - n) * incy);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += x0[static_cast<ptrdiff_t>(i) * incx] * y0[static_cast<ptrdiff_t>(i) * incy];
    }
    *result = acc;
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_real_distribution<double>(-10.0, 10.0)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!x || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float* x0 = x;
    if (incx < 0) x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(x0[static_cast<ptrdiff_t>(i) * incx]);
        acc += v * v;
    }
    *result = static_cast<float>(std::sqrt(acc));
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_real_distribution<float>(0.0f, 10.0f)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
    if (!result || n <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!x || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double* x0 = x;
    if (incx < 0) x0 = x + static_cast<ptrdiff_t>((1 - n) * incx);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = x0[static_cast<ptrdiff_t>(i) * incx];
        acc += v * v;
    }
    *result = std::sqrt(acc);
    return CUBLAS_STATUS_SUCCESS;
#else
    *result = std::uniform_real_distribution<double>(0.0, 10.0)(g_rng);
    return CUBLAS_STATUS_SUCCESS;
#endif
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

    FGPU_LOG("[FakeCUBLAS] cublasSgemv_v2 trans=%d m=%d n=%d\n", trans, m, n);
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

#if FAKEGPU_CPU_SIMULATION
    if (m < 0 || n < 0 || k < 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_handles.find(handle);
        if (it != g_handles.end()) {
            mode = it->second->pointerMode;
        }
    }
    const double a = static_cast<double>(getScalar(alpha, mode));
    const double b = static_cast<double>(getScalar(beta, mode));

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int a_cols = (transa == CUBLAS_OP_N) ? k : m;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    const int b_cols = (transb == CUBLAS_OP_N) ? n : k;
    if (a_rows > lda || b_rows > ldb || m > ldc) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const size_t a_max_idx = static_cast<size_t>(a_rows - 1) + static_cast<size_t>(a_cols - 1) * static_cast<size_t>(lda);
    const size_t b_max_idx = static_cast<size_t>(b_rows - 1) + static_cast<size_t>(b_cols - 1) * static_cast<size_t>(ldb);
    const size_t c_max_idx = static_cast<size_t>(m - 1) + static_cast<size_t>(n - 1) * static_cast<size_t>(ldc);
    if (!ensure_allocation_at_least(A, (a_max_idx + 1) * sizeof(float)) ||
        !ensure_allocation_at_least(B, (b_max_idx + 1) * sizeof(float)) ||
        !ensure_allocation_at_least(C, (c_max_idx + 1) * sizeof(float))) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    gemm_col_major(transa, transb, m, n, k, a, A, 0, lda, B, 0, ldb, b, C, 0, ldc);
    FGPU_LOG("[FakeCUBLAS] cublasSgemm_v2 m=%d n=%d k=%d (cpu)\n", m, n, k);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    // C is m x n - fill with random values
    size_t total_elements = static_cast<size_t>(m) * n;
    fillRandom(C, total_elements);

    FGPU_LOG("[FakeCUBLAS] cublasSgemm_v2 m=%d n=%d k=%d (output %zu elements)\n", m, n, k, total_elements);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

#if FAKEGPU_CPU_SIMULATION
    if (m < 0 || n < 0 || k < 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_handles.find(handle);
        if (it != g_handles.end()) {
            mode = it->second->pointerMode;
        }
    }
    const double a = static_cast<double>(getScalar(alpha, mode));
    const double b = static_cast<double>(getScalar(beta, mode));

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int a_cols = (transa == CUBLAS_OP_N) ? k : m;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    const int b_cols = (transb == CUBLAS_OP_N) ? n : k;
    if (a_rows > lda || b_rows > ldb || m > ldc) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const size_t a_max_idx = static_cast<size_t>(a_rows - 1) + static_cast<size_t>(a_cols - 1) * static_cast<size_t>(lda);
    const size_t b_max_idx = static_cast<size_t>(b_rows - 1) + static_cast<size_t>(b_cols - 1) * static_cast<size_t>(ldb);
    const size_t c_max_idx = static_cast<size_t>(m - 1) + static_cast<size_t>(n - 1) * static_cast<size_t>(ldc);
    if (!ensure_allocation_at_least(A, (a_max_idx + 1) * sizeof(double)) ||
        !ensure_allocation_at_least(B, (b_max_idx + 1) * sizeof(double)) ||
        !ensure_allocation_at_least(C, (c_max_idx + 1) * sizeof(double))) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    gemm_col_major(transa, transb, m, n, k, a, A, 1, lda, B, 1, ldb, b, C, 1, ldc);
    FGPU_LOG("[FakeCUBLAS] cublasDgemm_v2 m=%d n=%d k=%d (cpu)\n", m, n, k);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    size_t total_elements = static_cast<size_t>(m) * n;
    fillRandom(C, total_elements);

    FGPU_LOG("[FakeCUBLAS] cublasDgemm_v2 m=%d n=%d k=%d (output %zu elements)\n", m, n, k, total_elements);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
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

    FGPU_LOG("[FakeCUBLAS] cublasHgemm m=%d n=%d k=%d\n", m, n, k);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Strided Batched GEMM
// ============================================================================

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (batchCount < 0 || strideA < 0 || strideB < 0 || strideC < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_handles.find(handle);
        if (it != g_handles.end()) {
            mode = it->second->pointerMode;
        }
    }
    const double a = static_cast<double>(getScalar(alpha, mode));
    const double b = static_cast<double>(getScalar(beta, mode));
    for (int batch = 0; batch < batchCount; ++batch) {
        const float* Ab = A + static_cast<ptrdiff_t>(batch) * strideA;
        const float* Bb = B + static_cast<ptrdiff_t>(batch) * strideB;
        float* Cb = C + static_cast<ptrdiff_t>(batch) * strideC;
        gemm_col_major(transa, transb, m, n, k, a, Ab, 0, lda, Bb, 0, ldb, b, Cb, 0, ldc);
    }
    FGPU_LOG("[FakeCUBLAS] cublasSgemmStridedBatched m=%d n=%d k=%d batchCount=%d (cpu)\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    size_t total_elements = static_cast<size_t>(m) * n * batchCount;
    fillRandom(C, total_elements);

    FGPU_LOG("[FakeCUBLAS] cublasSgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (batchCount < 0 || strideA < 0 || strideB < 0 || strideC < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_handles.find(handle);
        if (it != g_handles.end()) {
            mode = it->second->pointerMode;
        }
    }
    const double a = static_cast<double>(getScalar(alpha, mode));
    const double b = static_cast<double>(getScalar(beta, mode));
    for (int batch = 0; batch < batchCount; ++batch) {
        const double* Ab = A + static_cast<ptrdiff_t>(batch) * strideA;
        const double* Bb = B + static_cast<ptrdiff_t>(batch) * strideB;
        double* Cb = C + static_cast<ptrdiff_t>(batch) * strideC;
        gemm_col_major(transa, transb, m, n, k, a, Ab, 1, lda, Bb, 1, ldb, b, Cb, 1, ldc);
    }
    FGPU_LOG("[FakeCUBLAS] cublasDgemmStridedBatched m=%d n=%d k=%d batchCount=%d (cpu)\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    size_t total_elements = static_cast<size_t>(m) * n * batchCount;
    fillRandom(C, total_elements);

    FGPU_LOG("[FakeCUBLAS] cublasDgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
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

    FGPU_LOG("[FakeCUBLAS] cublasHgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Batched GEMM
// ============================================================================

cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *const Aarray[], int lda, const float *const Barray[], int ldb, const float *beta, float *const Carray[], int ldc, int batchCount) {
    if (!Aarray || !Barray || !Carray || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_handles.find(handle);
        if (it != g_handles.end()) {
            mode = it->second->pointerMode;
        }
    }
    const double a = static_cast<double>(getScalar(alpha, mode));
    const double b = static_cast<double>(getScalar(beta, mode));
    for (int batch = 0; batch < batchCount; ++batch) {
        const float* Ab = Aarray[batch];
        const float* Bb = Barray[batch];
        float* Cb = Carray[batch];
        if (!Ab || !Bb || !Cb) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        gemm_col_major(transa, transb, m, n, k, a, Ab, 0, lda, Bb, 0, ldb, b, Cb, 0, ldc);
    }
    FGPU_LOG("[FakeCUBLAS] cublasSgemmBatched m=%d n=%d k=%d batchCount=%d (cpu)\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        const void* out_ptr = first_nonnull_ptr(reinterpret_cast<const void* const*>(Carray), batchCount);
        if (out_ptr) {
            fake_gpu::GlobalState::instance().record_cublas_gemm(
                out_ptr, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    size_t matrix_elements = static_cast<size_t>(m) * n;
    for (int i = 0; i < batchCount; i++) {
        if (Carray[i]) {
            fillRandom(Carray[i], matrix_elements);
        }
    }

    FGPU_LOG("[FakeCUBLAS] cublasSgemmBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        const void* out_ptr = first_nonnull_ptr(reinterpret_cast<const void* const*>(Carray), batchCount);
        if (out_ptr) {
            fake_gpu::GlobalState::instance().record_cublas_gemm(
                out_ptr, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *const Aarray[], int lda, const double *const Barray[], int ldb, const double *beta, double *const Carray[], int ldc, int batchCount) {
    if (!Aarray || !Barray || !Carray || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_handles.find(handle);
        if (it != g_handles.end()) {
            mode = it->second->pointerMode;
        }
    }
    const double a = static_cast<double>(getScalar(alpha, mode));
    const double b = static_cast<double>(getScalar(beta, mode));
    for (int batch = 0; batch < batchCount; ++batch) {
        const double* Ab = Aarray[batch];
        const double* Bb = Barray[batch];
        double* Cb = Carray[batch];
        if (!Ab || !Bb || !Cb) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        gemm_col_major(transa, transb, m, n, k, a, Ab, 1, lda, Bb, 1, ldb, b, Cb, 1, ldc);
    }
    FGPU_LOG("[FakeCUBLAS] cublasDgemmBatched m=%d n=%d k=%d batchCount=%d (cpu)\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        const void* out_ptr = first_nonnull_ptr(reinterpret_cast<const void* const*>(Carray), batchCount);
        if (out_ptr) {
            fake_gpu::GlobalState::instance().record_cublas_gemm(
                out_ptr, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    size_t matrix_elements = static_cast<size_t>(m) * n;
    for (int i = 0; i < batchCount; i++) {
        if (Carray[i]) {
            fillRandom(Carray[i], matrix_elements);
        }
    }

    FGPU_LOG("[FakeCUBLAS] cublasDgemmBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        const void* out_ptr = first_nonnull_ptr(reinterpret_cast<const void* const*>(Carray), batchCount);
        if (out_ptr) {
            fake_gpu::GlobalState::instance().record_cublas_gemm(
                out_ptr, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
}

// ============================================================================
// GEMMEx (Mixed Precision GEMM)
// ============================================================================

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int Atype, int lda, const void *B, int Btype, int ldb, const void *beta, void *C, int Ctype, int ldc, int computeType, cublasGemmAlgo_t algo) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!is_supported_gemm_datatype(Atype) || !is_supported_gemm_datatype(Btype) || !is_supported_gemm_datatype(Ctype)) {
        FGPU_LOG("[FakeCUBLAS] cublasGemmEx unsupported types A=%d B=%d C=%d; skipping cpu compute\n", Atype, Btype, Ctype);
        if (m > 0 && n > 0 && k > 0) {
            fake_gpu::GlobalState::instance().record_cublas_gemm(
                C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    if (m < 0 || n < 0 || k < 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }

    // Interpret alpha/beta using computeType (best-effort).
    double a = 0.0;
    double b = 0.0;
    switch (computeType) {
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            a = *reinterpret_cast<const double*>(alpha);
            b = *reinterpret_cast<const double*>(beta);
            break;
        default:
            a = *reinterpret_cast<const float*>(alpha);
            b = *reinterpret_cast<const float*>(beta);
            break;
    }

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int a_cols = (transa == CUBLAS_OP_N) ? k : m;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    const int b_cols = (transb == CUBLAS_OP_N) ? n : k;
    if (a_rows > lda || b_rows > ldb || m > ldc) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const size_t a_max_idx = static_cast<size_t>(a_rows - 1) + static_cast<size_t>(a_cols - 1) * static_cast<size_t>(lda);
    const size_t b_max_idx = static_cast<size_t>(b_rows - 1) + static_cast<size_t>(b_cols - 1) * static_cast<size_t>(ldb);
    const size_t c_max_idx = static_cast<size_t>(m - 1) + static_cast<size_t>(n - 1) * static_cast<size_t>(ldc);
    if (!ensure_allocation_at_least(A, (a_max_idx + 1) * getDataTypeSize(Atype)) ||
        !ensure_allocation_at_least(B, (b_max_idx + 1) * getDataTypeSize(Btype)) ||
        !ensure_allocation_at_least(C, (c_max_idx + 1) * getDataTypeSize(Ctype))) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    gemm_col_major(transa, transb, m, n, k, a, A, Atype, lda, B, Btype, ldb, b, C, Ctype, ldc);
    FGPU_LOG("[FakeCUBLAS] cublasGemmEx m=%d n=%d k=%d Atype=%d Btype=%d Ctype=%d computeType=%d (cpu)\n",
           m, n, k, Atype, Btype, Ctype, computeType);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    // Don't fill output buffer - PyTorch manages memory and filling it with random data
    // can corrupt its internal state. Just return success to let PyTorch continue.

    FGPU_LOG("[FakeCUBLAS] cublasGemmEx m=%d n=%d k=%d Atype=%d Btype=%d Ctype=%d computeType=%d\n",
           m, n, k, Atype, Btype, Ctype, computeType);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int Atype, int lda, long long int strideA, const void *B, int Btype, int ldb, long long int strideB, const void *beta, void *C, int Ctype, int ldc, long long int strideC, int batchCount, int computeType, cublasGemmAlgo_t algo) {
    if (!A || !B || !C || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!is_supported_gemm_datatype(Atype) || !is_supported_gemm_datatype(Btype) || !is_supported_gemm_datatype(Ctype)) {
        FGPU_LOG("[FakeCUBLAS] cublasGemmStridedBatchedEx unsupported types A=%d B=%d C=%d; skipping cpu compute\n", Atype, Btype, Ctype);
        if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
            fake_gpu::GlobalState::instance().record_cublas_gemm(
                C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    if (batchCount < 0 || strideA < 0 || strideB < 0 || strideC < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const size_t a_elem_size = getDataTypeSize(Atype);
    const size_t b_elem_size = getDataTypeSize(Btype);
    const size_t c_elem_size = getDataTypeSize(Ctype);

    double a = 0.0;
    double b = 0.0;
    switch (computeType) {
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            a = *reinterpret_cast<const double*>(alpha);
            b = *reinterpret_cast<const double*>(beta);
            break;
        default:
            a = *reinterpret_cast<const float*>(alpha);
            b = *reinterpret_cast<const float*>(beta);
            break;
    }

    for (int batch = 0; batch < batchCount; ++batch) {
        const void* Ab = reinterpret_cast<const char*>(A) + static_cast<long long>(batch) * strideA * static_cast<long long>(a_elem_size);
        const void* Bb = reinterpret_cast<const char*>(B) + static_cast<long long>(batch) * strideB * static_cast<long long>(b_elem_size);
        void* Cb = reinterpret_cast<char*>(C) + static_cast<long long>(batch) * strideC * static_cast<long long>(c_elem_size);
        gemm_col_major(transa, transb, m, n, k, a, Ab, Atype, lda, Bb, Btype, ldb, b, Cb, Ctype, ldc);
    }
    FGPU_LOG("[FakeCUBLAS] cublasGemmStridedBatchedEx m=%d n=%d k=%d batchCount=%d (cpu)\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    // Don't fill output buffer - PyTorch manages memory and filling it with random data
    // can corrupt its internal state. Just return success to let PyTorch continue.

    FGPU_LOG("[FakeCUBLAS] cublasGemmStridedBatchedEx m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *const Aarray[], int Atype, int lda, const void *const Barray[], int Btype, int ldb, const void *beta, void *const Carray[], int Ctype, int ldc, int batchCount, int computeType, cublasGemmAlgo_t algo) {
    if (!Aarray || !Barray || !Carray || !alpha || !beta) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
#if FAKEGPU_CPU_SIMULATION
    if (!is_supported_gemm_datatype(Atype) || !is_supported_gemm_datatype(Btype) || !is_supported_gemm_datatype(Ctype)) {
        FGPU_LOG("[FakeCUBLAS] cublasGemmBatchedEx unsupported types A=%d B=%d C=%d; skipping cpu compute\n", Atype, Btype, Ctype);
        if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
            const void* out_ptr = first_nonnull_ptr(reinterpret_cast<const void* const*>(Carray), batchCount);
            if (out_ptr) {
                fake_gpu::GlobalState::instance().record_cublas_gemm(
                    out_ptr, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
            }
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    double a = 0.0;
    double b = 0.0;
    switch (computeType) {
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            a = *reinterpret_cast<const double*>(alpha);
            b = *reinterpret_cast<const double*>(beta);
            break;
        default:
            a = *reinterpret_cast<const float*>(alpha);
            b = *reinterpret_cast<const float*>(beta);
            break;
    }

    for (int batch = 0; batch < batchCount; ++batch) {
        const void* Ab = Aarray[batch];
        const void* Bb = Barray[batch];
        void* Cb = Carray[batch];
        if (!Ab || !Bb || !Cb) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        gemm_col_major(transa, transb, m, n, k, a, Ab, Atype, lda, Bb, Btype, ldb, b, Cb, Ctype, ldc);
    }
    FGPU_LOG("[FakeCUBLAS] cublasGemmBatchedEx m=%d n=%d k=%d batchCount=%d (cpu)\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        const void* out_ptr = first_nonnull_ptr(reinterpret_cast<const void* const*>(Carray), batchCount);
        if (out_ptr) {
            fake_gpu::GlobalState::instance().record_cublas_gemm(
                out_ptr, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    // Don't fill output buffer - PyTorch manages memory and filling it with random data
    // can corrupt its internal state. Just return success to let PyTorch continue.

    FGPU_LOG("[FakeCUBLAS] cublasGemmBatchedEx m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        const void* out_ptr = first_nonnull_ptr(reinterpret_cast<const void* const*>(Carray), batchCount);
        if (out_ptr) {
            fake_gpu::GlobalState::instance().record_cublas_gemm(
                out_ptr, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount)));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
}

// ============================================================================
// TRSM (Triangular Solve)
// ============================================================================

cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb) {
    if (!A || !B || !alpha) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    fillRandom(B, m * n);
    FGPU_LOG("[FakeCUBLAS] cublasStrsm_v2 m=%d n=%d\n", m, n);
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

    FGPU_LOG("[FakeCUBLAS] cublasStrsmBatched m=%d n=%d batchCount=%d\n", m, n, batchCount);
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
    FGPU_LOG("[FakeCUBLAS] cublasSsymm_v2 m=%d n=%d\n", m, n);
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
    FGPU_LOG("[FakeCUBLAS] cublasSsyrk_v2 n=%d k=%d\n", n, k);
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
    FGPU_LOG("[FakeCUBLAS] cublasStrmm_v2 m=%d n=%d\n", m, n);
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
    FGPU_LOG("[FakeCUBLAS] cublasCgemm_v2 m=%d n=%d k=%d\n", m, n, k);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1, /*complex=*/true));
    }
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
    FGPU_LOG("[FakeCUBLAS] cublasZgemm_v2 m=%d n=%d k=%d\n", m, n, k);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1, /*complex=*/true));
    }
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
    FGPU_LOG("[FakeCUBLAS] cublasCgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount), /*complex=*/true));
    }
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
    FGPU_LOG("[FakeCUBLAS] cublasZgemmStridedBatched m=%d n=%d k=%d batchCount=%d\n", m, n, k, batchCount);
    if (m > 0 && n > 0 && k > 0 && batchCount > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), static_cast<uint64_t>(batchCount), /*complex=*/true));
    }
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
    FGPU_LOG("[FakeCUBLAS] cublasSgetrfBatched n=%d batchSize=%d\n", n, batchSize);
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
    FGPU_LOG("[FakeCUBLAS] cublasSetWorkspace_v2 workspaceSize=%zu\n", workspaceSizeInBytes);
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
#if FAKEGPU_CPU_SIMULATION
    if (!is_supported_gemm_datatype(Atype) || !is_supported_gemm_datatype(Btype) || !is_supported_gemm_datatype(Ctype)) {
        FGPU_LOG("[FakeCUBLAS] cublasSgemmEx unsupported types A=%d B=%d C=%d; skipping cpu compute\n", Atype, Btype, Ctype);
        if (m > 0 && n > 0 && k > 0) {
            fake_gpu::GlobalState::instance().record_cublas_gemm(
                C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    if (m < 0 || n < 0 || k < 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    const double a = static_cast<double>(*alpha);
    const double b = static_cast<double>(*beta);
    gemm_col_major(transa, transb, m, n, k, a, A, Atype, lda, B, Btype, ldb, b, C, Ctype, ldc);
    FGPU_LOG("[FakeCUBLAS] cublasSgemmEx m=%d n=%d k=%d (cpu)\n", m, n, k);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
    }
    return CUBLAS_STATUS_SUCCESS;
#else
    // Leave output buffer untouched; FakeGPU only needs to report success
    FGPU_LOG("[FakeCUBLAS] cublasSgemmEx m=%d n=%d k=%d\n", m, n, k);
    if (m > 0 && n > 0 && k > 0) {
        fake_gpu::GlobalState::instance().record_cublas_gemm(
            C, gemm_flops_u64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), static_cast<uint64_t>(k), 1));
    }
    return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t cublasGemmEx_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, int Atype, int lda, const void *B, int Btype, int ldb, const void *beta, void *C, int Ctype, int ldc, int computeType, cublasGemmAlgo_t algo) {
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}

// ============================================================================
// cuBLASLt (Lightweight BLAS) Implementation
// ============================================================================

namespace {

// Keep these numeric values aligned with CUDA's cublasLt.h for interoperability.
static constexpr int32_t kLtOrderCol = 0;
static constexpr int32_t kLtOrderRow = 1;

static constexpr int32_t kLtMatrixLayoutType = 0;
static constexpr int32_t kLtMatrixLayoutOrder = 1;
static constexpr int32_t kLtMatrixLayoutRows = 2;
static constexpr int32_t kLtMatrixLayoutCols = 3;
static constexpr int32_t kLtMatrixLayoutLd = 4;
static constexpr int32_t kLtMatrixLayoutBatchCount = 5;
static constexpr int32_t kLtMatrixLayoutStridedBatchOffset = 6;
static constexpr int32_t kLtMatrixLayoutPlaneOffset = 7;

static constexpr int32_t kLtMatmulDescComputeType = 0;
static constexpr int32_t kLtMatmulDescScaleType = 1;
static constexpr int32_t kLtMatmulDescPointerMode = 2;
static constexpr int32_t kLtMatmulDescTransA = 3;
static constexpr int32_t kLtMatmulDescTransB = 4;
static constexpr int32_t kLtMatmulDescTransC = 5;
static constexpr int32_t kLtMatmulDescEpilogue = 7;
static constexpr int32_t kLtMatmulDescBiasPointer = 8;
static constexpr int32_t kLtMatmulDescBiasBatchStride = 10;
static constexpr int32_t kLtMatmulDescBiasDataType = 26;

static constexpr uint32_t kLtEpilogueDefault = 1;
static constexpr uint32_t kLtEpilogueRelu = 2;
static constexpr uint32_t kLtEpilogueBias = 4;
static constexpr uint32_t kLtEpilogueGelu = 32;

struct LtMatmulDescState {
    int32_t computeType = 0;
    int32_t scaleType = 0;
    int32_t pointerMode = CUBLAS_POINTER_MODE_HOST;
    int32_t transa = CUBLAS_OP_N;
    int32_t transb = CUBLAS_OP_N;
    int32_t transc = CUBLAS_OP_N;
    uint32_t epilogue = kLtEpilogueDefault;
    const void* biasPointer = nullptr;
    int64_t biasBatchStride = 0;
    int32_t biasDataType = -1;
};

struct LtMatrixLayoutState {
    int32_t type = 0;
    int32_t order = kLtOrderCol;
    uint64_t rows = 0;
    uint64_t cols = 0;
    int64_t ld = 0;
    int32_t batchCount = 1;
    int64_t strideBatchOffset = 0;
    int64_t planeOffset = 0;
};

bool lt_order_supported(int32_t order) {
    return order == kLtOrderCol || order == kLtOrderRow;
}

size_t lt_index(const LtMatrixLayoutState& layout, uint64_t row, uint64_t col) {
    if (layout.order == kLtOrderRow) {
        return static_cast<size_t>(row * static_cast<uint64_t>(layout.ld) + col);
    }
    return static_cast<size_t>(row + col * static_cast<uint64_t>(layout.ld));
}

bool lt_valid_ld(const LtMatrixLayoutState& layout) {
    if (layout.rows == 0 || layout.cols == 0) return true;
    if (layout.ld <= 0) return false;
    if (layout.order == kLtOrderRow) {
        return static_cast<uint64_t>(layout.ld) >= layout.cols;
    }
    return static_cast<uint64_t>(layout.ld) >= layout.rows;
}

int64_t lt_default_batch_stride(const LtMatrixLayoutState& layout) {
    if (layout.batchCount <= 1) return 0;
    if (layout.strideBatchOffset != 0) return layout.strideBatchOffset;
    if (layout.order == kLtOrderRow) {
        return layout.ld * static_cast<int64_t>(layout.rows);
    }
    return layout.ld * static_cast<int64_t>(layout.cols);
}

} // namespace

// Global map to track cuBLASLt handles and descriptors
static std::map<cublasLtHandle_t, void*> g_lt_handles;
static std::map<cublasLtMatmulDesc_t, LtMatmulDescState*> g_lt_matmul_descs;
static std::map<cublasLtMatrixLayout_t, LtMatrixLayoutState*> g_lt_matrix_layouts;
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

    FGPU_LOG("[FakeCUBLASLt] cublasLtCreate handle=%p\n", *lightHandle);
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

    FGPU_LOG("[FakeCUBLASLt] cublasLtDestroy handle=%p\n", lightHandle);
    return CUBLAS_STATUS_SUCCESS;
}

// Matmul descriptor management
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, int computeType, int scaleType) {
    if (!matmulDesc) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    auto* desc = new (std::nothrow) LtMatmulDescState();
    if (!desc) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    desc->computeType = computeType;
    desc->scaleType = scaleType;
    *matmulDesc = reinterpret_cast<cublasLtMatmulDesc_t>(desc);
    g_lt_matmul_descs[*matmulDesc] = desc;

    FGPU_LOG("[FakeCUBLASLt] cublasLtMatmulDescCreate desc=%p computeType=%d scaleType=%d\n", *matmulDesc, computeType, scaleType);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_lt_matmul_descs.find(matmulDesc);
    if (it == g_lt_matmul_descs.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    delete it->second;
    g_lt_matmul_descs.erase(it);

    FGPU_LOG("[FakeCUBLASLt] cublasLtMatmulDescDestroy desc=%p\n", matmulDesc);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, int attr, const void *buf, size_t sizeInBytes) {
    if (!matmulDesc) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_lt_matmul_descs.find(matmulDesc);
    if (it == g_lt_matmul_descs.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    LtMatmulDescState* desc = it->second;
    if (sizeInBytes > 0 && !buf) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    switch (attr) {
        case kLtMatmulDescComputeType: {
            if (sizeInBytes != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->computeType, buf, sizeof(int32_t));
            break;
        }
        case kLtMatmulDescScaleType: {
            if (sizeInBytes != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->scaleType, buf, sizeof(int32_t));
            break;
        }
        case kLtMatmulDescPointerMode: {
            if (sizeInBytes != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->pointerMode, buf, sizeof(int32_t));
            break;
        }
        case kLtMatmulDescTransA: {
            if (sizeInBytes != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->transa, buf, sizeof(int32_t));
            break;
        }
        case kLtMatmulDescTransB: {
            if (sizeInBytes != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->transb, buf, sizeof(int32_t));
            break;
        }
        case kLtMatmulDescTransC: {
            if (sizeInBytes != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->transc, buf, sizeof(int32_t));
            break;
        }
        case kLtMatmulDescEpilogue: {
            if (sizeInBytes != sizeof(uint32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->epilogue, buf, sizeof(uint32_t));
            break;
        }
        case kLtMatmulDescBiasPointer: {
            if (sizeInBytes != sizeof(void*)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->biasPointer, buf, sizeof(void*));
            break;
        }
        case kLtMatmulDescBiasBatchStride: {
            if (sizeInBytes != sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->biasBatchStride, buf, sizeof(int64_t));
            break;
        }
        case kLtMatmulDescBiasDataType: {
            if (sizeInBytes != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&desc->biasDataType, buf, sizeof(int32_t));
            break;
        }
        default:
            break;
    }
    FGPU_LOG("[FakeCUBLASLt] cublasLtMatmulDescSetAttribute desc=%p attr=%d size=%zu\n", matmulDesc, attr, sizeInBytes);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, int attr, void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    if (sizeWritten) *sizeWritten = 0;
    if (!matmulDesc) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_lt_matmul_descs.find(matmulDesc);
    if (it == g_lt_matmul_descs.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const LtMatmulDescState* desc = it->second;

    auto write_out = [&](const void* src, size_t src_size) -> cublasStatus_t {
        if (sizeWritten) *sizeWritten = src_size;
        if (sizeInBytes == 0) {
            return buf ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
        }
        if (!buf || sizeInBytes != src_size) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        std::memcpy(buf, src, src_size);
        return CUBLAS_STATUS_SUCCESS;
    };

    switch (attr) {
        case kLtMatmulDescComputeType:
            return write_out(&desc->computeType, sizeof(desc->computeType));
        case kLtMatmulDescScaleType:
            return write_out(&desc->scaleType, sizeof(desc->scaleType));
        case kLtMatmulDescPointerMode:
            return write_out(&desc->pointerMode, sizeof(desc->pointerMode));
        case kLtMatmulDescTransA:
            return write_out(&desc->transa, sizeof(desc->transa));
        case kLtMatmulDescTransB:
            return write_out(&desc->transb, sizeof(desc->transb));
        case kLtMatmulDescTransC:
            return write_out(&desc->transc, sizeof(desc->transc));
        case kLtMatmulDescEpilogue:
            return write_out(&desc->epilogue, sizeof(desc->epilogue));
        case kLtMatmulDescBiasPointer:
            return write_out(&desc->biasPointer, sizeof(desc->biasPointer));
        case kLtMatmulDescBiasBatchStride:
            return write_out(&desc->biasBatchStride, sizeof(desc->biasBatchStride));
        case kLtMatmulDescBiasDataType:
            return write_out(&desc->biasDataType, sizeof(desc->biasDataType));
        default:
            return CUBLAS_STATUS_SUCCESS;
    }
}

// Matrix layout management
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, int type, uint64_t rows, uint64_t cols, int64_t ld) {
    if (!matLayout) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    auto* layout = new (std::nothrow) LtMatrixLayoutState();
    if (!layout) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    layout->type = type;
    layout->rows = rows;
    layout->cols = cols;
    layout->ld = ld;
    *matLayout = reinterpret_cast<cublasLtMatrixLayout_t>(layout);
    g_lt_matrix_layouts[*matLayout] = layout;

    FGPU_LOG("[FakeCUBLASLt] cublasLtMatrixLayoutCreate layout=%p type=%d rows=%lu cols=%lu ld=%ld\n",
           *matLayout, type, rows, cols, ld);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_lt_matrix_layouts.find(matLayout);
    if (it == g_lt_matrix_layouts.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    delete it->second;
    g_lt_matrix_layouts.erase(it);

    FGPU_LOG("[FakeCUBLASLt] cublasLtMatrixLayoutDestroy layout=%p\n", matLayout);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, int attr, const void *buf, size_t sizeInBytes) {
    if (!matLayout) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_lt_matrix_layouts.find(matLayout);
    if (it == g_lt_matrix_layouts.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    LtMatrixLayoutState* layout = it->second;
    if (sizeInBytes > 0 && !buf) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    switch (attr) {
        case kLtMatrixLayoutType: {
            if (sizeInBytes != sizeof(int32_t) && sizeInBytes != sizeof(uint32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&layout->type, buf, sizeof(int32_t));
            break;
        }
        case kLtMatrixLayoutOrder: {
            if (sizeInBytes != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&layout->order, buf, sizeof(int32_t));
            break;
        }
        case kLtMatrixLayoutRows: {
            if (sizeInBytes != sizeof(uint64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&layout->rows, buf, sizeof(uint64_t));
            break;
        }
        case kLtMatrixLayoutCols: {
            if (sizeInBytes != sizeof(uint64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&layout->cols, buf, sizeof(uint64_t));
            break;
        }
        case kLtMatrixLayoutLd: {
            if (sizeInBytes != sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&layout->ld, buf, sizeof(int64_t));
            break;
        }
        case kLtMatrixLayoutBatchCount: {
            if (sizeInBytes != sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&layout->batchCount, buf, sizeof(int32_t));
            break;
        }
        case kLtMatrixLayoutStridedBatchOffset: {
            if (sizeInBytes != sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&layout->strideBatchOffset, buf, sizeof(int64_t));
            break;
        }
        case kLtMatrixLayoutPlaneOffset: {
            if (sizeInBytes != sizeof(int64_t)) return CUBLAS_STATUS_INVALID_VALUE;
            std::memcpy(&layout->planeOffset, buf, sizeof(int64_t));
            break;
        }
        default:
            break;
    }
    FGPU_LOG("[FakeCUBLASLt] cublasLtMatrixLayoutSetAttribute layout=%p attr=%d size=%zu\n", matLayout, attr, sizeInBytes);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout, int attr, void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    if (sizeWritten) *sizeWritten = 0;
    if (!matLayout) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_lt_matrix_layouts.find(matLayout);
    if (it == g_lt_matrix_layouts.end()) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const LtMatrixLayoutState* layout = it->second;

    auto write_out = [&](const void* src, size_t src_size) -> cublasStatus_t {
        if (sizeWritten) *sizeWritten = src_size;
        if (sizeInBytes == 0) {
            return buf ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
        }
        if (!buf || sizeInBytes != src_size) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        std::memcpy(buf, src, src_size);
        return CUBLAS_STATUS_SUCCESS;
    };

    switch (attr) {
        case kLtMatrixLayoutType:
            return write_out(&layout->type, sizeof(layout->type));
        case kLtMatrixLayoutOrder:
            return write_out(&layout->order, sizeof(layout->order));
        case kLtMatrixLayoutRows:
            return write_out(&layout->rows, sizeof(layout->rows));
        case kLtMatrixLayoutCols:
            return write_out(&layout->cols, sizeof(layout->cols));
        case kLtMatrixLayoutLd:
            return write_out(&layout->ld, sizeof(layout->ld));
        case kLtMatrixLayoutBatchCount:
            return write_out(&layout->batchCount, sizeof(layout->batchCount));
        case kLtMatrixLayoutStridedBatchOffset:
            return write_out(&layout->strideBatchOffset, sizeof(layout->strideBatchOffset));
        case kLtMatrixLayoutPlaneOffset:
            return write_out(&layout->planeOffset, sizeof(layout->planeOffset));
        default:
            return CUBLAS_STATUS_SUCCESS;
    }
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

    FGPU_LOG("[FakeCUBLASLt] cublasLtMatmulPreferenceCreate pref=%p\n", *pref);
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

    FGPU_LOG("[FakeCUBLASLt] cublasLtMatmulPreferenceDestroy pref=%p\n", pref);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, int attr, const void *buf, size_t sizeInBytes) {
    FGPU_LOG("[FakeCUBLASLt] cublasLtMatmulPreferenceSetAttribute pref=%p attr=%d size=%zu\n", pref, attr, sizeInBytes);
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

    FGPU_LOG("[FakeCUBLASLt] cublasLtMatmulAlgoGetHeuristic requested=%d returned=%d\n",
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

#if !FAKEGPU_CPU_SIMULATION
    FGPU_LOG("[FakeCUBLASLt] cublasLtMatmul A=%p B=%p D=%p workspace=%zu\n",
           A, B, D, workspaceSizeInBytes);
    return CUBLAS_STATUS_SUCCESS;
#else
    LtMatmulDescState desc;
    LtMatrixLayoutState a_layout;
    LtMatrixLayoutState b_layout;
    LtMatrixLayoutState c_layout;
    LtMatrixLayoutState d_layout;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it_desc = g_lt_matmul_descs.find(computeDesc);
        auto it_a = g_lt_matrix_layouts.find(Adesc);
        auto it_b = g_lt_matrix_layouts.find(Bdesc);
        auto it_c = g_lt_matrix_layouts.find(Cdesc);
        auto it_d = g_lt_matrix_layouts.find(Ddesc);
        if (it_desc == g_lt_matmul_descs.end() || it_a == g_lt_matrix_layouts.end() || it_b == g_lt_matrix_layouts.end() || it_d == g_lt_matrix_layouts.end()) {
            return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        desc = *it_desc->second;
        a_layout = *it_a->second;
        b_layout = *it_b->second;
        d_layout = *it_d->second;
        if (it_c != g_lt_matrix_layouts.end()) {
            c_layout = *it_c->second;
        } else {
            c_layout = d_layout;
        }
    }

    if (!lt_order_supported(a_layout.order) || !lt_order_supported(b_layout.order) || !lt_order_supported(c_layout.order) || !lt_order_supported(d_layout.order)) {
        FGPU_LOG("[FakeCUBLASLt] cublasLtMatmul unsupported matrix order; skipping cpu compute\n");
        return CUBLAS_STATUS_SUCCESS;
    }
    if (!lt_valid_ld(a_layout) || !lt_valid_ld(b_layout) || !lt_valid_ld(c_layout) || !lt_valid_ld(d_layout)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_supported_gemm_datatype(a_layout.type) || !is_supported_gemm_datatype(b_layout.type) || !is_supported_gemm_datatype(d_layout.type) || (C && !is_supported_gemm_datatype(c_layout.type))) {
        FGPU_LOG("[FakeCUBLASLt] cublasLtMatmul unsupported types; skipping cpu compute\n");
        return CUBLAS_STATUS_SUCCESS;
    }

    if (desc.pointerMode != CUBLAS_POINTER_MODE_HOST && desc.pointerMode != CUBLAS_POINTER_MODE_DEVICE) {
        FGPU_LOG("[FakeCUBLASLt] cublasLtMatmul unsupported pointerMode=%d; skipping cpu compute\n", desc.pointerMode);
        return CUBLAS_STATUS_SUCCESS;
    }

    const double a = read_scalar_typed(alpha, desc.scaleType);
    const double b = read_scalar_typed(beta, desc.scaleType);

    const cublasOperation_t transa = static_cast<cublasOperation_t>(desc.transa);
    const cublasOperation_t transb = static_cast<cublasOperation_t>(desc.transb);

    const uint64_t opA_rows = (transa == CUBLAS_OP_N) ? a_layout.rows : a_layout.cols;
    const uint64_t opA_cols = (transa == CUBLAS_OP_N) ? a_layout.cols : a_layout.rows;
    const uint64_t opB_rows = (transb == CUBLAS_OP_N) ? b_layout.rows : b_layout.cols;
    const uint64_t opB_cols = (transb == CUBLAS_OP_N) ? b_layout.cols : b_layout.rows;

    if (opA_cols != opB_rows) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (d_layout.rows != opA_rows || d_layout.cols != opB_cols) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (C && (c_layout.rows != d_layout.rows || c_layout.cols != d_layout.cols)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (d_layout.rows == 0 || d_layout.cols == 0 || opA_cols == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }

    if (d_layout.batchCount <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const int batchCount = d_layout.batchCount;
    auto batch_ok = [&](int32_t other) { return other == 1 || other == batchCount; };
    if (!batch_ok(a_layout.batchCount) || !batch_ok(b_layout.batchCount) || !batch_ok(c_layout.batchCount)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int64_t strideA = lt_default_batch_stride(a_layout);
    const int64_t strideB = lt_default_batch_stride(b_layout);
    const int64_t strideC = lt_default_batch_stride(c_layout);
    const int64_t strideD = lt_default_batch_stride(d_layout);
    const size_t a_elem_size = getDataTypeSize(a_layout.type);
    const size_t b_elem_size = getDataTypeSize(b_layout.type);
    const size_t c_elem_size = getDataTypeSize(c_layout.type);
    const size_t d_elem_size = getDataTypeSize(d_layout.type);

    const uint32_t epilogue = desc.epilogue;
    const bool do_bias = (epilogue & kLtEpilogueBias) != 0;
    const bool do_relu = (epilogue & kLtEpilogueRelu) != 0;
    const bool do_gelu = (epilogue & kLtEpilogueGelu) != 0;
    const int biasType = (desc.biasDataType != -1) ? desc.biasDataType : d_layout.type;
    if (do_bias && (!desc.biasPointer || !is_supported_gemm_datatype(biasType))) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const size_t a_required = (lt_index(a_layout, a_layout.rows - 1, a_layout.cols - 1) + 1) * a_elem_size;
    const size_t b_required = (lt_index(b_layout, b_layout.rows - 1, b_layout.cols - 1) + 1) * b_elem_size;
    const size_t c_required = (lt_index(c_layout, c_layout.rows - 1, c_layout.cols - 1) + 1) * c_elem_size;
    const size_t d_required = (lt_index(d_layout, d_layout.rows - 1, d_layout.cols - 1) + 1) * d_elem_size;
    if (!ensure_allocation_at_least(A, a_required) || !ensure_allocation_at_least(B, b_required) || !ensure_allocation_at_least(D, d_required) || (C && !ensure_allocation_at_least(C, c_required))) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    if (should_skip_cpu_gemm(d_layout.rows, d_layout.cols, opA_cols, static_cast<uint64_t>(batchCount))) {
        for (int batch = 0; batch < batchCount; ++batch) {
            const int64_t d_batch = (d_layout.batchCount == 1) ? 0 : strideD * batch;
            void* Db = reinterpret_cast<char*>(D) + d_batch * static_cast<int64_t>(d_elem_size);
            std::memset(Db, 0, d_required);
        }
        FGPU_LOG("[FakeCUBLASLt] cublasLtMatmul skipped CPU compute (m=%llu n=%llu k=%llu batch=%d)\n",
               static_cast<unsigned long long>(d_layout.rows),
               static_cast<unsigned long long>(d_layout.cols),
               static_cast<unsigned long long>(opA_cols),
               batchCount);
        fake_gpu::GlobalState::instance().record_cublaslt_matmul(
            D, gemm_flops_u64(d_layout.rows, d_layout.cols, opA_cols, static_cast<uint64_t>(batchCount)));
        return CUBLAS_STATUS_SUCCESS;
    }

    for (int batch = 0; batch < batchCount; ++batch) {
        const int64_t a_batch = (a_layout.batchCount == 1) ? 0 : strideA * batch;
        const int64_t b_batch = (b_layout.batchCount == 1) ? 0 : strideB * batch;
        const int64_t c_batch = (c_layout.batchCount == 1) ? 0 : strideC * batch;
        const int64_t d_batch = (d_layout.batchCount == 1) ? 0 : strideD * batch;

        const void* Ab = reinterpret_cast<const char*>(A) + a_batch * static_cast<int64_t>(a_elem_size);
        const void* Bb = reinterpret_cast<const char*>(B) + b_batch * static_cast<int64_t>(b_elem_size);
        const void* Cb = C ? (reinterpret_cast<const char*>(C) + c_batch * static_cast<int64_t>(c_elem_size)) : nullptr;
        void* Db = reinterpret_cast<char*>(D) + d_batch * static_cast<int64_t>(d_elem_size);

        const uint64_t m = d_layout.rows;
        const uint64_t n = d_layout.cols;
        const uint64_t kk = opA_cols;

        for (uint64_t col = 0; col < n; ++col) {
            for (uint64_t row = 0; row < m; ++row) {
                double acc = 0.0;
                for (uint64_t p = 0; p < kk; ++p) {
                    const uint64_t a_r = (transa == CUBLAS_OP_N) ? row : p;
                    const uint64_t a_c = (transa == CUBLAS_OP_N) ? p : row;
                    const uint64_t b_r = (transb == CUBLAS_OP_N) ? p : col;
                    const uint64_t b_c = (transb == CUBLAS_OP_N) ? col : p;

                    const size_t a_idx = lt_index(a_layout, a_r, a_c);
                    const size_t b_idx = lt_index(b_layout, b_r, b_c);
                    acc += read_elem(Ab, a_layout.type, a_idx) * read_elem(Bb, b_layout.type, b_idx);
                }

                const size_t out_idx = lt_index(d_layout, row, col);
                double c_prev = 0.0;
                if (Cb && b != 0.0) {
                    const size_t c_idx = lt_index(c_layout, row, col);
                    c_prev = read_elem(Cb, c_layout.type, c_idx);
                }
                double out = a * acc + b * c_prev;

                if (do_bias) {
                    const int64_t bias_base = (desc.biasBatchStride != 0) ? desc.biasBatchStride * static_cast<int64_t>(batch) : 0;
                    out += read_elem(desc.biasPointer, biasType, static_cast<size_t>(bias_base + static_cast<int64_t>(row)));
                }
                if (do_relu) {
                    out = std::max(0.0, out);
                }
                if (do_gelu) {
                    out = gelu(out);
                }
                write_elem(Db, d_layout.type, out_idx, out);
            }
        }
    }

    FGPU_LOG("[FakeCUBLASLt] cublasLtMatmul computed on CPU (epilogue=%u)\n", epilogue);
    fake_gpu::GlobalState::instance().record_cublaslt_matmul(
        D, gemm_flops_u64(d_layout.rows, d_layout.cols, opA_cols, static_cast<uint64_t>(batchCount)));
    return CUBLAS_STATUS_SUCCESS;
#endif
}
