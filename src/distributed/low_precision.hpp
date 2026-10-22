#pragma once

#include <cstdint>
#include <cstring>

namespace fake_gpu::distributed {

inline float bfloat16_bits_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline std::uint16_t float_to_bfloat16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    // Preserve NaNs as NaNs and keep a non-zero payload.
    if ((bits & 0x7f800000U) == 0x7f800000U &&
        (bits & 0x007fffffU) != 0) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040U);
    }

    // Round to nearest, ties to even, matching CUDA/PyTorch conversions.
    const std::uint32_t rounding_bias = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

inline float float16_bits_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
    int exponent = static_cast<int>((value >> 10) & 0x1fU);
    std::uint32_t mantissa = value & 0x03ffU;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffU;
            const std::uint32_t float_exponent =
                static_cast<std::uint32_t>(exponent + (127 - 15));
            bits = sign | (float_exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000U | (mantissa << 13);
        if (mantissa != 0) {
            bits |= 0x00400000U;
        }
    } else {
        const std::uint32_t float_exponent =
            static_cast<std::uint32_t>(exponent + (127 - 15));
        bits = sign | (float_exponent << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline std::uint16_t float_to_float16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
    const std::uint32_t exponent = (bits >> 23) & 0xffU;
    const std::uint32_t mantissa = bits & 0x007fffffU;

    if (exponent == 0xffU) {
        if (mantissa == 0) {
            return static_cast<std::uint16_t>(sign | 0x7c00U);
        }
        return static_cast<std::uint16_t>(
            sign | 0x7c00U | static_cast<std::uint16_t>((mantissa >> 13) | 1U));
    }

    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }

    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return sign;
        }
        const std::uint32_t normalized_mantissa = mantissa | 0x00800000U;
        const int shift = 14 - half_exponent;
        std::uint32_t half_mantissa = normalized_mantissa >> shift;
        const std::uint32_t remainder_mask = (1U << shift) - 1U;
        const std::uint32_t remainder = normalized_mantissa & remainder_mask;
        const std::uint32_t halfway = 1U << (shift - 1);
        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1U) != 0)) {
            ++half_mantissa;
        }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half =
        static_cast<std::uint32_t>(sign) |
        (static_cast<std::uint32_t>(half_exponent) << 10) |
        (mantissa >> 13);
    const std::uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U ||
        (remainder == 0x1000U && (half & 1U) != 0)) {
        ++half;
    }
    return static_cast<std::uint16_t>(half);
}

// FP8 storage formats of ncclFloat8e4m3 and ncclFloat8e5m2.
enum class Fp8Format {
    E4M3,
    E5M2,
};

// Port of the host path of __nv_cvt_double_to_fp8 (CUDA 12.4 cuda_fp8.hpp)
// with __NV_SATFINITE, the saturation NCCL uses for every FP8 result.
inline std::uint8_t double_to_fp8_bits(double value, Fp8Format format) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const bool e4m3 = format == Fp8Format::E4M3;
    const std::uint16_t exp_bias = e4m3 ? 7U : 15U;
    const std::uint64_t significand_bits = e4m3 ? 4U : 3U;
    const std::uint8_t mantissa_mask = e4m3 ? 0x7U : 0x3U;
    const std::uint64_t min_denorm_over_2 =
        e4m3 ? 0x3F50000000000000ULL : 0x3EE0000000000000ULL;
    const std::uint64_t overflow_threshold =
        e4m3 ? 0x407D000000000000ULL : 0x40EE000000000000ULL - 1ULL;
    const std::uint8_t max_norm = e4m3 ? 0x7EU : 0x7BU;
    const std::uint64_t min_norm =
        e4m3 ? 0x3F90000000000000ULL : 0x3F10000000000000ULL;
    constexpr std::uint64_t kDoubleInfBits = 0x7FF0000000000000ULL;

    const std::uint64_t half_ulp = 1ULL << (53ULL - significand_bits - 1ULL);
    const std::uint8_t sign = static_cast<std::uint8_t>((bits >> 63U) << 7U);
    const std::uint8_t exponent = static_cast<std::uint8_t>(
        ((static_cast<std::uint16_t>(bits >> 52U)) & 0x7FFU) - 1023U + exp_bias);
    std::uint8_t mantissa = static_cast<std::uint8_t>(
        (bits >> (53U - significand_bits)) & mantissa_mask);
    const std::uint64_t absolute = bits & 0x7FFFFFFFFFFFFFFFULL;

    std::uint8_t result = 0;
    if (absolute <= min_denorm_over_2) {
        result = 0;
    } else if (absolute > kDoubleInfBits) {
        result = e4m3 ? 0x7FU : static_cast<std::uint8_t>(0x7EU | mantissa);
    } else if (absolute > overflow_threshold) {
        result = max_norm;
    } else if (absolute >= min_norm) {
        result = static_cast<std::uint8_t>(
            (exponent << (significand_bits - 1U)) | mantissa);
        const std::uint64_t round = bits & ((half_ulp << 1U) - 1U);
        if (round > half_ulp || (round == half_ulp && (mantissa & 1U) != 0)) {
            result = static_cast<std::uint8_t>(result + 1U);
        }
    } else {
        const std::uint8_t shift = static_cast<std::uint8_t>(1U - exponent);
        mantissa = static_cast<std::uint8_t>(
            mantissa | (1U << (significand_bits - 1U)));
        result = static_cast<std::uint8_t>(mantissa >> shift);
        const std::uint64_t round =
            (bits | (1ULL << 52U)) & ((half_ulp << (shift + 1U)) - 1U);
        if (round > (half_ulp << shift) ||
            (round == (half_ulp << shift) && (result & 1U) != 0)) {
            result = static_cast<std::uint8_t>(result + 1U);
        }
    }
    return static_cast<std::uint8_t>(result | sign);
}

// __nv_cvt_float_to_fp8(value, __NV_SATFINITE, format): NaN becomes the
// canonical NaN before the double conversion.
inline std::uint8_t float_to_fp8_bits(float value, Fp8Format format) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {
        bits = 0x7FFFFFFFU;
        std::memcpy(&value, &bits, sizeof(value));
    }
    return double_to_fp8_bits(static_cast<double>(value), format);
}

// Port of the host path of __nv_cvt_fp8_to_halfraw (CUDA 12.4). Every FP8
// value is exact in float16.
inline std::uint16_t fp8_bits_to_float16_bits(std::uint8_t value, Fp8Format format) {
    std::uint16_t result = static_cast<std::uint16_t>(value << 8U);
    if (format == Fp8Format::E5M2) {
        if ((result & 0x7FFFU) > 0x7C00U) {
            result = 0x7FFFU;
        }
        return result;
    }
    const std::uint16_t sign = result & 0x8000U;
    std::uint16_t exponent =
        static_cast<std::uint16_t>(((result & 0x7800U) >> 1U) + 0x2000U);
    std::uint16_t mantissa = static_cast<std::uint16_t>((result & 0x0700U) >> 1U);
    if ((value & 0x7FU) == 0x7FU) {
        return 0x7FFFU;
    }
    if (exponent == 0x2000U) {
        if (mantissa != 0U) {
            mantissa = static_cast<std::uint16_t>(mantissa << 1U);
            while ((mantissa & 0x0400U) == 0U) {
                mantissa = static_cast<std::uint16_t>(mantissa << 1U);
                exponent = static_cast<std::uint16_t>(exponent - 0x0400U);
            }
            mantissa &= 0x03FFU;
        } else {
            exponent = 0U;
        }
    }
    return static_cast<std::uint16_t>(sign | exponent | mantissa);
}

inline float fp8_bits_to_float(std::uint8_t value, Fp8Format format) {
    return float16_bits_to_float(fp8_bits_to_float16_bits(value, format));
}

// NCCL computes FP8 arithmetic in float16 and stores the result back as FP8
// (reduce_kernel.h: __nv_fp8_e4m3(__hadd(__half(x), __half(y))) and the
// __hmul PreOp). The exact float result is rounded to float16 first, then
// converted with __NV_SATFINITE as the __nv_fp8_e4m3(__half) constructor does.
inline std::uint8_t fp8_bits_from_float_via_float16(float value, Fp8Format format) {
    return float_to_fp8_bits(float16_bits_to_float(float_to_float16_bits(value)), format);
}

}  // namespace fake_gpu::distributed
