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

}  // namespace fake_gpu::distributed
