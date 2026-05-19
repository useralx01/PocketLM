#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <intrin.h>
#include <omp.h>

static float fp8_e4m3fn_to_float(uint8_t byte) {
    const int sign = (byte & 0x80u) ? -1 : 1;
    const int exp = static_cast<int>((byte >> 3) & 0x0Fu);
    const int mant = static_cast<int>(byte & 0x07u);

    if (exp == 0) {
        if (mant == 0) {
            return sign < 0 ? -0.0f : 0.0f;
        }
        return static_cast<float>(sign) * std::ldexp(static_cast<float>(mant) / 8.0f, -6);
    }

    if (exp == 0x0F && mant == 0x07) {
        const uint32_t bits = sign < 0 ? 0xFFF00000u : 0x7FF00000u;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    return static_cast<float>(sign) * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

static const float* fp8_e4m3fn_lut() {
    static float table[256] = {0.0f};
    static bool initialized = false;
    if (!initialized) {
        for (int i = 0; i < 256; ++i) {
            table[i] = fp8_e4m3fn_to_float(static_cast<uint8_t>(i));
        }
        initialized = true;
    }
    return table;
}

static bool cpu_has_avx2_f16c() {
    int info[4] = {0, 0, 0, 0};
    __cpuid(info, 0);
    if (info[0] < 7) {
        return false;
    }

    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    const bool f16c = (info[2] & (1 << 29)) != 0;
    if (!osxsave || !avx || !f16c) {
        return false;
    }
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return false;
    }

    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
}

extern "C" __declspec(dllexport) int fp8_dequant_cpu_has_avx2_f16c() {
    return cpu_has_avx2_f16c() ? 1 : 0;
}

static void set_native_threads_from_env() {
    const char* thread_env = std::getenv("PCKETLM_NATIVE_THREADS");
    if (thread_env != nullptr) {
        const int requested = std::atoi(thread_env);
        if (requested > 0) {
            omp_set_num_threads(requested);
        }
    }
}

static inline void store_8_fp16(
    const uint8_t* input,
    uint16_t* output,
    const float* lut,
    const float scale
) {
    const __m256 values = _mm256_set_ps(
        lut[input[7]],
        lut[input[6]],
        lut[input[5]],
        lut[input[4]],
        lut[input[3]],
        lut[input[2]],
        lut[input[1]],
        lut[input[0]]
    );
    const __m256 scaled = _mm256_mul_ps(values, _mm256_set1_ps(scale));
    const __m128i half = _mm256_cvtps_ph(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), half);
}

static uint16_t float_to_fp16_scalar(float value) {
    const __m128 v = _mm_set_ss(value);
    const __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
}

static uint16_t float_to_bf16_scalar(float value) {
    if (std::isnan(value)) {
        return 0xFFFFu;
    }
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

extern "C" __declspec(dllexport) int fp8_e4m3_dequant_to_fp16(
    const uint8_t* fp8_weight,
    const float* scale_inv,
    uint16_t* out_fp16,
    int64_t rows,
    int64_t cols,
    int64_t scale_cols
) {
    if (fp8_weight == nullptr || scale_inv == nullptr || out_fp16 == nullptr) {
        return -1;
    }
    if (rows <= 0 || cols <= 0 || scale_cols <= 0) {
        return -2;
    }
    const int64_t expected_scale_cols = (cols + 127) / 128;
    if (scale_cols != expected_scale_cols) {
        return -3;
    }
    if (!cpu_has_avx2_f16c()) {
        return -4;
    }

    set_native_threads_from_env();
    const float* lut = fp8_e4m3fn_lut();

#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t scale_row = row / 128;
        const uint8_t* weight_row = fp8_weight + row * cols;
        uint16_t* out_row = out_fp16 + row * cols;
        for (int64_t scale_col = 0; scale_col < scale_cols; ++scale_col) {
            const int64_t start_col = scale_col * 128;
            const int64_t end_col = std::min<int64_t>(cols, start_col + 128);
            const float scale = scale_inv[scale_row * scale_cols + scale_col];
            int64_t col = start_col;
            for (; col + 8 <= end_col; col += 8) {
                store_8_fp16(weight_row + col, out_row + col, lut, scale);
            }
            for (; col < end_col; ++col) {
                out_row[col] = float_to_fp16_scalar(lut[weight_row[col]] * scale);
            }
        }
    }

    return 0;
}

extern "C" __declspec(dllexport) int fp8_e4m3_dequant_to_bf16(
    const uint8_t* fp8_weight,
    const float* scale_inv,
    uint16_t* out_bf16,
    int64_t rows,
    int64_t cols,
    int64_t scale_cols
) {
    if (fp8_weight == nullptr || scale_inv == nullptr || out_bf16 == nullptr) {
        return -1;
    }
    if (rows <= 0 || cols <= 0 || scale_cols <= 0) {
        return -2;
    }
    const int64_t expected_scale_cols = (cols + 127) / 128;
    if (scale_cols != expected_scale_cols) {
        return -3;
    }

    set_native_threads_from_env();
    const float* lut = fp8_e4m3fn_lut();

#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t scale_row = row / 128;
        const uint8_t* weight_row = fp8_weight + row * cols;
        uint16_t* out_row = out_bf16 + row * cols;
        for (int64_t scale_col = 0; scale_col < scale_cols; ++scale_col) {
            const int64_t start_col = scale_col * 128;
            const int64_t end_col = std::min<int64_t>(cols, start_col + 128);
            const float scale = scale_inv[scale_row * scale_cols + scale_col];
            for (int64_t col = start_col; col < end_col; ++col) {
                out_row[col] = float_to_bf16_scalar(lut[weight_row[col]] * scale);
            }
        }
    }

    return 0;
}
