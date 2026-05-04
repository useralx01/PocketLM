#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <immintrin.h>
#include <intrin.h>
#include <new>
#include <omp.h>

static float fp16_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t out = 0;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t m = mant;
            int e = -14;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            const uint32_t out_exp = static_cast<uint32_t>(e + 127);
            out = sign | (out_exp << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        const uint32_t out_exp = exp + (127 - 15);
        out = sign | (out_exp << 23) | (mant << 13);
    }

    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

static uint16_t float_to_fp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mant = bits & 0x007FFFFFu;
    int exp = static_cast<int>((bits >> 23) & 0xFFu);

    if (exp == 255) {
        if (mant == 0) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        return static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13) | 1u);
    }

    exp = exp - 127 + 15;
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) {
            ++half_mant;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }

    uint32_t half_exp = static_cast<uint32_t>(exp) << 10;
    uint32_t half_mant = mant >> 13;
    const uint32_t round_bits = mant & 0x1FFFu;
    if (round_bits > 0x1000u || (round_bits == 0x1000u && (half_mant & 1u))) {
        ++half_mant;
        if (half_mant == 0x0400u) {
            half_mant = 0;
            half_exp += 0x0400u;
            if (half_exp >= 0x7C00u) {
                return static_cast<uint16_t>(sign | 0x7C00u);
            }
        }
    }
    return static_cast<uint16_t>(sign | half_exp | half_mant);
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
    const bool avx2 = (info[1] & (1 << 5)) != 0;
    return avx2;
}

extern "C" __declspec(dllexport) int q4_cpu_has_avx2_f16c() {
    return cpu_has_avx2_f16c() ? 1 : 0;
}

static inline int8_t unpack_int4_scalar(const uint8_t* packed, int64_t value_index) {
    const uint8_t byte = packed[value_index / 2];
    uint8_t nibble = (value_index % 2 == 0) ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
    int8_t q = static_cast<int8_t>(nibble);
    if (q >= 8) {
        q = static_cast<int8_t>(q - 16);
    }
    return q;
}

static inline void dequant_16_values_avx2(
    const uint8_t* packed,
    uint16_t* out_fp16,
    int64_t value_index,
    const __m256 scale
) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed + value_index / 2));
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i bias = _mm_set1_epi8(0x08);
    const __m128i low_unsigned = _mm_and_si128(bytes, mask);
    const __m128i high_unsigned = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    const __m128i low_signed = _mm_sub_epi8(_mm_xor_si128(low_unsigned, bias), bias);
    const __m128i high_signed = _mm_sub_epi8(_mm_xor_si128(high_unsigned, bias), bias);

    const __m256 low_f = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low_signed)), scale);
    const __m256 high_f = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high_signed)), scale);
    const __m128i low_h = _mm256_cvtps_ph(low_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m128i high_h = _mm256_cvtps_ph(high_f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m128i out_lo = _mm_unpacklo_epi16(low_h, high_h);
    const __m128i out_hi = _mm_unpackhi_epi16(low_h, high_h);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_fp16 + value_index), out_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_fp16 + value_index + 8), out_hi);
}

static void q4_dequant_to_fp16_scalar(
    const uint8_t* packed,
    const uint16_t* scales,
    uint16_t* out_fp16,
    int64_t num_channels,
    int64_t channel_size
) {
    for (int64_t c = 0; c < num_channels; ++c) {
        const float scale = fp16_to_float(scales[c]);
        const int64_t channel_offset = c * channel_size;
        for (int64_t i = 0; i < channel_size; ++i) {
            const int64_t value_index = channel_offset + i;
            out_fp16[value_index] = float_to_fp16(static_cast<float>(unpack_int4_scalar(packed, value_index)) * scale);
        }
    }
}

extern "C" __declspec(dllexport) void q4_dequant_to_fp16(
    const uint8_t* packed,
    const uint16_t* scales,
    uint16_t* out_fp16,
    int64_t num_channels,
    int64_t channel_size
) {
    if (!cpu_has_avx2_f16c()) {
        q4_dequant_to_fp16_scalar(packed, scales, out_fp16, num_channels, channel_size);
        return;
    }

    const char* thread_env = std::getenv("PCKETLM_NATIVE_THREADS");
    if (thread_env != nullptr) {
        const int requested = std::atoi(thread_env);
        if (requested > 0) {
            omp_set_num_threads(requested);
        }
    }

#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < num_channels; ++c) {
        const float scale = fp16_to_float(scales[c]);
        const __m256 scale_v = _mm256_set1_ps(scale);
        const int64_t channel_offset = c * channel_size;

        int64_t i = 0;
        if ((channel_offset % 2) == 0) {
            for (; i + 16 <= channel_size; i += 16) {
                dequant_16_values_avx2(packed, out_fp16, channel_offset + i, scale_v);
            }
        }
        for (; i < channel_size; ++i) {
            const int64_t value_index = channel_offset + i;
            out_fp16[value_index] = float_to_fp16(static_cast<float>(unpack_int4_scalar(packed, value_index)) * scale);
        }
    }
}

extern "C" __declspec(dllexport) int q4_dequant_many_to_fp16(
    const uint8_t** packed_ptrs,
    const uint16_t** scale_ptrs,
    uint16_t** out_ptrs,
    const int64_t* num_channels,
    const int64_t* channel_sizes,
    int64_t tensor_count
) {
    if (
        packed_ptrs == nullptr || scale_ptrs == nullptr || out_ptrs == nullptr ||
        num_channels == nullptr || channel_sizes == nullptr
    ) {
        return 1;
    }
    if (tensor_count <= 0) {
        return 2;
    }

    const char* thread_env = std::getenv("PCKETLM_NATIVE_THREADS");
    if (thread_env != nullptr) {
        const int requested = std::atoi(thread_env);
        if (requested > 0) {
            omp_set_num_threads(requested);
        }
    }

    if (!cpu_has_avx2_f16c()) {
        for (int64_t t = 0; t < tensor_count; ++t) {
            if (
                packed_ptrs[t] == nullptr || scale_ptrs[t] == nullptr || out_ptrs[t] == nullptr ||
                num_channels[t] <= 0 || channel_sizes[t] <= 0
            ) {
                return 3;
            }
            q4_dequant_to_fp16_scalar(
                packed_ptrs[t],
                scale_ptrs[t],
                out_ptrs[t],
                num_channels[t],
                channel_sizes[t]
            );
        }
        return 0;
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t t = 0; t < tensor_count; ++t) {
        if (
            packed_ptrs[t] == nullptr || scale_ptrs[t] == nullptr || out_ptrs[t] == nullptr ||
            num_channels[t] <= 0 || channel_sizes[t] <= 0
        ) {
            continue;
        }
        const uint8_t* packed = packed_ptrs[t];
        const uint16_t* scales = scale_ptrs[t];
        uint16_t* out_fp16 = out_ptrs[t];
        const int64_t channels = num_channels[t];
        const int64_t channel_size = channel_sizes[t];
        for (int64_t c = 0; c < channels; ++c) {
            const float scale = fp16_to_float(scales[c]);
            const __m256 scale_v = _mm256_set1_ps(scale);
            const int64_t channel_offset = c * channel_size;

            int64_t i = 0;
            if ((channel_offset % 2) == 0) {
                for (; i + 16 <= channel_size; i += 16) {
                    dequant_16_values_avx2(packed, out_fp16, channel_offset + i, scale_v);
                }
            }
            for (; i < channel_size; ++i) {
                const int64_t value_index = channel_offset + i;
                out_fp16[value_index] = float_to_fp16(static_cast<float>(unpack_int4_scalar(packed, value_index)) * scale);
            }
        }
    }
    return 0;
}

static inline float q4_dot_row_scalar(
    const uint8_t* packed,
    const uint16_t* scales,
    int64_t row,
    int64_t row_size,
    const float* input
) {
    const float scale = fp16_to_float(scales[row]);
    const int64_t row_offset = row * row_size;
    float sum = 0.0f;
    for (int64_t i = 0; i < row_size; ++i) {
        const int64_t value_index = row_offset + i;
        sum += static_cast<float>(unpack_int4_scalar(packed, value_index)) * scale * input[i];
    }
    return sum;
}

static inline float hsum256_ps(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline float q4_dot_row_avx2(
    const uint8_t* packed,
    const uint16_t* scales,
    int64_t row,
    int64_t row_size,
    const float* input
) {
    const float scale_f = fp16_to_float(scales[row]);
    const __m256 scale = _mm256_set1_ps(scale_f);
    const int64_t row_offset = row * row_size;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 15 < row_size; i += 16) {
        const int64_t value_index = row_offset + i;
        const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed + value_index / 2));
        const __m128i mask = _mm_set1_epi8(0x0F);
        const __m128i bias = _mm_set1_epi8(0x08);
        const __m128i low_unsigned = _mm_and_si128(bytes, mask);
        const __m128i high_unsigned = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        const __m128i low_signed = _mm_sub_epi8(_mm_xor_si128(low_unsigned, bias), bias);
        const __m128i high_signed = _mm_sub_epi8(_mm_xor_si128(high_unsigned, bias), bias);

        const __m256 low_f = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low_signed)), scale);
        const __m256 high_f = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high_signed)), scale);
        const __m256 interleaved_lo = _mm256_unpacklo_ps(low_f, high_f);
        const __m256 interleaved_hi = _mm256_unpackhi_ps(low_f, high_f);
        const __m256 q0 = _mm256_permute2f128_ps(interleaved_lo, interleaved_hi, 0x20);
        const __m256 q1 = _mm256_permute2f128_ps(interleaved_lo, interleaved_hi, 0x31);

        const __m256 in0 = _mm256_loadu_ps(input + i);
        const __m256 in1 = _mm256_loadu_ps(input + i + 8);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(q0, in0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(q1, in1));
    }
    float sum = hsum256_ps(_mm256_add_ps(acc0, acc1));
    for (; i < row_size; ++i) {
        const int64_t value_index = row_offset + i;
        sum += static_cast<float>(unpack_int4_scalar(packed, value_index)) * scale_f * input[i];
    }
    return sum;
}

extern "C" __declspec(dllexport) int q4_moe_selected_forward_u16(
    const uint16_t* hidden,
    const uint8_t** gate_packed_ptrs,
    const uint16_t** gate_scale_ptrs,
    const uint8_t** up_packed_ptrs,
    const uint16_t** up_scale_ptrs,
    const uint8_t** down_packed_ptrs,
    const uint16_t** down_scale_ptrs,
    const float* route_weights,
    uint16_t* out_fp16,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t selected_count
) {
    if (
        hidden == nullptr || gate_packed_ptrs == nullptr || gate_scale_ptrs == nullptr ||
        up_packed_ptrs == nullptr || up_scale_ptrs == nullptr ||
        down_packed_ptrs == nullptr || down_scale_ptrs == nullptr ||
        route_weights == nullptr || out_fp16 == nullptr ||
        hidden_size <= 0 || intermediate_size <= 0 || selected_count <= 0
    ) {
        return 1;
    }

    const char* thread_env = std::getenv("PCKETLM_NATIVE_THREADS");
    if (thread_env != nullptr) {
        const int requested = std::atoi(thread_env);
        if (requested > 0) {
            omp_set_num_threads(requested);
        }
    }
    const bool use_avx2 = cpu_has_avx2_f16c();

    float* hidden_f = new (std::nothrow) float[hidden_size];
    float* combined = new (std::nothrow) float[hidden_size];
    float* gate_values = new (std::nothrow) float[intermediate_size];
    float* up_values = new (std::nothrow) float[intermediate_size];
    float* expert_hidden = new (std::nothrow) float[intermediate_size];
    if (
        hidden_f == nullptr || combined == nullptr || gate_values == nullptr ||
        up_values == nullptr || expert_hidden == nullptr
    ) {
        delete[] hidden_f;
        delete[] combined;
        delete[] gate_values;
        delete[] up_values;
        delete[] expert_hidden;
        return 2;
    }

    for (int64_t i = 0; i < hidden_size; ++i) {
        hidden_f[i] = fp16_to_float(hidden[i]);
        combined[i] = 0.0f;
    }

    for (int64_t expert = 0; expert < selected_count; ++expert) {
        const uint8_t* gate_packed = gate_packed_ptrs[expert];
        const uint16_t* gate_scales = gate_scale_ptrs[expert];
        const uint8_t* up_packed = up_packed_ptrs[expert];
        const uint16_t* up_scales = up_scale_ptrs[expert];
        const uint8_t* down_packed = down_packed_ptrs[expert];
        const uint16_t* down_scales = down_scale_ptrs[expert];
        if (
            gate_packed == nullptr || gate_scales == nullptr || up_packed == nullptr ||
            up_scales == nullptr || down_packed == nullptr || down_scales == nullptr
        ) {
            delete[] hidden_f;
            delete[] combined;
            delete[] gate_values;
            delete[] up_values;
            delete[] expert_hidden;
            return 3;
        }

#pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < intermediate_size; ++row) {
            gate_values[row] = use_avx2
                ? q4_dot_row_avx2(gate_packed, gate_scales, row, hidden_size, hidden_f)
                : q4_dot_row_scalar(gate_packed, gate_scales, row, hidden_size, hidden_f);
            up_values[row] = use_avx2
                ? q4_dot_row_avx2(up_packed, up_scales, row, hidden_size, hidden_f)
                : q4_dot_row_scalar(up_packed, up_scales, row, hidden_size, hidden_f);
        }

        for (int64_t row = 0; row < intermediate_size; ++row) {
            const float gate = gate_values[row];
            const float silu = gate / (1.0f + std::exp(-gate));
            expert_hidden[row] = silu * up_values[row];
        }

        const float route = route_weights[expert];
#pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < hidden_size; ++row) {
            const float down = use_avx2
                ? q4_dot_row_avx2(down_packed, down_scales, row, intermediate_size, expert_hidden)
                : q4_dot_row_scalar(down_packed, down_scales, row, intermediate_size, expert_hidden);
            combined[row] += route * down;
        }
    }

    for (int64_t i = 0; i < hidden_size; ++i) {
        out_fp16[i] = float_to_fp16(combined[i]);
    }

    delete[] hidden_f;
    delete[] combined;
    delete[] gate_values;
    delete[] up_values;
    delete[] expert_hidden;
    return 0;
}
