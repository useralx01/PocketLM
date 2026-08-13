#include <algorithm>
#include <cstdint>
#include <cmath>
#include <immintrin.h>
#include <intrin.h>
#include <new>
#include <omp.h>

static inline float fp8_e4m3fn_to_float(uint8_t byte) {
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
        return 0.0f;
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

static bool cpu_has_avx512_runtime() {
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, 0, 0);
    if (regs[0] < 7) {
        return false;
    }

    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    if (!osxsave) {
        return false;
    }
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0xE6) != 0xE6) {
        return false;
    }

    __cpuidex(regs, 7, 0);
    const bool avx512f = (regs[1] & (1 << 16)) != 0;
    const bool avx512bw = (regs[1] & (1 << 30)) != 0;
    return avx512f && avx512bw;
}

extern "C" __declspec(dllexport) int fp8_linear_avx512_cpu_has_avx512() {
    return cpu_has_avx512_runtime() ? 1 : 0;
}

static inline __m512 fp8_decode16_e4m3fn(__m128i bytes) {
    const __m512i byte_values = _mm512_cvtepu8_epi32(bytes);
    const __m512i sign_bits = _mm512_slli_epi32(_mm512_and_si512(byte_values, _mm512_set1_epi32(0x80)), 24);
    const __m512i exp_bits = _mm512_and_si512(_mm512_srli_epi32(byte_values, 3), _mm512_set1_epi32(0x0F));
    const __m512i mant_bits = _mm512_and_si512(byte_values, _mm512_set1_epi32(0x07));

    const __m512i normal_exp = _mm512_slli_epi32(_mm512_add_epi32(exp_bits, _mm512_set1_epi32(120)), 23);
    const __m512i normal_mant = _mm512_slli_epi32(mant_bits, 20);
    const __m512i normal_bits = _mm512_or_si512(sign_bits, _mm512_or_si512(normal_exp, normal_mant));

    const __m512 subnormal = _mm512_mul_ps(_mm512_cvtepi32_ps(mant_bits), _mm512_set1_ps(0.001953125f));
    const __m512i subnormal_bits = _mm512_xor_si512(_mm512_castps_si512(subnormal), sign_bits);
    const __mmask16 subnormal_mask = _mm512_cmpeq_epi32_mask(exp_bits, _mm512_setzero_si512());
    const __mmask16 nan_mask = _mm512_kand(
        _mm512_cmpeq_epi32_mask(exp_bits, _mm512_set1_epi32(0x0F)),
        _mm512_cmpeq_epi32_mask(mant_bits, _mm512_set1_epi32(0x07))
    );

    __m512i decoded_bits = _mm512_mask_mov_epi32(normal_bits, subnormal_mask, subnormal_bits);
    decoded_bits = _mm512_mask_mov_epi32(decoded_bits, nan_mask, _mm512_setzero_si512());
    return _mm512_castsi512_ps(decoded_bits);
}

static inline void fp8_mul16_products(
    const uint8_t* weight,
    const float* values,
    const float* lut,
    float* products
) {
    (void)lut;
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight));
    const __m512 decoded = fp8_decode16_e4m3fn(bytes);
    const __m512 input = _mm512_loadu_ps(values);
    _mm512_storeu_ps(products, _mm512_mul_ps(input, decoded));
}

static inline void dot2_fp8_lut_avx512_exact(
    const uint8_t* gate_row,
    const uint8_t* up_row,
    const float* hidden_row,
    const float* lut,
    int64_t start_col,
    int64_t end_col,
    float* gate_block,
    float* up_block
) {
    alignas(64) float gate_products[16];
    alignas(64) float up_products[16];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    int64_t col = start_col;
    for (; col + 16 <= end_col; col += 16) {
        fp8_mul16_products(gate_row + col, hidden_row + col, lut, gate_products);
        fp8_mul16_products(up_row + col, hidden_row + col, lut, up_products);
        for (int lane = 0; lane < 16; ++lane) {
            gate_acc += gate_products[lane];
            up_acc += up_products[lane];
        }
    }
    for (; col < end_col; ++col) {
        const float h = hidden_row[col];
        gate_acc += h * lut[gate_row[col]];
        up_acc += h * lut[up_row[col]];
    }
    *gate_block = gate_acc;
    *up_block = up_acc;
}

static inline float dot_fp8_lut_avx512_exact(
    const uint8_t* weight_row,
    const float* values,
    const float* lut,
    int64_t start_col,
    int64_t end_col
) {
    alignas(64) float products[16];
    float acc = 0.0f;
    int64_t col = start_col;
    for (; col + 16 <= end_col; col += 16) {
        fp8_mul16_products(weight_row + col, values + col, lut, products);
        for (int lane = 0; lane < 16; ++lane) {
            acc += products[lane];
        }
    }
    for (; col < end_col; ++col) {
        acc += values[col] * lut[weight_row[col]];
    }
    return acc;
}

extern "C" __declspec(dllexport) int fp8_e4m3_block_mlp_avx512_f32(
    const uint8_t* gate_weight,
    const float* gate_scale_inv,
    const uint8_t* up_weight,
    const float* up_scale_inv,
    const uint8_t* down_weight,
    const float* down_scale_inv,
    const float* hidden,
    float* out,
    int64_t batch,
    int64_t intermediate_rows,
    int64_t hidden_cols,
    int64_t gate_scale_cols,
    int64_t down_scale_cols
) {
    if (!cpu_has_avx512_runtime()) {
        return -4;
    }
    if (
        gate_weight == nullptr || gate_scale_inv == nullptr ||
        up_weight == nullptr || up_scale_inv == nullptr ||
        down_weight == nullptr || down_scale_inv == nullptr ||
        hidden == nullptr || out == nullptr
    ) {
        return -1;
    }
    if (
        batch <= 0 || intermediate_rows <= 0 || hidden_cols <= 0 ||
        gate_scale_cols <= 0 || down_scale_cols <= 0
    ) {
        return -2;
    }

    const float* lut = fp8_e4m3fn_lut();
    float* activation = new (std::nothrow) float[static_cast<size_t>(batch * intermediate_rows)];
    if (activation == nullptr) {
        return -3;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t row = 0; row < intermediate_rows; ++row) {
            const int64_t scale_row = row / 128;
            const uint8_t* gate_row = gate_weight + row * hidden_cols;
            const uint8_t* up_row = up_weight + row * hidden_cols;
            const float* hidden_row = hidden + b * hidden_cols;
            float gate_acc = 0.0f;
            float up_acc = 0.0f;
            for (int64_t scale_col = 0; scale_col < gate_scale_cols; ++scale_col) {
                const int64_t start_col = scale_col * 128;
                const int64_t end_col = std::min<int64_t>(hidden_cols, start_col + 128);
                const float gate_scale = gate_scale_inv[scale_row * gate_scale_cols + scale_col];
                const float up_scale = up_scale_inv[scale_row * gate_scale_cols + scale_col];
                float gate_block = 0.0f;
                float up_block = 0.0f;
                dot2_fp8_lut_avx512_exact(
                    gate_row,
                    up_row,
                    hidden_row,
                    lut,
                    start_col,
                    end_col,
                    &gate_block,
                    &up_block
                );
                gate_acc += gate_block * gate_scale;
                up_acc += up_block * up_scale;
            }
            const float silu = gate_acc / (1.0f + std::exp(-gate_acc));
            activation[b * intermediate_rows + row] = silu * up_acc;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t row = 0; row < hidden_cols; ++row) {
            const int64_t scale_row = row / 128;
            const uint8_t* down_row = down_weight + row * intermediate_rows;
            const float* activation_row = activation + b * intermediate_rows;
            float acc = 0.0f;
            for (int64_t scale_col = 0; scale_col < down_scale_cols; ++scale_col) {
                const int64_t start_col = scale_col * 128;
                const int64_t end_col = std::min<int64_t>(intermediate_rows, start_col + 128);
                const float scale = down_scale_inv[scale_row * down_scale_cols + scale_col];
                const float block_acc = dot_fp8_lut_avx512_exact(down_row, activation_row, lut, start_col, end_col);
                acc += block_acc * scale;
            }
            out[b * hidden_cols + row] = acc;
        }
    }

    delete[] activation;
    return 0;
}

extern "C" __declspec(dllexport) int fp8_e4m3_block_mlp_many_weighted_avx512_f32(
    const uint64_t* gate_weight_ptrs,
    const uint64_t* gate_scale_ptrs,
    const uint64_t* up_weight_ptrs,
    const uint64_t* up_scale_ptrs,
    const uint64_t* down_weight_ptrs,
    const uint64_t* down_scale_ptrs,
    const float* hidden,
    const float* route_weights,
    float* out,
    int64_t expert_count,
    int64_t batch,
    int64_t intermediate_rows,
    int64_t hidden_cols,
    int64_t gate_scale_cols,
    int64_t down_scale_cols
) {
    if (!cpu_has_avx512_runtime()) {
        return -4;
    }
    if (
        gate_weight_ptrs == nullptr || gate_scale_ptrs == nullptr ||
        up_weight_ptrs == nullptr || up_scale_ptrs == nullptr ||
        down_weight_ptrs == nullptr || down_scale_ptrs == nullptr ||
        hidden == nullptr || route_weights == nullptr || out == nullptr
    ) {
        return -1;
    }
    if (
        expert_count <= 0 || batch <= 0 || intermediate_rows <= 0 || hidden_cols <= 0 ||
        gate_scale_cols <= 0 || down_scale_cols <= 0
    ) {
        return -2;
    }

    const float* lut = fp8_e4m3fn_lut();
    const int64_t activation_count = expert_count * batch * intermediate_rows;
    float* activation = new (std::nothrow) float[static_cast<size_t>(activation_count)];
    if (activation == nullptr) {
        return -3;
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t row = 0; row < intermediate_rows; ++row) {
                const uint8_t* gate_weight = reinterpret_cast<const uint8_t*>(gate_weight_ptrs[expert]);
                const float* gate_scale_inv = reinterpret_cast<const float*>(gate_scale_ptrs[expert]);
                const uint8_t* up_weight = reinterpret_cast<const uint8_t*>(up_weight_ptrs[expert]);
                const float* up_scale_inv = reinterpret_cast<const float*>(up_scale_ptrs[expert]);
                if (
                    gate_weight == nullptr || gate_scale_inv == nullptr ||
                    up_weight == nullptr || up_scale_inv == nullptr
                ) {
                    activation[(expert * batch + b) * intermediate_rows + row] = 0.0f;
                    continue;
                }
                const int64_t scale_row = row / 128;
                const uint8_t* gate_row = gate_weight + row * hidden_cols;
                const uint8_t* up_row = up_weight + row * hidden_cols;
                const float* hidden_row = hidden + b * hidden_cols;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int64_t scale_col = 0; scale_col < gate_scale_cols; ++scale_col) {
                    const int64_t start_col = scale_col * 128;
                    const int64_t end_col = std::min<int64_t>(hidden_cols, start_col + 128);
                    const float gate_scale = gate_scale_inv[scale_row * gate_scale_cols + scale_col];
                    const float up_scale = up_scale_inv[scale_row * gate_scale_cols + scale_col];
                    float gate_block = 0.0f;
                    float up_block = 0.0f;
                    dot2_fp8_lut_avx512_exact(
                        gate_row,
                        up_row,
                        hidden_row,
                        lut,
                        start_col,
                        end_col,
                        &gate_block,
                        &up_block
                    );
                    gate_acc += gate_block * gate_scale;
                    up_acc += up_block * up_scale;
                }
                const float silu = gate_acc / (1.0f + std::exp(-gate_acc));
                activation[(expert * batch + b) * intermediate_rows + row] = silu * up_acc;
            }
        }
    }

    const int64_t batch_width = batch * hidden_cols;
    #pragma omp parallel for schedule(static)
    for (int64_t offset = 0; offset < batch_width; ++offset) {
        const int64_t b = offset / hidden_cols;
        const int64_t row = offset - b * hidden_cols;
        const int64_t scale_row = row / 128;
        float routed_acc = 0.0f;
        for (int64_t expert = 0; expert < expert_count; ++expert) {
            const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(down_weight_ptrs[expert]);
            const float* down_scale_inv = reinterpret_cast<const float*>(down_scale_ptrs[expert]);
            if (down_weight == nullptr || down_scale_inv == nullptr) {
                continue;
            }
            const uint8_t* down_row = down_weight + row * intermediate_rows;
            const float* activation_row = activation + (expert * batch + b) * intermediate_rows;
            float expert_acc = 0.0f;
            for (int64_t scale_col = 0; scale_col < down_scale_cols; ++scale_col) {
                const int64_t start_col = scale_col * 128;
                const int64_t end_col = std::min<int64_t>(intermediate_rows, start_col + 128);
                const float scale = down_scale_inv[scale_row * down_scale_cols + scale_col];
                const float block_acc = dot_fp8_lut_avx512_exact(down_row, activation_row, lut, start_col, end_col);
                expert_acc += block_acc * scale;
            }
            routed_acc += route_weights[expert] * expert_acc;
        }
        out[offset] = routed_acc;
    }

    delete[] activation;
    return 0;
}
