#include <cstdint>
#include <cmath>
#include <algorithm>
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

extern "C" __declspec(dllexport) int fp8_e4m3_block_linear_f32(
    const uint8_t* fp8_weight,
    const float* scale_inv,
    const float* hidden,
    float* out,
    int64_t batch,
    int64_t out_rows,
    int64_t in_cols,
    int64_t scale_cols
) {
    if (fp8_weight == nullptr || scale_inv == nullptr || hidden == nullptr || out == nullptr) {
        return -1;
    }
    if (batch <= 0 || out_rows <= 0 || in_cols <= 0 || scale_cols <= 0) {
        return -2;
    }

    const float* lut = fp8_e4m3fn_lut();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t row = 0; row < out_rows; ++row) {
            const int64_t scale_row = row / 128;
            const uint8_t* weight_row = fp8_weight + row * in_cols;
            const float* hidden_row = hidden + b * in_cols;
            float acc = 0.0f;
            for (int64_t scale_col = 0; scale_col < scale_cols; ++scale_col) {
                const int64_t start_col = scale_col * 128;
                const int64_t end_col = std::min<int64_t>(in_cols, start_col + 128);
                const float scale = scale_inv[scale_row * scale_cols + scale_col];
                float block_acc = 0.0f;
                for (int64_t col = start_col; col < end_col; ++col) {
                    block_acc += hidden_row[col] * lut[weight_row[col]];
                }
                acc += block_acc * scale;
            }
            out[b * out_rows + row] = acc;
        }
    }
    return 0;
}

extern "C" __declspec(dllexport) int fp8_e4m3_block_dual_linear_f32(
    const uint8_t* fp8_weight_a,
    const float* scale_inv_a,
    const uint8_t* fp8_weight_b,
    const float* scale_inv_b,
    const float* hidden,
    float* out_a,
    float* out_b,
    int64_t batch,
    int64_t out_rows,
    int64_t in_cols,
    int64_t scale_cols
) {
    if (
        fp8_weight_a == nullptr || scale_inv_a == nullptr ||
        fp8_weight_b == nullptr || scale_inv_b == nullptr ||
        hidden == nullptr || out_a == nullptr || out_b == nullptr
    ) {
        return -1;
    }
    if (batch <= 0 || out_rows <= 0 || in_cols <= 0 || scale_cols <= 0) {
        return -2;
    }

    const float* lut = fp8_e4m3fn_lut();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t row = 0; row < out_rows; ++row) {
            const int64_t scale_row = row / 128;
            const uint8_t* weight_row_a = fp8_weight_a + row * in_cols;
            const uint8_t* weight_row_b = fp8_weight_b + row * in_cols;
            const float* hidden_row = hidden + b * in_cols;
            float acc_a = 0.0f;
            float acc_b = 0.0f;
            for (int64_t scale_col = 0; scale_col < scale_cols; ++scale_col) {
                const int64_t start_col = scale_col * 128;
                const int64_t end_col = std::min<int64_t>(in_cols, start_col + 128);
                const float scale_a = scale_inv_a[scale_row * scale_cols + scale_col];
                const float scale_b = scale_inv_b[scale_row * scale_cols + scale_col];
                float block_acc_a = 0.0f;
                float block_acc_b = 0.0f;
                for (int64_t col = start_col; col < end_col; ++col) {
                    const float h = hidden_row[col];
                    block_acc_a += h * lut[weight_row_a[col]];
                    block_acc_b += h * lut[weight_row_b[col]];
                }
                acc_a += block_acc_a * scale_a;
                acc_b += block_acc_b * scale_b;
            }
            out_a[b * out_rows + row] = acc_a;
            out_b[b * out_rows + row] = acc_b;
        }
    }
    return 0;
}

extern "C" __declspec(dllexport) int fp8_e4m3_block_mlp_f32(
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
                for (int64_t col = start_col; col < end_col; ++col) {
                    const float h = hidden_row[col];
                    gate_block += h * lut[gate_row[col]];
                    up_block += h * lut[up_row[col]];
                }
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
                float block_acc = 0.0f;
                for (int64_t col = start_col; col < end_col; ++col) {
                    block_acc += activation_row[col] * lut[down_row[col]];
                }
                acc += block_acc * scale;
            }
            out[b * hidden_cols + row] = acc;
        }
    }

    delete[] activation;
    return 0;
}
