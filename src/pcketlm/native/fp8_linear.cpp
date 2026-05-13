#include <cstdint>
#include <cmath>
#include <algorithm>
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
