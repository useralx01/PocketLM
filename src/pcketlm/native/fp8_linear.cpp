#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <new>
#include <omp.h>

static constexpr int64_t kMaxBatchReuse = 64;

static bool fp8_batch_reuse_enabled() {
    const char* raw = std::getenv("PCKETLM_ENABLE_NATIVE_FP8_BATCH_REUSE");
    if (raw == nullptr) {
        return false;
    }
    const char value = raw[0];
    return value == '1' || value == 't' || value == 'T' || value == 'y' || value == 'Y';
}

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
    if (fp8_batch_reuse_enabled() && batch > 1 && batch <= kMaxBatchReuse) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < out_rows; ++row) {
            const int64_t scale_row = row / 128;
            const uint8_t* weight_row = fp8_weight + row * in_cols;
            float accs[kMaxBatchReuse] = {0.0f};
            for (int64_t scale_col = 0; scale_col < scale_cols; ++scale_col) {
                const int64_t start_col = scale_col * 128;
                const int64_t end_col = std::min<int64_t>(in_cols, start_col + 128);
                const float scale = scale_inv[scale_row * scale_cols + scale_col];
                float blocks[kMaxBatchReuse] = {0.0f};
                for (int64_t col = start_col; col < end_col; ++col) {
                    const float w = lut[weight_row[col]];
                    for (int64_t b = 0; b < batch; ++b) {
                        blocks[b] += hidden[b * in_cols + col] * w;
                    }
                }
                for (int64_t b = 0; b < batch; ++b) {
                    accs[b] += blocks[b] * scale;
                }
            }
            for (int64_t b = 0; b < batch; ++b) {
                out[b * out_rows + row] = accs[b];
            }
        }
        return 0;
    }

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
    if (fp8_batch_reuse_enabled() && batch > 1 && batch <= kMaxBatchReuse) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < out_rows; ++row) {
            const int64_t scale_row = row / 128;
            const uint8_t* weight_row_a = fp8_weight_a + row * in_cols;
            const uint8_t* weight_row_b = fp8_weight_b + row * in_cols;
            float accs_a[kMaxBatchReuse] = {0.0f};
            float accs_b[kMaxBatchReuse] = {0.0f};
            for (int64_t scale_col = 0; scale_col < scale_cols; ++scale_col) {
                const int64_t start_col = scale_col * 128;
                const int64_t end_col = std::min<int64_t>(in_cols, start_col + 128);
                const float scale_a = scale_inv_a[scale_row * scale_cols + scale_col];
                const float scale_b = scale_inv_b[scale_row * scale_cols + scale_col];
                float blocks_a[kMaxBatchReuse] = {0.0f};
                float blocks_b[kMaxBatchReuse] = {0.0f};
                for (int64_t col = start_col; col < end_col; ++col) {
                    const float wa = lut[weight_row_a[col]];
                    const float wb = lut[weight_row_b[col]];
                    for (int64_t b = 0; b < batch; ++b) {
                        const float h = hidden[b * in_cols + col];
                        blocks_a[b] += h * wa;
                        blocks_b[b] += h * wb;
                    }
                }
                for (int64_t b = 0; b < batch; ++b) {
                    accs_a[b] += blocks_a[b] * scale_a;
                    accs_b[b] += blocks_b[b] * scale_b;
                }
            }
            for (int64_t b = 0; b < batch; ++b) {
                out_a[b * out_rows + row] = accs_a[b];
                out_b[b * out_rows + row] = accs_b[b];
            }
        }
        return 0;
    }

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

    if (fp8_batch_reuse_enabled() && batch > 1 && batch <= kMaxBatchReuse) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < intermediate_rows; ++row) {
            const int64_t scale_row = row / 128;
            const uint8_t* gate_row = gate_weight + row * hidden_cols;
            const uint8_t* up_row = up_weight + row * hidden_cols;
            float gate_accs[kMaxBatchReuse] = {0.0f};
            float up_accs[kMaxBatchReuse] = {0.0f};
            for (int64_t scale_col = 0; scale_col < gate_scale_cols; ++scale_col) {
                const int64_t start_col = scale_col * 128;
                const int64_t end_col = std::min<int64_t>(hidden_cols, start_col + 128);
                const float gate_scale = gate_scale_inv[scale_row * gate_scale_cols + scale_col];
                const float up_scale = up_scale_inv[scale_row * gate_scale_cols + scale_col];
                float gate_blocks[kMaxBatchReuse] = {0.0f};
                float up_blocks[kMaxBatchReuse] = {0.0f};
                for (int64_t col = start_col; col < end_col; ++col) {
                    const float gw = lut[gate_row[col]];
                    const float uw = lut[up_row[col]];
                    for (int64_t b = 0; b < batch; ++b) {
                        const float h = hidden[b * hidden_cols + col];
                        gate_blocks[b] += h * gw;
                        up_blocks[b] += h * uw;
                    }
                }
                for (int64_t b = 0; b < batch; ++b) {
                    gate_accs[b] += gate_blocks[b] * gate_scale;
                    up_accs[b] += up_blocks[b] * up_scale;
                }
            }
            for (int64_t b = 0; b < batch; ++b) {
                const float gate_acc = gate_accs[b];
                const float silu = gate_acc / (1.0f + std::exp(-gate_acc));
                activation[b * intermediate_rows + row] = silu * up_accs[b];
            }
        }

        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < hidden_cols; ++row) {
            const int64_t scale_row = row / 128;
            const uint8_t* down_row = down_weight + row * intermediate_rows;
            float accs[kMaxBatchReuse] = {0.0f};
            for (int64_t scale_col = 0; scale_col < down_scale_cols; ++scale_col) {
                const int64_t start_col = scale_col * 128;
                const int64_t end_col = std::min<int64_t>(intermediate_rows, start_col + 128);
                const float scale = down_scale_inv[scale_row * down_scale_cols + scale_col];
                float blocks[kMaxBatchReuse] = {0.0f};
                for (int64_t col = start_col; col < end_col; ++col) {
                    const float w = lut[down_row[col]];
                    for (int64_t b = 0; b < batch; ++b) {
                        blocks[b] += activation[b * intermediate_rows + col] * w;
                    }
                }
                for (int64_t b = 0; b < batch; ++b) {
                    accs[b] += blocks[b] * scale;
                }
            }
            for (int64_t b = 0; b < batch; ++b) {
                out[b * hidden_cols + row] = accs[b];
            }
        }
    } else {
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
    }

    delete[] activation;
    return 0;
}

extern "C" __declspec(dllexport) int fp8_e4m3_block_mlp_many_f32(
    const uint64_t* gate_weight_ptrs,
    const uint64_t* gate_scale_ptrs,
    const uint64_t* up_weight_ptrs,
    const uint64_t* up_scale_ptrs,
    const uint64_t* down_weight_ptrs,
    const uint64_t* down_scale_ptrs,
    const float* hidden,
    float* out,
    int64_t expert_count,
    int64_t batch,
    int64_t intermediate_rows,
    int64_t hidden_cols,
    int64_t gate_scale_cols,
    int64_t down_scale_cols
) {
    if (
        gate_weight_ptrs == nullptr || gate_scale_ptrs == nullptr ||
        up_weight_ptrs == nullptr || up_scale_ptrs == nullptr ||
        down_weight_ptrs == nullptr || down_scale_ptrs == nullptr ||
        hidden == nullptr || out == nullptr
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
                    for (int64_t col = start_col; col < end_col; ++col) {
                        const float h = hidden_row[col];
                        gate_block += h * lut[gate_row[col]];
                        up_block += h * lut[up_row[col]];
                    }
                    gate_acc += gate_block * gate_scale;
                    up_acc += up_block * up_scale;
                }
                const float silu = gate_acc / (1.0f + std::exp(-gate_acc));
                activation[(expert * batch + b) * intermediate_rows + row] = silu * up_acc;
            }
        }
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t row = 0; row < hidden_cols; ++row) {
                const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(down_weight_ptrs[expert]);
                const float* down_scale_inv = reinterpret_cast<const float*>(down_scale_ptrs[expert]);
                if (down_weight == nullptr || down_scale_inv == nullptr) {
                    continue;
                }
                const int64_t scale_row = row / 128;
                const uint8_t* down_row = down_weight + row * intermediate_rows;
                const float* activation_row = activation + (expert * batch + b) * intermediate_rows;
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
                out[(expert * batch + b) * hidden_cols + row] = acc;
            }
        }
    }

    delete[] activation;
    return 0;
}

extern "C" __declspec(dllexport) int fp8_e4m3_block_mlp_many_weighted_f32(
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
                    for (int64_t col = start_col; col < end_col; ++col) {
                        const float h = hidden_row[col];
                        gate_block += h * lut[gate_row[col]];
                        up_block += h * lut[up_row[col]];
                    }
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
                float block_acc = 0.0f;
                for (int64_t col = start_col; col < end_col; ++col) {
                    block_acc += activation_row[col] * lut[down_row[col]];
                }
                expert_acc += block_acc * scale;
            }
            routed_acc += route_weights[expert] * expert_acc;
        }
        out[offset] = routed_acc;
    }

    delete[] activation;
    return 0;
}

extern "C" __declspec(dllexport) int fp8_e4m3_block_mlp_many_row_weighted_f32(
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

    if (fp8_batch_reuse_enabled() && batch > 1 && batch <= kMaxBatchReuse) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t expert = 0; expert < expert_count; ++expert) {
            for (int64_t row = 0; row < intermediate_rows; ++row) {
                int64_t active[kMaxBatchReuse];
                int64_t active_count = 0;
                for (int64_t b = 0; b < batch; ++b) {
                    if (route_weights[expert * batch + b] != 0.0f) {
                        active[active_count++] = b;
                    }
                }
                if (active_count <= 0) {
                    continue;
                }
                const uint8_t* gate_weight = reinterpret_cast<const uint8_t*>(gate_weight_ptrs[expert]);
                const float* gate_scale_inv = reinterpret_cast<const float*>(gate_scale_ptrs[expert]);
                const uint8_t* up_weight = reinterpret_cast<const uint8_t*>(up_weight_ptrs[expert]);
                const float* up_scale_inv = reinterpret_cast<const float*>(up_scale_ptrs[expert]);
                if (
                    gate_weight == nullptr || gate_scale_inv == nullptr ||
                    up_weight == nullptr || up_scale_inv == nullptr
                ) {
                    for (int64_t index = 0; index < active_count; ++index) {
                        const int64_t b = active[index];
                        activation[(expert * batch + b) * intermediate_rows + row] = 0.0f;
                    }
                    continue;
                }
                const int64_t scale_row = row / 128;
                const uint8_t* gate_row = gate_weight + row * hidden_cols;
                const uint8_t* up_row = up_weight + row * hidden_cols;
                float gate_accs[kMaxBatchReuse] = {0.0f};
                float up_accs[kMaxBatchReuse] = {0.0f};
                for (int64_t scale_col = 0; scale_col < gate_scale_cols; ++scale_col) {
                    const int64_t start_col = scale_col * 128;
                    const int64_t end_col = std::min<int64_t>(hidden_cols, start_col + 128);
                    const float gate_scale = gate_scale_inv[scale_row * gate_scale_cols + scale_col];
                    const float up_scale = up_scale_inv[scale_row * gate_scale_cols + scale_col];
                    float gate_blocks[kMaxBatchReuse] = {0.0f};
                    float up_blocks[kMaxBatchReuse] = {0.0f};
                    for (int64_t col = start_col; col < end_col; ++col) {
                        const float gw = lut[gate_row[col]];
                        const float uw = lut[up_row[col]];
                        for (int64_t index = 0; index < active_count; ++index) {
                            const int64_t b = active[index];
                            const float h = hidden[b * hidden_cols + col];
                            gate_blocks[b] += h * gw;
                            up_blocks[b] += h * uw;
                        }
                    }
                    for (int64_t index = 0; index < active_count; ++index) {
                        const int64_t b = active[index];
                        gate_accs[b] += gate_blocks[b] * gate_scale;
                        up_accs[b] += up_blocks[b] * up_scale;
                    }
                }
                for (int64_t index = 0; index < active_count; ++index) {
                    const int64_t b = active[index];
                    const float gate_acc = gate_accs[b];
                    const float silu = gate_acc / (1.0f + std::exp(-gate_acc));
                    activation[(expert * batch + b) * intermediate_rows + row] = silu * up_accs[b];
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < hidden_cols; ++row) {
            const int64_t scale_row = row / 128;
            float routed_accs[kMaxBatchReuse] = {0.0f};
            for (int64_t expert = 0; expert < expert_count; ++expert) {
                int64_t active[kMaxBatchReuse];
                int64_t active_count = 0;
                for (int64_t b = 0; b < batch; ++b) {
                    if (route_weights[expert * batch + b] != 0.0f) {
                        active[active_count++] = b;
                    }
                }
                if (active_count <= 0) {
                    continue;
                }
                const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(down_weight_ptrs[expert]);
                const float* down_scale_inv = reinterpret_cast<const float*>(down_scale_ptrs[expert]);
                if (down_weight == nullptr || down_scale_inv == nullptr) {
                    continue;
                }
                const uint8_t* down_row = down_weight + row * intermediate_rows;
                float expert_accs[kMaxBatchReuse] = {0.0f};
                for (int64_t scale_col = 0; scale_col < down_scale_cols; ++scale_col) {
                    const int64_t start_col = scale_col * 128;
                    const int64_t end_col = std::min<int64_t>(intermediate_rows, start_col + 128);
                    const float scale = down_scale_inv[scale_row * down_scale_cols + scale_col];
                    float blocks[kMaxBatchReuse] = {0.0f};
                    for (int64_t col = start_col; col < end_col; ++col) {
                        const float w = lut[down_row[col]];
                        for (int64_t index = 0; index < active_count; ++index) {
                            const int64_t b = active[index];
                            const float* activation_row = activation + (expert * batch + b) * intermediate_rows;
                            blocks[b] += activation_row[col] * w;
                        }
                    }
                    for (int64_t index = 0; index < active_count; ++index) {
                        const int64_t b = active[index];
                        expert_accs[b] += blocks[b] * scale;
                    }
                }
                for (int64_t index = 0; index < active_count; ++index) {
                    const int64_t b = active[index];
                    routed_accs[b] += route_weights[expert * batch + b] * expert_accs[b];
                }
            }
            for (int64_t b = 0; b < batch; ++b) {
                out[b * hidden_cols + row] = routed_accs[b];
            }
        }

        delete[] activation;
        return 0;
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t row = 0; row < intermediate_rows; ++row) {
                if (route_weights[expert * batch + b] == 0.0f) {
                    continue;
                }
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
                    for (int64_t col = start_col; col < end_col; ++col) {
                        const float h = hidden_row[col];
                        gate_block += h * lut[gate_row[col]];
                        up_block += h * lut[up_row[col]];
                    }
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
            const float route_weight = route_weights[expert * batch + b];
            if (route_weight == 0.0f) {
                continue;
            }
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
                float block_acc = 0.0f;
                for (int64_t col = start_col; col < end_col; ++col) {
                    block_acc += activation_row[col] * lut[down_row[col]];
                }
                expert_acc += block_acc * scale;
            }
            routed_acc += route_weight * expert_acc;
        }
        out[offset] = routed_acc;
    }

    delete[] activation;
    return 0;
}

extern "C" __declspec(dllexport) int fp8_e4m3_block_mlp_many_pair_weighted_f32(
    const uint64_t* gate_weight_ptrs,
    const uint64_t* gate_scale_ptrs,
    const uint64_t* up_weight_ptrs,
    const uint64_t* up_scale_ptrs,
    const uint64_t* down_weight_ptrs,
    const uint64_t* down_scale_ptrs,
    const float* hidden,
    const int64_t* pair_item_indices,
    const int64_t* pair_row_indices,
    const float* pair_weights,
    float* out,
    int64_t expert_count,
    int64_t pair_count,
    int64_t batch,
    int64_t intermediate_rows,
    int64_t hidden_cols,
    int64_t gate_scale_cols,
    int64_t down_scale_cols
) {
    if (
        gate_weight_ptrs == nullptr || gate_scale_ptrs == nullptr ||
        up_weight_ptrs == nullptr || up_scale_ptrs == nullptr ||
        down_weight_ptrs == nullptr || down_scale_ptrs == nullptr ||
        hidden == nullptr || pair_item_indices == nullptr || pair_row_indices == nullptr ||
        pair_weights == nullptr || out == nullptr
    ) {
        return -1;
    }
    if (
        expert_count <= 0 || pair_count < 0 || batch <= 0 || intermediate_rows <= 0 ||
        hidden_cols <= 0 || gate_scale_cols <= 0 || down_scale_cols <= 0
    ) {
        return -2;
    }

    std::fill(out, out + batch * hidden_cols, 0.0f);
    if (pair_count == 0) {
        return 0;
    }
    for (int64_t pair = 0; pair < pair_count; ++pair) {
        if (
            pair_item_indices[pair] < 0 || pair_item_indices[pair] >= expert_count ||
            pair_row_indices[pair] < 0 || pair_row_indices[pair] >= batch
        ) {
            return -4;
        }
    }

    const float* lut = fp8_e4m3fn_lut();
    const int64_t activation_count = pair_count * intermediate_rows;
    float* activation = new (std::nothrow) float[static_cast<size_t>(activation_count)];
    if (activation == nullptr) {
        return -3;
    }
    float* pair_output = new (std::nothrow) float[static_cast<size_t>(pair_count * hidden_cols)];
    if (pair_output == nullptr) {
        delete[] activation;
        return -3;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t pair = 0; pair < pair_count; ++pair) {
        for (int64_t row = 0; row < intermediate_rows; ++row) {
            const int64_t expert = pair_item_indices[pair];
            const int64_t b = pair_row_indices[pair];
            const uint8_t* gate_weight = reinterpret_cast<const uint8_t*>(gate_weight_ptrs[expert]);
            const float* gate_scale_inv = reinterpret_cast<const float*>(gate_scale_ptrs[expert]);
            const uint8_t* up_weight = reinterpret_cast<const uint8_t*>(up_weight_ptrs[expert]);
            const float* up_scale_inv = reinterpret_cast<const float*>(up_scale_ptrs[expert]);
            if (
                gate_weight == nullptr || gate_scale_inv == nullptr ||
                up_weight == nullptr || up_scale_inv == nullptr
            ) {
                activation[pair * intermediate_rows + row] = 0.0f;
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
                for (int64_t col = start_col; col < end_col; ++col) {
                    const float h = hidden_row[col];
                    gate_block += h * lut[gate_row[col]];
                    up_block += h * lut[up_row[col]];
                }
                gate_acc += gate_block * gate_scale;
                up_acc += up_block * up_scale;
            }
            const float silu = gate_acc / (1.0f + std::exp(-gate_acc));
            activation[pair * intermediate_rows + row] = silu * up_acc;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t pair = 0; pair < pair_count; ++pair) {
        for (int64_t row = 0; row < hidden_cols; ++row) {
            const int64_t expert = pair_item_indices[pair];
            const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(down_weight_ptrs[expert]);
            const float* down_scale_inv = reinterpret_cast<const float*>(down_scale_ptrs[expert]);
            if (down_weight == nullptr || down_scale_inv == nullptr) {
                pair_output[pair * hidden_cols + row] = 0.0f;
                continue;
            }
            const int64_t scale_row = row / 128;
            const uint8_t* down_row = down_weight + row * intermediate_rows;
            const float* activation_row = activation + pair * intermediate_rows;
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
            pair_output[pair * hidden_cols + row] = acc;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t row = 0; row < hidden_cols; ++row) {
            float routed_acc = 0.0f;
            for (int64_t pair = 0; pair < pair_count; ++pair) {
                if (pair_row_indices[pair] == b) {
                    routed_acc += pair_weights[pair] * pair_output[pair * hidden_cols + row];
                }
            }
            out[b * hidden_cols + row] = routed_acc;
        }
    }

    delete[] pair_output;
    delete[] activation;
    return 0;
}
