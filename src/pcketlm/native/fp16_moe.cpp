#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <immintrin.h>

static inline float fp16_to_fp32(uint16_t value) {
    const __m128i half = _mm_cvtsi32_si128(static_cast<int>(value));
    const __m128 full = _mm_cvtph_ps(half);
    return _mm_cvtss_f32(full);
}

static inline uint16_t fp32_to_fp16(float value) {
    const __m128 full = _mm_set_ss(value);
    const __m128i half = _mm_cvtps_ph(full, 0);
    return static_cast<uint16_t>(_mm_cvtsi128_si32(half));
}

static inline float bf16_to_fp32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

static inline uint16_t fp32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(uint32_t));
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

static inline float read_u16(uint16_t value, int dtype_code) {
    return dtype_code == 1 ? bf16_to_fp32(value) : fp16_to_fp32(value);
}

static inline __m256 load_u16_as_ps(const uint16_t* values, int dtype_code) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
    if (dtype_code == 1) {
        const __m256i widened = _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16);
        return _mm256_castsi256_ps(widened);
    }
    return _mm256_cvtph_ps(packed);
}

static inline float horizontal_sum_ps(__m256 values) {
    const __m128 low = _mm256_castps256_ps128(values);
    const __m128 high = _mm256_extractf128_ps(values, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline uint16_t write_u16(float value, int dtype_code) {
    return dtype_code == 1 ? fp32_to_bf16(value) : fp32_to_fp16(value);
}

static inline float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

static void load_vector_u16_as_float(const uint16_t* values, float* out, int64_t count, int dtype_code) {
    int64_t index = 0;
    for (; index + 8 <= count; index += 8) {
        _mm256_storeu_ps(out + index, load_u16_as_ps(values + index, dtype_code));
    }
    for (; index < count; ++index) {
        out[index] = read_u16(values[index], dtype_code);
    }
}

static float dot_float_u16(const float* left, const uint16_t* right, int64_t count, int dtype_code) {
    __m256 acc_vec = _mm256_setzero_ps();
    int64_t index = 0;
    for (; index + 8 <= count; index += 8) {
        const __m256 l = _mm256_loadu_ps(left + index);
        const __m256 r = load_u16_as_ps(right + index, dtype_code);
        acc_vec = _mm256_fmadd_ps(l, r, acc_vec);
    }
    float acc = horizontal_sum_ps(acc_vec);
    for (; index < count; ++index) {
        acc += left[index] * read_u16(right[index], dtype_code);
    }
    return acc;
}

extern "C" __declspec(dllexport) int native_moe_forward_fp16(
    const uint16_t* hidden,
    const uint16_t* router_weight,
    const uint16_t* gate_weight,
    const uint16_t* up_weight,
    const uint16_t* down_weight,
    uint16_t* out,
    int64_t* selected_experts,
    float* selected_weights,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t top_k,
    int64_t intermediate_size,
    int normalize_topk
) {
    if (
        hidden == nullptr || router_weight == nullptr || gate_weight == nullptr ||
        up_weight == nullptr || down_weight == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (seq_len <= 0 || hidden_size <= 0 || num_experts <= 0 || top_k <= 0 || intermediate_size <= 0) {
        return 2;
    }
    if (top_k > num_experts) {
        return 3;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t token = 0; token < seq_len; ++token) {
        const uint16_t* token_hidden = hidden + token * hidden_size;
        std::vector<float> token_hidden_f(static_cast<size_t>(hidden_size), 0.0f);
        load_vector_u16_as_float(token_hidden, token_hidden_f.data(), hidden_size, 0);
        std::vector<float> logits(static_cast<size_t>(num_experts), 0.0f);
        for (int64_t expert = 0; expert < num_experts; ++expert) {
            const uint16_t* row = router_weight + expert * hidden_size;
            logits[static_cast<size_t>(expert)] = dot_float_u16(token_hidden_f.data(), row, hidden_size, 0);
        }

        const float max_logit = *std::max_element(logits.begin(), logits.end());
        float denom = 0.0f;
        for (float& value : logits) {
            value = std::exp(value - max_logit);
            denom += value;
        }
        for (float& value : logits) {
            value /= denom;
        }

        std::vector<int64_t> order(static_cast<size_t>(num_experts), 0);
        for (int64_t expert = 0; expert < num_experts; ++expert) {
            order[static_cast<size_t>(expert)] = expert;
        }
        std::partial_sort(
            order.begin(),
            order.begin() + top_k,
            order.end(),
            [&](int64_t left, int64_t right) {
                const float l = logits[static_cast<size_t>(left)];
                const float r = logits[static_cast<size_t>(right)];
                if (l == r) {
                    return left < right;
                }
                return l > r;
            }
        );

        float topk_denom = 0.0f;
        for (int64_t rank = 0; rank < top_k; ++rank) {
            topk_denom += logits[static_cast<size_t>(order[static_cast<size_t>(rank)])];
        }

        std::vector<float> combined(static_cast<size_t>(hidden_size), 0.0f);
        std::vector<float> expert_hidden(static_cast<size_t>(intermediate_size), 0.0f);
        for (int64_t rank = 0; rank < top_k; ++rank) {
            const int64_t expert = order[static_cast<size_t>(rank)];
            float route_weight = logits[static_cast<size_t>(expert)];
            if (normalize_topk && topk_denom > 0.0f) {
                route_weight /= topk_denom;
            }
            if (selected_experts != nullptr) {
                selected_experts[token * top_k + rank] = expert;
            }
            if (selected_weights != nullptr) {
                selected_weights[token * top_k + rank] = route_weight;
            }

            const uint16_t* gate_base = gate_weight + (expert * intermediate_size * hidden_size);
            const uint16_t* up_base = up_weight + (expert * intermediate_size * hidden_size);
            const uint16_t* down_base = down_weight + (expert * hidden_size * intermediate_size);
            for (int64_t row = 0; row < intermediate_size; ++row) {
                const float gate_acc = dot_float_u16(token_hidden_f.data(), gate_base + row * hidden_size, hidden_size, 0);
                const float up_acc = dot_float_u16(token_hidden_f.data(), up_base + row * hidden_size, hidden_size, 0);
                expert_hidden[static_cast<size_t>(row)] = silu(gate_acc) * up_acc;
            }
            for (int64_t row = 0; row < hidden_size; ++row) {
                const float down_acc = dot_float_u16(expert_hidden.data(), down_base + row * intermediate_size, intermediate_size, 0);
                combined[static_cast<size_t>(row)] += route_weight * down_acc;
            }
        }

        for (int64_t dim = 0; dim < hidden_size; ++dim) {
            out[token * hidden_size + dim] = fp32_to_fp16(combined[static_cast<size_t>(dim)]);
        }
    }
    return 0;
}

extern "C" __declspec(dllexport) int native_moe_selected_forward_u16(
    const uint16_t* hidden,
    const uint16_t* gate_weight,
    const uint16_t* up_weight,
    const uint16_t* down_weight,
    const float* route_weights,
    uint16_t* out,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t selected_count,
    int64_t intermediate_size,
    int dtype_code
) {
    if (
        hidden == nullptr || gate_weight == nullptr || up_weight == nullptr ||
        down_weight == nullptr || route_weights == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (seq_len <= 0 || hidden_size <= 0 || selected_count <= 0 || intermediate_size <= 0) {
        return 2;
    }
    if (dtype_code != 0 && dtype_code != 1) {
        return 3;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t token = 0; token < seq_len; ++token) {
        const uint16_t* token_hidden = hidden + token * hidden_size;
        std::vector<float> token_hidden_f(static_cast<size_t>(hidden_size), 0.0f);
        load_vector_u16_as_float(token_hidden, token_hidden_f.data(), hidden_size, dtype_code);
        std::vector<float> combined(static_cast<size_t>(hidden_size), 0.0f);
        std::vector<float> expert_hidden(static_cast<size_t>(intermediate_size), 0.0f);
        for (int64_t rank = 0; rank < selected_count; ++rank) {
            const float route_weight = route_weights[token * selected_count + rank];
            const uint16_t* gate_base = gate_weight + (rank * intermediate_size * hidden_size);
            const uint16_t* up_base = up_weight + (rank * intermediate_size * hidden_size);
            const uint16_t* down_base = down_weight + (rank * hidden_size * intermediate_size);
            for (int64_t row = 0; row < intermediate_size; ++row) {
                const float gate_acc = dot_float_u16(
                    token_hidden_f.data(),
                    gate_base + row * hidden_size,
                    hidden_size,
                    dtype_code
                );
                const float up_acc = dot_float_u16(
                    token_hidden_f.data(),
                    up_base + row * hidden_size,
                    hidden_size,
                    dtype_code
                );
                expert_hidden[static_cast<size_t>(row)] = silu(gate_acc) * up_acc;
            }
            for (int64_t row = 0; row < hidden_size; ++row) {
                const float down_acc = dot_float_u16(
                    expert_hidden.data(),
                    down_base + row * intermediate_size,
                    intermediate_size,
                    dtype_code
                );
                combined[static_cast<size_t>(row)] += route_weight * down_acc;
            }
        }
        for (int64_t dim = 0; dim < hidden_size; ++dim) {
            out[token * hidden_size + dim] = write_u16(combined[static_cast<size_t>(dim)], dtype_code);
        }
    }
    return 0;
}
