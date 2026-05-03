#include <algorithm>
#include <cmath>
#include <cstdint>
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

static inline float silu(float value) {
    return value / (1.0f + std::exp(-value));
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
        std::vector<float> logits(static_cast<size_t>(num_experts), 0.0f);
        for (int64_t expert = 0; expert < num_experts; ++expert) {
            float acc = 0.0f;
            const uint16_t* row = router_weight + expert * hidden_size;
            for (int64_t dim = 0; dim < hidden_size; ++dim) {
                acc += fp16_to_fp32(token_hidden[dim]) * fp16_to_fp32(row[dim]);
            }
            logits[static_cast<size_t>(expert)] = acc;
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
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int64_t dim = 0; dim < hidden_size; ++dim) {
                    const float h = fp16_to_fp32(token_hidden[dim]);
                    gate_acc += h * fp16_to_fp32(gate_base[row * hidden_size + dim]);
                    up_acc += h * fp16_to_fp32(up_base[row * hidden_size + dim]);
                }
                expert_hidden[static_cast<size_t>(row)] = silu(gate_acc) * up_acc;
            }
            for (int64_t row = 0; row < hidden_size; ++row) {
                float down_acc = 0.0f;
                for (int64_t dim = 0; dim < intermediate_size; ++dim) {
                    down_acc += expert_hidden[static_cast<size_t>(dim)] * fp16_to_fp32(down_base[row * intermediate_size + dim]);
                }
                combined[static_cast<size_t>(row)] += route_weight * down_acc;
            }
        }

        for (int64_t dim = 0; dim < hidden_size; ++dim) {
            out[token * hidden_size + dim] = fp32_to_fp16(combined[static_cast<size_t>(dim)]);
        }
    }
    return 0;
}
