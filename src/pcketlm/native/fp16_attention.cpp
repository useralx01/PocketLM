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

static inline __m256 load_fp16_as_ps(const uint16_t* values) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
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

static void linear(
    const uint16_t* hidden,
    const uint16_t* weight,
    float* out,
    int64_t seq_len,
    int64_t in_features,
    int64_t out_features
) {
    #pragma omp parallel for schedule(static)
    for (int64_t token = 0; token < seq_len; ++token) {
        for (int64_t row = 0; row < out_features; ++row) {
            const uint16_t* hidden_row = hidden + token * in_features;
            const uint16_t* weight_row = weight + row * in_features;
            __m256 acc_vec = _mm256_setzero_ps();
            int64_t col = 0;
            for (; col + 8 <= in_features; col += 8) {
                const __m256 h = load_fp16_as_ps(hidden_row + col);
                const __m256 w = load_fp16_as_ps(weight_row + col);
                acc_vec = _mm256_fmadd_ps(h, w, acc_vec);
            }
            float acc = horizontal_sum_ps(acc_vec);
            for (; col < in_features; ++col) {
                acc += fp16_to_fp32(hidden_row[col]) * fp16_to_fp32(weight_row[col]);
            }
            out[token * out_features + row] = acc;
        }
    }
}

static void apply_rope(
    float* values,
    int64_t seq_len,
    int64_t head_count,
    int64_t head_dim,
    int64_t position_offset,
    float rope_theta
) {
    for (int64_t token = 0; token < seq_len; ++token) {
        const float position = static_cast<float>(position_offset + token);
        for (int64_t head = 0; head < head_count; ++head) {
            float* base = values + (token * head_count + head) * head_dim;
            const int64_t half_dim = head_dim / 2;
            for (int64_t dim = 0; dim < half_dim; ++dim) {
                const float inv_freq = std::pow(rope_theta, -static_cast<float>(dim) / static_cast<float>(head_dim));
                const float angle = position * inv_freq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const float x0 = base[dim];
                const float x1 = base[dim + half_dim];
                base[dim] = x0 * c - x1 * s;
                base[dim + half_dim] = x1 * c + x0 * s;
            }
        }
    }
}

extern "C" __declspec(dllexport) int native_attention_prefill_fp16(
    const uint16_t* hidden,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    const uint16_t* o_weight,
    uint16_t* out,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    int64_t position_offset,
    float rope_theta
) {
    if (
        hidden == nullptr || q_weight == nullptr || k_weight == nullptr ||
        v_weight == nullptr || o_weight == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (seq_len <= 0 || hidden_size <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0) {
        return 2;
    }
    if (hidden_size % num_attention_heads != 0 || num_attention_heads % num_key_value_heads != 0) {
        return 3;
    }

    const int64_t head_dim = hidden_size / num_attention_heads;
    const int64_t kv_hidden = num_key_value_heads * head_dim;
    const int64_t kv_repeat = num_attention_heads / num_key_value_heads;
    std::vector<float> q(static_cast<size_t>(seq_len * hidden_size), 0.0f);
    std::vector<float> k(static_cast<size_t>(seq_len * kv_hidden), 0.0f);
    std::vector<float> v(static_cast<size_t>(seq_len * kv_hidden), 0.0f);
    std::vector<float> context(static_cast<size_t>(seq_len * hidden_size), 0.0f);

    linear(hidden, q_weight, q.data(), seq_len, hidden_size, hidden_size);
    linear(hidden, k_weight, k.data(), seq_len, hidden_size, kv_hidden);
    linear(hidden, v_weight, v.data(), seq_len, hidden_size, kv_hidden);
    apply_rope(q.data(), seq_len, num_attention_heads, head_dim, position_offset, rope_theta);
    apply_rope(k.data(), seq_len, num_key_value_heads, head_dim, position_offset, rope_theta);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    #pragma omp parallel for schedule(static)
    for (int64_t token = 0; token < seq_len; ++token) {
        std::vector<float> scores(static_cast<size_t>(seq_len), 0.0f);
        for (int64_t head = 0; head < num_attention_heads; ++head) {
            const int64_t kv_head = head / kv_repeat;
            float max_score = -INFINITY;
            for (int64_t source = 0; source <= token; ++source) {
                float score = 0.0f;
                const float* q_base = q.data() + (token * num_attention_heads + head) * head_dim;
                const float* k_base = k.data() + (source * num_key_value_heads + kv_head) * head_dim;
                for (int64_t dim = 0; dim < head_dim; ++dim) {
                    score += q_base[dim] * k_base[dim];
                }
                scores[static_cast<size_t>(source)] = score * scale;
                max_score = std::max(max_score, scores[static_cast<size_t>(source)]);
            }

            float denom = 0.0f;
            for (int64_t source = 0; source <= token; ++source) {
                scores[static_cast<size_t>(source)] = std::exp(scores[static_cast<size_t>(source)] - max_score);
                denom += scores[static_cast<size_t>(source)];
            }

            float* out_head = context.data() + (token * num_attention_heads + head) * head_dim;
            for (int64_t source = 0; source <= token; ++source) {
                const float weight = scores[static_cast<size_t>(source)] / denom;
                const float* v_base = v.data() + (source * num_key_value_heads + kv_head) * head_dim;
                for (int64_t dim = 0; dim < head_dim; ++dim) {
                    out_head[dim] += weight * v_base[dim];
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t token = 0; token < seq_len; ++token) {
        for (int64_t row = 0; row < hidden_size; ++row) {
            float acc = 0.0f;
            for (int64_t col = 0; col < hidden_size; ++col) {
                acc += context[token * hidden_size + col] * fp16_to_fp32(o_weight[row * hidden_size + col]);
            }
            out[token * hidden_size + row] = fp32_to_fp16(acc);
        }
    }

    return 0;
}
