#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

struct LayerKvState {
    std::vector<uint16_t> committed_k;
    std::vector<uint16_t> committed_v;
    std::vector<uint16_t> tentative_k;
    std::vector<uint16_t> tentative_v;
    int64_t committed_len = 0;
    int64_t tentative_len = 0;
};

struct KvSession {
    int64_t layer_count = 0;
    int64_t max_seq_len = 0;
    int64_t kv_width = 0;
    int dtype_code = 0; // 0 = fp16, 1 = bf16
    std::vector<LayerKvState> layers;
    std::vector<float> scratch_input_norm;
    std::vector<float> scratch_residual;
    std::vector<float> scratch_post_norm;
    std::vector<float> scratch_gate;
    std::vector<float> scratch_up;
    std::vector<float> scratch_activated;
    std::vector<float> scratch_mlp_out;
    std::vector<uint16_t> scratch_input_norm_u16;
    std::vector<uint16_t> scratch_attention_out_u16;
    std::vector<uint16_t> scratch_residual_u16;
    std::vector<uint16_t> scratch_post_norm_u16;
    std::vector<float> scratch_q;
    std::vector<float> scratch_k;
    std::vector<float> scratch_v;
    std::vector<float> scratch_context;
    std::vector<float> scratch_scores;
    std::vector<float> scratch_projected;
    std::vector<uint16_t> scratch_k_u16;
    std::vector<uint16_t> scratch_v_u16;
};

static thread_local bool g_use_prepacked_rows8 = false;

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

static inline uint16_t write_u16(float value, int dtype_code) {
    return dtype_code == 1 ? fp32_to_bf16(value) : fp32_to_fp16(value);
}

static void configure_openmp_threads() {
    #ifdef _OPENMP
    const char* requested_threads = std::getenv("PCKETLM_NATIVE_THREADS");
    if (requested_threads != nullptr && requested_threads[0] != '\0') {
        const int parsed = std::atoi(requested_threads);
        if (parsed > 0) {
            omp_set_num_threads(parsed);
        }
    }
    #endif
}


static inline __m256 load_u16_as_ps(const uint16_t* values, int dtype_code) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
    if (dtype_code == 1) {
        __m256i widened = _mm256_cvtepu16_epi32(packed);
        widened = _mm256_slli_epi32(widened, 16);
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

static bool packed_rows8_gemv_u16_hidden(
    const uint16_t* hidden,
    const uint16_t* packed,
    const uint16_t* bias,
    float* out,
    int64_t in_features,
    int64_t out_features,
    int dtype_code
) {
    constexpr int64_t row_block = 8;
    const int64_t block_count = (out_features + row_block - 1) / row_block;
    #pragma omp parallel for schedule(static)
    for (int64_t block = 0; block < block_count; ++block) {
        const int64_t row_base = block * row_block;
        const uint16_t* weight_block = packed + block * in_features * row_block;
        __m256 acc = _mm256_setzero_ps();
        for (int64_t col = 0; col < in_features; ++col) {
            const __m256 w = load_u16_as_ps(weight_block + col * row_block, dtype_code);
            const __m256 h = _mm256_set1_ps(read_u16(hidden[col], dtype_code));
            acc = _mm256_fmadd_ps(h, w, acc);
        }
        float sums[row_block];
        _mm256_storeu_ps(sums, acc);
        for (int64_t lane = 0; lane < row_block; ++lane) {
            const int64_t row = row_base + lane;
            if (row < out_features) {
                out[row] = sums[lane] + (bias != nullptr ? read_u16(bias[row], dtype_code) : 0.0f);
            }
        }
    }
    return true;
}

static bool packed_rows8_gemv_float_hidden(
    const float* hidden,
    const uint16_t* packed,
    float* out,
    int64_t in_features,
    int64_t out_features,
    int dtype_code
) {
    constexpr int64_t row_block = 8;
    const int64_t block_count = (out_features + row_block - 1) / row_block;
    #pragma omp parallel for schedule(static)
    for (int64_t block = 0; block < block_count; ++block) {
        const int64_t row_base = block * row_block;
        const uint16_t* weight_block = packed + block * in_features * row_block;
        __m256 acc = _mm256_setzero_ps();
        for (int64_t col = 0; col < in_features; ++col) {
            const __m256 w = load_u16_as_ps(weight_block + col * row_block, dtype_code);
            const __m256 h = _mm256_set1_ps(hidden[col]);
            acc = _mm256_fmadd_ps(h, w, acc);
        }
        float sums[row_block];
        _mm256_storeu_ps(sums, acc);
        for (int64_t lane = 0; lane < row_block; ++lane) {
            const int64_t row = row_base + lane;
            if (row < out_features) {
                out[row] = sums[lane];
            }
        }
    }
    return true;
}

static void packed_rows8_gemv_u16_hidden_block(
    const uint16_t* hidden,
    const uint16_t* packed,
    const uint16_t* bias,
    float* out,
    int64_t in_features,
    int64_t out_features,
    int64_t block,
    int dtype_code
) {
    constexpr int64_t row_block = 8;
    const int64_t row_base = block * row_block;
    const uint16_t* weight_block = packed + block * in_features * row_block;
    __m256 acc = _mm256_setzero_ps();
    for (int64_t col = 0; col < in_features; ++col) {
        const __m256 w = load_u16_as_ps(weight_block + col * row_block, dtype_code);
        const __m256 h = _mm256_set1_ps(read_u16(hidden[col], dtype_code));
        acc = _mm256_fmadd_ps(h, w, acc);
    }
    float sums[row_block];
    _mm256_storeu_ps(sums, acc);
    for (int64_t lane = 0; lane < row_block; ++lane) {
        const int64_t row = row_base + lane;
        if (row < out_features) {
            out[row] = sums[lane] + (bias != nullptr ? read_u16(bias[row], dtype_code) : 0.0f);
        }
    }
}

static void packed_rows8_gemv_float_hidden_block(
    const float* hidden,
    const uint16_t* packed,
    float* out,
    int64_t in_features,
    int64_t out_features,
    int64_t block,
    int dtype_code
) {
    constexpr int64_t row_block = 8;
    const int64_t row_base = block * row_block;
    const uint16_t* weight_block = packed + block * in_features * row_block;
    __m256 acc = _mm256_setzero_ps();
    for (int64_t col = 0; col < in_features; ++col) {
        const __m256 w = load_u16_as_ps(weight_block + col * row_block, dtype_code);
        const __m256 h = _mm256_set1_ps(hidden[col]);
        acc = _mm256_fmadd_ps(h, w, acc);
    }
    float sums[row_block];
    _mm256_storeu_ps(sums, acc);
    for (int64_t lane = 0; lane < row_block; ++lane) {
        const int64_t row = row_base + lane;
        if (row < out_features) {
            out[row] = sums[lane];
        }
    }
}

static bool packed_rows8_gemv_two_u16_hidden(
    const uint16_t* hidden,
    const uint16_t* weight_a,
    const uint16_t* weight_b,
    float* out_a,
    float* out_b,
    int64_t in_features,
    int64_t out_features,
    int dtype_code
) {
    constexpr int64_t row_block = 8;
    const int64_t block_count = (out_features + row_block - 1) / row_block;
    #pragma omp parallel for schedule(static)
    for (int64_t block = 0; block < block_count * 2; ++block) {
        if (block < block_count) {
            packed_rows8_gemv_u16_hidden_block(
                hidden, weight_a, nullptr, out_a, in_features, out_features, block, dtype_code);
        } else {
            packed_rows8_gemv_u16_hidden_block(
                hidden, weight_b, nullptr, out_b, in_features, out_features, block - block_count, dtype_code);
        }
    }
    return true;
}

static bool packed_rows8_gemv_three_u16_hidden(
    const uint16_t* hidden,
    const uint16_t* weight_q,
    const uint16_t* weight_k,
    const uint16_t* weight_v,
    const uint16_t* bias_q,
    const uint16_t* bias_k,
    const uint16_t* bias_v,
    float* out_q,
    float* out_k,
    float* out_v,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features,
    int dtype_code
) {
    constexpr int64_t row_block = 8;
    const int64_t q_blocks = (q_features + row_block - 1) / row_block;
    const int64_t kv_blocks = (kv_features + row_block - 1) / row_block;
    const int64_t total_blocks = q_blocks + 2 * kv_blocks;
    #pragma omp parallel for schedule(static)
    for (int64_t block = 0; block < total_blocks; ++block) {
        if (block < q_blocks) {
            packed_rows8_gemv_u16_hidden_block(
                hidden, weight_q, bias_q, out_q, in_features, q_features, block, dtype_code);
        } else if (block < q_blocks + kv_blocks) {
            packed_rows8_gemv_u16_hidden_block(
                hidden, weight_k, bias_k, out_k, in_features, kv_features, block - q_blocks, dtype_code);
        } else {
            packed_rows8_gemv_u16_hidden_block(
                hidden, weight_v, bias_v, out_v, in_features, kv_features, block - q_blocks - kv_blocks, dtype_code);
        }
    }
    return true;
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

static void load_matrix_u16_as_float(const uint16_t* values, float* out, int64_t count, int dtype_code) {
    load_vector_u16_as_float(values, out, count, dtype_code);
}

static bool linear_one_blas(
    const uint16_t* hidden,
    const uint16_t* weight,
    const uint16_t* bias,
    float* out,
    int64_t in_features,
    int64_t out_features,
    int dtype_code
) {
    if (g_use_prepacked_rows8) {
        return packed_rows8_gemv_u16_hidden(hidden, weight, bias, out, in_features, out_features, dtype_code);
    }
    (void)hidden;
    (void)weight;
    (void)bias;
    (void)out;
    (void)in_features;
    (void)out_features;
    (void)dtype_code;
    return false;
}

static bool linear_float_u16_blas(
    const float* hidden,
    const uint16_t* weight,
    float* out,
    int64_t in_features,
    int64_t out_features,
    int dtype_code
) {
    if (g_use_prepacked_rows8) {
        return packed_rows8_gemv_float_hidden(hidden, weight, out, in_features, out_features, dtype_code);
    }
    (void)hidden;
    (void)weight;
    (void)out;
    (void)in_features;
    (void)out_features;
    (void)dtype_code;
    return false;
}

static inline float dot_float_u16(const float* left, const uint16_t* right, int64_t count, int dtype_code) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int64_t index = 0;
    for (; index + 32 <= count; index += 32) {
        const __m256 l0 = _mm256_loadu_ps(left + index);
        const __m256 l1 = _mm256_loadu_ps(left + index + 8);
        const __m256 l2 = _mm256_loadu_ps(left + index + 16);
        const __m256 l3 = _mm256_loadu_ps(left + index + 24);
        acc0 = _mm256_fmadd_ps(l0, load_u16_as_ps(right + index, dtype_code), acc0);
        acc1 = _mm256_fmadd_ps(l1, load_u16_as_ps(right + index + 8, dtype_code), acc1);
        acc2 = _mm256_fmadd_ps(l2, load_u16_as_ps(right + index + 16, dtype_code), acc2);
        acc3 = _mm256_fmadd_ps(l3, load_u16_as_ps(right + index + 24, dtype_code), acc3);
    }
    for (; index + 8 <= count; index += 8) {
        const __m256 l = _mm256_loadu_ps(left + index);
        const __m256 r = load_u16_as_ps(right + index, dtype_code);
        acc0 = _mm256_fmadd_ps(l, r, acc0);
    }
    const __m256 acc_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float acc = horizontal_sum_ps(acc_vec);
    for (; index < count; ++index) {
        acc += left[index] * read_u16(right[index], dtype_code);
    }
    return acc;
}

static inline bool valid_layer(KvSession* session, int64_t layer) {
    return session != nullptr && layer >= 0 && layer < session->layer_count;
}

static void linear_one(
    const uint16_t* hidden,
    const uint16_t* weight,
    const uint16_t* bias,
    float* out,
    int64_t in_features,
    int64_t out_features,
    int dtype_code
) {
    if (linear_one_blas(hidden, weight, bias, out, in_features, out_features, dtype_code)) {
        return;
    }
    std::vector<float> hidden_f(static_cast<size_t>(in_features), 0.0f);
    load_vector_u16_as_float(hidden, hidden_f.data(), in_features, dtype_code);

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < out_features; ++row) {
        const uint16_t* weight_row = weight + row * in_features;
        float acc = dot_float_u16(hidden_f.data(), weight_row, in_features, dtype_code);
        if (bias != nullptr) {
            acc += read_u16(bias[row], dtype_code);
        }
        out[row] = acc;
    }
}

static void linear_two_same_input(
    const uint16_t* hidden,
    const uint16_t* weight_a,
    const uint16_t* weight_b,
    float* out_a,
    float* out_b,
    int64_t in_features,
    int64_t out_features,
    int dtype_code
) {
    if (g_use_prepacked_rows8) {
        packed_rows8_gemv_two_u16_hidden(
            hidden, weight_a, weight_b, out_a, out_b, in_features, out_features, dtype_code);
        return;
    }
    if (
        linear_one_blas(hidden, weight_a, nullptr, out_a, in_features, out_features, dtype_code) &&
        linear_one_blas(hidden, weight_b, nullptr, out_b, in_features, out_features, dtype_code)
    ) {
        return;
    }
    std::vector<float> hidden_f(static_cast<size_t>(in_features), 0.0f);
    load_vector_u16_as_float(hidden, hidden_f.data(), in_features, dtype_code);

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < out_features; ++row) {
        out_a[row] = dot_float_u16(hidden_f.data(), weight_a + row * in_features, in_features, dtype_code);
        out_b[row] = dot_float_u16(hidden_f.data(), weight_b + row * in_features, in_features, dtype_code);
    }
}

static void linear_three_same_input(
    const uint16_t* hidden,
    const uint16_t* weight_q,
    const uint16_t* weight_k,
    const uint16_t* weight_v,
    const uint16_t* bias_q,
    const uint16_t* bias_k,
    const uint16_t* bias_v,
    float* out_q,
    float* out_k,
    float* out_v,
    int64_t in_features,
    int64_t q_features,
    int64_t kv_features,
    int dtype_code
) {
    if (g_use_prepacked_rows8) {
        packed_rows8_gemv_three_u16_hidden(
            hidden,
            weight_q,
            weight_k,
            weight_v,
            bias_q,
            bias_k,
            bias_v,
            out_q,
            out_k,
            out_v,
            in_features,
            q_features,
            kv_features,
            dtype_code
        );
        return;
    }
    if (
        linear_one_blas(hidden, weight_q, bias_q, out_q, in_features, q_features, dtype_code) &&
        linear_one_blas(hidden, weight_k, bias_k, out_k, in_features, kv_features, dtype_code) &&
        linear_one_blas(hidden, weight_v, bias_v, out_v, in_features, kv_features, dtype_code)
    ) {
        return;
    }
    std::vector<float> hidden_f(static_cast<size_t>(in_features), 0.0f);
    load_vector_u16_as_float(hidden, hidden_f.data(), in_features, dtype_code);

    const int64_t total_rows = q_features + 2 * kv_features;
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < total_rows; ++row) {
        if (row < q_features) {
            float acc = dot_float_u16(hidden_f.data(), weight_q + row * in_features, in_features, dtype_code);
            if (bias_q != nullptr) {
                acc += read_u16(bias_q[row], dtype_code);
            }
            out_q[row] = acc;
        } else if (row < q_features + kv_features) {
            const int64_t k_row = row - q_features;
            float acc = dot_float_u16(hidden_f.data(), weight_k + k_row * in_features, in_features, dtype_code);
            if (bias_k != nullptr) {
                acc += read_u16(bias_k[k_row], dtype_code);
            }
            out_k[k_row] = acc;
        } else {
            const int64_t v_row = row - q_features - kv_features;
            float acc = dot_float_u16(hidden_f.data(), weight_v + v_row * in_features, in_features, dtype_code);
            if (bias_v != nullptr) {
                acc += read_u16(bias_v[v_row], dtype_code);
            }
            out_v[v_row] = acc;
        }
    }
}

static void apply_rope_one(
    float* values,
    int64_t head_count,
    int64_t head_dim,
    int64_t position,
    float rope_theta
) {
    const int64_t half_dim = head_dim / 2;
    for (int64_t head = 0; head < head_count; ++head) {
        float* base = values + head * head_dim;
        for (int64_t dim = 0; dim < half_dim; ++dim) {
            const float inv_freq = std::pow(rope_theta, -(2.0f * static_cast<float>(dim)) / static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * inv_freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float x0 = base[dim];
            const float x1 = base[dim + half_dim];
            base[dim] = x0 * c - x1 * s;
            base[dim + half_dim] = x1 * c + x0 * s;
        }
    }
}

static void rms_norm_one(
    const uint16_t* hidden,
    const uint16_t* weight,
    float* out,
    int64_t hidden_size,
    float eps,
    int dtype_code
) {
    float mean_square = 0.0f;
    for (int64_t dim = 0; dim < hidden_size; ++dim) {
        const float value = read_u16(hidden[dim], dtype_code);
        mean_square += value * value;
    }
    mean_square /= static_cast<float>(hidden_size);
    const float scale = 1.0f / std::sqrt(mean_square + eps);
    for (int64_t dim = 0; dim < hidden_size; ++dim) {
        out[dim] = read_u16(hidden[dim], dtype_code) * scale * read_u16(weight[dim], dtype_code);
    }
}

static void rms_norm_heads_inplace(
    float* values,
    const uint16_t* weight,
    int64_t head_count,
    int64_t head_dim,
    float eps,
    int dtype_code
) {
    if (weight == nullptr) {
        return;
    }
    for (int64_t head = 0; head < head_count; ++head) {
        float* base = values + head * head_dim;
        float mean_square = 0.0f;
        for (int64_t dim = 0; dim < head_dim; ++dim) {
            mean_square += base[dim] * base[dim];
        }
        mean_square /= static_cast<float>(head_dim);
        const float scale = 1.0f / std::sqrt(mean_square + eps);
        for (int64_t dim = 0; dim < head_dim; ++dim) {
            base[dim] = base[dim] * scale * read_u16(weight[dim], dtype_code);
        }
    }
}

static uint16_t* floats_to_u16_buffer(const std::vector<float>& values, std::vector<uint16_t>& storage, int dtype_code) {
    storage.resize(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        storage[index] = write_u16(values[index], dtype_code);
    }
    return storage.data();
}

extern "C" __declspec(dllexport) void* kv_prefill_init_typed(
    int64_t layer_count,
    int64_t max_seq_len,
    int64_t kv_width,
    int dtype_code
) {
    if (layer_count <= 0 || max_seq_len <= 0 || kv_width <= 0 || (dtype_code != 0 && dtype_code != 1)) {
        return nullptr;
    }
    configure_openmp_threads();
    KvSession* session = new KvSession();
    session->layer_count = layer_count;
    session->max_seq_len = max_seq_len;
    session->kv_width = kv_width;
    session->dtype_code = dtype_code;
    session->layers.resize(static_cast<size_t>(layer_count));
    return session;
}

extern "C" __declspec(dllexport) void* kv_prefill_init(
    int64_t layer_count,
    int64_t max_seq_len,
    int64_t kv_width
) {
    return kv_prefill_init_typed(layer_count, max_seq_len, kv_width, 0);
}

extern "C" __declspec(dllexport) void kv_free(void* handle) {
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    delete session;
}

static int append_to_region(
    KvSession* session,
    int64_t layer,
    const uint16_t* k_new,
    const uint16_t* v_new,
    int64_t count,
    bool tentative
) {
    if (!valid_layer(session, layer) || k_new == nullptr || v_new == nullptr || count < 0) {
        return 1;
    }
    LayerKvState& state = session->layers[static_cast<size_t>(layer)];
    const int64_t total_len = state.committed_len + state.tentative_len + count;
    if (total_len > session->max_seq_len) {
        return 2;
    }
    const size_t value_count = static_cast<size_t>(count * session->kv_width);
    if (tentative) {
        const size_t old = state.tentative_k.size();
        state.tentative_k.resize(old + value_count);
        state.tentative_v.resize(old + value_count);
        std::memcpy(state.tentative_k.data() + old, k_new, value_count * sizeof(uint16_t));
        std::memcpy(state.tentative_v.data() + old, v_new, value_count * sizeof(uint16_t));
        state.tentative_len += count;
    } else {
        const size_t old = state.committed_k.size();
        state.committed_k.resize(old + value_count);
        state.committed_v.resize(old + value_count);
        std::memcpy(state.committed_k.data() + old, k_new, value_count * sizeof(uint16_t));
        std::memcpy(state.committed_v.data() + old, v_new, value_count * sizeof(uint16_t));
        state.committed_len += count;
    }
    return 0;
}

extern "C" __declspec(dllexport) int kv_append_committed(
    void* handle,
    int64_t layer,
    const uint16_t* k_new,
    const uint16_t* v_new,
    int64_t count
) {
    return append_to_region(reinterpret_cast<KvSession*>(handle), layer, k_new, v_new, count, false);
}

extern "C" __declspec(dllexport) int kv_append_tentative(
    void* handle,
    int64_t layer,
    const uint16_t* k_new,
    const uint16_t* v_new,
    int64_t count
) {
    return append_to_region(reinterpret_cast<KvSession*>(handle), layer, k_new, v_new, count, true);
}

extern "C" __declspec(dllexport) int kv_commit(void* handle, int64_t count) {
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (session == nullptr || count < 0) {
        return 1;
    }
    for (LayerKvState& state : session->layers) {
        const int64_t take = std::min(count, state.tentative_len);
        const size_t value_count = static_cast<size_t>(take * session->kv_width);
        const size_t old = state.committed_k.size();
        state.committed_k.resize(old + value_count);
        state.committed_v.resize(old + value_count);
        if (value_count) {
            std::memcpy(state.committed_k.data() + old, state.tentative_k.data(), value_count * sizeof(uint16_t));
            std::memcpy(state.committed_v.data() + old, state.tentative_v.data(), value_count * sizeof(uint16_t));
        }
        state.committed_len += take;
        const size_t remaining_values = static_cast<size_t>((state.tentative_len - take) * session->kv_width);
        if (remaining_values) {
            std::memmove(
                state.tentative_k.data(),
                state.tentative_k.data() + value_count,
                remaining_values * sizeof(uint16_t)
            );
            std::memmove(
                state.tentative_v.data(),
                state.tentative_v.data() + value_count,
                remaining_values * sizeof(uint16_t)
            );
        }
        state.tentative_k.resize(remaining_values);
        state.tentative_v.resize(remaining_values);
        state.tentative_len -= take;
    }
    return 0;
}

extern "C" __declspec(dllexport) int kv_rollback(void* handle) {
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (session == nullptr) {
        return 1;
    }
    for (LayerKvState& state : session->layers) {
        state.tentative_k.clear();
        state.tentative_v.clear();
        state.tentative_len = 0;
    }
    return 0;
}

extern "C" __declspec(dllexport) int64_t kv_committed_length(void* handle, int64_t layer) {
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (!valid_layer(session, layer)) {
        return -1;
    }
    return session->layers[static_cast<size_t>(layer)].committed_len;
}

extern "C" __declspec(dllexport) int64_t kv_tentative_length(void* handle, int64_t layer) {
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (!valid_layer(session, layer)) {
        return -1;
    }
    return session->layers[static_cast<size_t>(layer)].tentative_len;
}

extern "C" __declspec(dllexport) int kv_copy_layer(
    void* handle,
    int64_t layer,
    uint16_t* k_out,
    uint16_t* v_out,
    int64_t max_count,
    int include_tentative
) {
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (!valid_layer(session, layer) || k_out == nullptr || v_out == nullptr || max_count < 0) {
        return 1;
    }
    LayerKvState& state = session->layers[static_cast<size_t>(layer)];
    const int64_t requested = state.committed_len + (include_tentative ? state.tentative_len : 0);
    if (requested > max_count) {
        return 2;
    }
    size_t offset = 0;
    const size_t committed_values = static_cast<size_t>(state.committed_len * session->kv_width);
    if (committed_values) {
        std::memcpy(k_out, state.committed_k.data(), committed_values * sizeof(uint16_t));
        std::memcpy(v_out, state.committed_v.data(), committed_values * sizeof(uint16_t));
        offset += committed_values;
    }
    if (include_tentative) {
        const size_t tentative_values = static_cast<size_t>(state.tentative_len * session->kv_width);
        if (tentative_values) {
            std::memcpy(k_out + offset, state.tentative_k.data(), tentative_values * sizeof(uint16_t));
            std::memcpy(v_out + offset, state.tentative_v.data(), tentative_values * sizeof(uint16_t));
        }
    }
    return 0;
}

extern "C" __declspec(dllexport) int kv_attention_decode_fp16(
    void* handle,
    int64_t layer,
    const uint16_t* hidden,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    const uint16_t* o_weight,
    uint16_t* out,
    int64_t hidden_size,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    float rope_theta
) {
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (
        !valid_layer(session, layer) || hidden == nullptr || q_weight == nullptr || k_weight == nullptr ||
        v_weight == nullptr || o_weight == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (hidden_size <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0) {
        return 2;
    }
    if (hidden_size % num_attention_heads != 0 || num_attention_heads % num_key_value_heads != 0) {
        return 3;
    }
    const int64_t head_dim = hidden_size / num_attention_heads;
    const int64_t kv_width = num_key_value_heads * head_dim;
    if (kv_width != session->kv_width) {
        return 4;
    }

    LayerKvState& state = session->layers[static_cast<size_t>(layer)];
    const int64_t position = state.committed_len + state.tentative_len;
    if (position + 1 > session->max_seq_len) {
        return 5;
    }
    const int64_t kv_repeat = num_attention_heads / num_key_value_heads;
    std::vector<float>& q = session->scratch_q;
    std::vector<float>& k = session->scratch_k;
    std::vector<float>& v = session->scratch_v;
    q.resize(static_cast<size_t>(hidden_size));
    k.resize(static_cast<size_t>(kv_width));
    v.resize(static_cast<size_t>(kv_width));
    linear_three_same_input(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        nullptr,
        nullptr,
        nullptr,
        q.data(),
        k.data(),
        v.data(),
        hidden_size,
        hidden_size,
        kv_width,
        session->dtype_code
    );
    apply_rope_one(q.data(), num_attention_heads, head_dim, position, rope_theta);
    apply_rope_one(k.data(), num_key_value_heads, head_dim, position, rope_theta);

    std::vector<uint16_t>& k_half = session->scratch_k_u16;
    std::vector<uint16_t>& v_half = session->scratch_v_u16;
    k_half.resize(static_cast<size_t>(kv_width));
    v_half.resize(static_cast<size_t>(kv_width));
    for (int64_t index = 0; index < kv_width; ++index) {
        k_half[static_cast<size_t>(index)] = write_u16(k[static_cast<size_t>(index)], session->dtype_code);
        v_half[static_cast<size_t>(index)] = write_u16(v[static_cast<size_t>(index)], session->dtype_code);
    }
    int append_code = append_to_region(session, layer, k_half.data(), v_half.data(), 1, true);
    if (append_code != 0) {
        return 10 + append_code;
    }

    const int64_t total_len = state.committed_len + state.tentative_len;
    std::vector<float>& context = session->scratch_context;
    std::vector<float>& scores = session->scratch_scores;
    context.assign(static_cast<size_t>(hidden_size), 0.0f);
    scores.resize(static_cast<size_t>(total_len));
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int64_t head = 0; head < num_attention_heads; ++head) {
        const int64_t kv_head = head / kv_repeat;
        float max_score = -INFINITY;
        const float* q_base = q.data() + head * head_dim;
        for (int64_t token = 0; token < total_len; ++token) {
            const uint16_t* k_base = nullptr;
            if (token < state.committed_len) {
                k_base = state.committed_k.data() + (token * kv_width + kv_head * head_dim);
            } else {
                k_base = state.tentative_k.data() + ((token - state.committed_len) * kv_width + kv_head * head_dim);
            }
            float score = 0.0f;
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                score += q_base[dim] * read_u16(k_base[dim], session->dtype_code);
            }
            scores[static_cast<size_t>(token)] = score * scale;
            max_score = std::max(max_score, scores[static_cast<size_t>(token)]);
        }
        float denom = 0.0f;
        for (int64_t token = 0; token < total_len; ++token) {
            scores[static_cast<size_t>(token)] = std::exp(scores[static_cast<size_t>(token)] - max_score);
            denom += scores[static_cast<size_t>(token)];
        }
        float* out_head = context.data() + head * head_dim;
        for (int64_t token = 0; token < total_len; ++token) {
            const float weight = scores[static_cast<size_t>(token)] / denom;
            const uint16_t* v_base = nullptr;
            if (token < state.committed_len) {
                v_base = state.committed_v.data() + (token * kv_width + kv_head * head_dim);
            } else {
                v_base = state.tentative_v.data() + ((token - state.committed_len) * kv_width + kv_head * head_dim);
            }
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                out_head[dim] += weight * read_u16(v_base[dim], session->dtype_code);
            }
        }
    }

    std::vector<float>& projected = session->scratch_projected;
    projected.resize(static_cast<size_t>(hidden_size));
    if (!linear_float_u16_blas(context.data(), o_weight, projected.data(), hidden_size, hidden_size, session->dtype_code)) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < hidden_size; ++row) {
            projected[static_cast<size_t>(row)] = dot_float_u16(
                context.data(),
                o_weight + row * hidden_size,
                hidden_size,
                session->dtype_code
            );
        }
    }
    for (int64_t row = 0; row < hidden_size; ++row) {
        out[row] = write_u16(projected[static_cast<size_t>(row)], session->dtype_code);
    }
    return 0;
}

extern "C" __declspec(dllexport) int kv_attention_decode_u16_ext(
    void* handle,
    int64_t layer,
    const uint16_t* hidden,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    const uint16_t* o_weight,
    const uint16_t* q_bias,
    const uint16_t* k_bias,
    const uint16_t* v_bias,
    const uint16_t* q_norm_weight,
    const uint16_t* k_norm_weight,
    uint16_t* out,
    int64_t hidden_size,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    float rope_theta,
    float rms_eps
) {
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (
        !valid_layer(session, layer) || hidden == nullptr || q_weight == nullptr || k_weight == nullptr ||
        v_weight == nullptr || o_weight == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (hidden_size <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0) {
        return 2;
    }
    if (hidden_size % num_attention_heads != 0 || num_attention_heads % num_key_value_heads != 0) {
        return 3;
    }
    const int64_t head_dim = hidden_size / num_attention_heads;
    const int64_t kv_width = num_key_value_heads * head_dim;
    if (kv_width != session->kv_width) {
        return 4;
    }

    LayerKvState& state = session->layers[static_cast<size_t>(layer)];
    const int64_t position = state.committed_len + state.tentative_len;
    if (position + 1 > session->max_seq_len) {
        return 5;
    }
    const int64_t kv_repeat = num_attention_heads / num_key_value_heads;
    std::vector<float>& q = session->scratch_q;
    std::vector<float>& k = session->scratch_k;
    std::vector<float>& v = session->scratch_v;
    q.resize(static_cast<size_t>(hidden_size));
    k.resize(static_cast<size_t>(kv_width));
    v.resize(static_cast<size_t>(kv_width));
    linear_three_same_input(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        q.data(),
        k.data(),
        v.data(),
        hidden_size,
        hidden_size,
        kv_width,
        session->dtype_code
    );
    rms_norm_heads_inplace(q.data(), q_norm_weight, num_attention_heads, head_dim, rms_eps, session->dtype_code);
    rms_norm_heads_inplace(k.data(), k_norm_weight, num_key_value_heads, head_dim, rms_eps, session->dtype_code);
    apply_rope_one(q.data(), num_attention_heads, head_dim, position, rope_theta);
    apply_rope_one(k.data(), num_key_value_heads, head_dim, position, rope_theta);

    std::vector<uint16_t>& k_half = session->scratch_k_u16;
    std::vector<uint16_t>& v_half = session->scratch_v_u16;
    k_half.resize(static_cast<size_t>(kv_width));
    v_half.resize(static_cast<size_t>(kv_width));
    for (int64_t index = 0; index < kv_width; ++index) {
        k_half[static_cast<size_t>(index)] = write_u16(k[static_cast<size_t>(index)], session->dtype_code);
        v_half[static_cast<size_t>(index)] = write_u16(v[static_cast<size_t>(index)], session->dtype_code);
    }
    int append_code = append_to_region(session, layer, k_half.data(), v_half.data(), 1, true);
    if (append_code != 0) {
        return 10 + append_code;
    }

    const int64_t total_len = state.committed_len + state.tentative_len;
    std::vector<float>& context = session->scratch_context;
    std::vector<float>& scores = session->scratch_scores;
    context.assign(static_cast<size_t>(hidden_size), 0.0f);
    scores.resize(static_cast<size_t>(total_len));
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int64_t head = 0; head < num_attention_heads; ++head) {
        const int64_t kv_head = head / kv_repeat;
        float max_score = -INFINITY;
        const float* q_base = q.data() + head * head_dim;
        for (int64_t token = 0; token < total_len; ++token) {
            const uint16_t* k_base = token < state.committed_len
                ? state.committed_k.data() + (token * kv_width + kv_head * head_dim)
                : state.tentative_k.data() + ((token - state.committed_len) * kv_width + kv_head * head_dim);
            float score = 0.0f;
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                score += q_base[dim] * read_u16(k_base[dim], session->dtype_code);
            }
            scores[static_cast<size_t>(token)] = score * scale;
            max_score = std::max(max_score, scores[static_cast<size_t>(token)]);
        }
        float denom = 0.0f;
        for (int64_t token = 0; token < total_len; ++token) {
            scores[static_cast<size_t>(token)] = std::exp(scores[static_cast<size_t>(token)] - max_score);
            denom += scores[static_cast<size_t>(token)];
        }
        float* out_head = context.data() + head * head_dim;
        for (int64_t token = 0; token < total_len; ++token) {
            const float weight = scores[static_cast<size_t>(token)] / denom;
            const uint16_t* v_base = token < state.committed_len
                ? state.committed_v.data() + (token * kv_width + kv_head * head_dim)
                : state.tentative_v.data() + ((token - state.committed_len) * kv_width + kv_head * head_dim);
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                out_head[dim] += weight * read_u16(v_base[dim], session->dtype_code);
            }
        }
    }

    std::vector<float>& projected = session->scratch_projected;
    projected.resize(static_cast<size_t>(hidden_size));
    if (!linear_float_u16_blas(context.data(), o_weight, projected.data(), hidden_size, hidden_size, session->dtype_code)) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < hidden_size; ++row) {
            projected[static_cast<size_t>(row)] = dot_float_u16(
                context.data(),
                o_weight + row * hidden_size,
                hidden_size,
                session->dtype_code
            );
        }
    }
    for (int64_t row = 0; row < hidden_size; ++row) {
        out[row] = write_u16(projected[static_cast<size_t>(row)], session->dtype_code);
    }
    return 0;
}

extern "C" __declspec(dllexport) int kv_attention_decode_u16_ext_hd(
    void* handle,
    int64_t layer,
    const uint16_t* hidden,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    const uint16_t* o_weight,
    const uint16_t* q_bias,
    const uint16_t* k_bias,
    const uint16_t* v_bias,
    const uint16_t* q_norm_weight,
    const uint16_t* k_norm_weight,
    uint16_t* out,
    int64_t hidden_size,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float rope_theta,
    float rms_eps
) {
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (
        !valid_layer(session, layer) || hidden == nullptr || q_weight == nullptr || k_weight == nullptr ||
        v_weight == nullptr || o_weight == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (hidden_size <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0) {
        return 2;
    }
    if (num_attention_heads % num_key_value_heads != 0) {
        return 3;
    }
    const int64_t attention_width = num_attention_heads * head_dim;
    const int64_t kv_width = num_key_value_heads * head_dim;
    if (kv_width != session->kv_width) {
        return 4;
    }

    LayerKvState& state = session->layers[static_cast<size_t>(layer)];
    const int64_t position = state.committed_len + state.tentative_len;
    if (position + 1 > session->max_seq_len) {
        return 5;
    }
    const int64_t kv_repeat = num_attention_heads / num_key_value_heads;
    std::vector<float>& q = session->scratch_q;
    std::vector<float>& k = session->scratch_k;
    std::vector<float>& v = session->scratch_v;
    q.resize(static_cast<size_t>(attention_width));
    k.resize(static_cast<size_t>(kv_width));
    v.resize(static_cast<size_t>(kv_width));
    linear_three_same_input(
        hidden,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        q.data(),
        k.data(),
        v.data(),
        hidden_size,
        attention_width,
        kv_width,
        session->dtype_code
    );
    rms_norm_heads_inplace(q.data(), q_norm_weight, num_attention_heads, head_dim, rms_eps, session->dtype_code);
    rms_norm_heads_inplace(k.data(), k_norm_weight, num_key_value_heads, head_dim, rms_eps, session->dtype_code);
    apply_rope_one(q.data(), num_attention_heads, head_dim, position, rope_theta);
    apply_rope_one(k.data(), num_key_value_heads, head_dim, position, rope_theta);

    std::vector<uint16_t>& k_half = session->scratch_k_u16;
    std::vector<uint16_t>& v_half = session->scratch_v_u16;
    k_half.resize(static_cast<size_t>(kv_width));
    v_half.resize(static_cast<size_t>(kv_width));
    for (int64_t index = 0; index < kv_width; ++index) {
        k_half[static_cast<size_t>(index)] = write_u16(k[static_cast<size_t>(index)], session->dtype_code);
        v_half[static_cast<size_t>(index)] = write_u16(v[static_cast<size_t>(index)], session->dtype_code);
    }
    int append_code = append_to_region(session, layer, k_half.data(), v_half.data(), 1, true);
    if (append_code != 0) {
        return 10 + append_code;
    }

    const int64_t total_len = state.committed_len + state.tentative_len;
    std::vector<float>& context = session->scratch_context;
    std::vector<float>& scores = session->scratch_scores;
    context.assign(static_cast<size_t>(attention_width), 0.0f);
    scores.resize(static_cast<size_t>(total_len));
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int64_t head = 0; head < num_attention_heads; ++head) {
        const int64_t kv_head = head / kv_repeat;
        float max_score = -INFINITY;
        const float* q_base = q.data() + head * head_dim;
        for (int64_t token = 0; token < total_len; ++token) {
            const uint16_t* k_base = token < state.committed_len
                ? state.committed_k.data() + (token * kv_width + kv_head * head_dim)
                : state.tentative_k.data() + ((token - state.committed_len) * kv_width + kv_head * head_dim);
            float score = 0.0f;
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                score += q_base[dim] * read_u16(k_base[dim], session->dtype_code);
            }
            scores[static_cast<size_t>(token)] = score * scale;
            max_score = std::max(max_score, scores[static_cast<size_t>(token)]);
        }
        float denom = 0.0f;
        for (int64_t token = 0; token < total_len; ++token) {
            scores[static_cast<size_t>(token)] = std::exp(scores[static_cast<size_t>(token)] - max_score);
            denom += scores[static_cast<size_t>(token)];
        }
        float* out_head = context.data() + head * head_dim;
        for (int64_t token = 0; token < total_len; ++token) {
            const float weight = scores[static_cast<size_t>(token)] / denom;
            const uint16_t* v_base = token < state.committed_len
                ? state.committed_v.data() + (token * kv_width + kv_head * head_dim)
                : state.tentative_v.data() + ((token - state.committed_len) * kv_width + kv_head * head_dim);
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                out_head[dim] += weight * read_u16(v_base[dim], session->dtype_code);
            }
        }
    }

    std::vector<float>& projected = session->scratch_projected;
    projected.resize(static_cast<size_t>(hidden_size));
    if (!linear_float_u16_blas(context.data(), o_weight, projected.data(), attention_width, hidden_size, session->dtype_code)) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < hidden_size; ++row) {
            const uint16_t* weight_row = o_weight + row * attention_width;
            projected[static_cast<size_t>(row)] = dot_float_u16(
                context.data(),
                weight_row,
                attention_width,
                session->dtype_code
            );
        }
    }
    for (int64_t row = 0; row < hidden_size; ++row) {
        out[row] = write_u16(projected[static_cast<size_t>(row)], session->dtype_code);
    }
    return 0;
}

extern "C" __declspec(dllexport) int kv_dense_layer_decode_fp16(
    void* handle,
    int64_t layer,
    const uint16_t* hidden,
    const uint16_t* input_norm_weight,
    const uint16_t* post_norm_weight,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    const uint16_t* o_weight,
    const uint16_t* gate_weight,
    const uint16_t* up_weight,
    const uint16_t* down_weight,
    uint16_t* out,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    float rms_eps,
    float rope_theta
) {
    if (
        handle == nullptr || hidden == nullptr || input_norm_weight == nullptr || post_norm_weight == nullptr ||
        q_weight == nullptr || k_weight == nullptr || v_weight == nullptr || o_weight == nullptr ||
        gate_weight == nullptr || up_weight == nullptr || down_weight == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (hidden_size <= 0 || intermediate_size <= 0) {
        return 2;
    }

    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (session == nullptr) {
        return 3;
    }
    const int dtype_code = session->dtype_code;
    std::vector<float>& input_norm = session->scratch_input_norm;
    input_norm.resize(static_cast<size_t>(hidden_size));
    rms_norm_one(hidden, input_norm_weight, input_norm.data(), hidden_size, rms_eps, dtype_code);
    std::vector<uint16_t>& input_norm_half = session->scratch_input_norm_u16;
    std::vector<uint16_t>& attention_out = session->scratch_attention_out_u16;
    attention_out.resize(static_cast<size_t>(hidden_size));
    const int attention_code = kv_attention_decode_fp16(
        handle,
        layer,
        floats_to_u16_buffer(input_norm, input_norm_half, dtype_code),
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        attention_out.data(),
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        rope_theta
    );
    if (attention_code != 0) {
        return 100 + attention_code;
    }

    std::vector<float>& residual_after_attention = session->scratch_residual;
    residual_after_attention.resize(static_cast<size_t>(hidden_size));
    for (int64_t dim = 0; dim < hidden_size; ++dim) {
        residual_after_attention[static_cast<size_t>(dim)] =
            read_u16(hidden[dim], dtype_code) + read_u16(attention_out[static_cast<size_t>(dim)], dtype_code);
    }

    std::vector<uint16_t>& residual_half = session->scratch_residual_u16;
    std::vector<float>& post_norm = session->scratch_post_norm;
    post_norm.resize(static_cast<size_t>(hidden_size));
    rms_norm_one(floats_to_u16_buffer(residual_after_attention, residual_half, dtype_code), post_norm_weight, post_norm.data(), hidden_size, rms_eps, dtype_code);

    std::vector<float>& gate = session->scratch_gate;
    std::vector<float>& up = session->scratch_up;
    gate.resize(static_cast<size_t>(intermediate_size));
    up.resize(static_cast<size_t>(intermediate_size));
    std::vector<uint16_t>& post_norm_half = session->scratch_post_norm_u16;
    uint16_t* post_norm_half_ptr = floats_to_u16_buffer(post_norm, post_norm_half, dtype_code);
    linear_two_same_input(
        post_norm_half_ptr,
        gate_weight,
        up_weight,
        gate.data(),
        up.data(),
        hidden_size,
        intermediate_size,
        dtype_code
    );
    std::vector<float>& activated = session->scratch_activated;
    activated.resize(static_cast<size_t>(intermediate_size));
    for (int64_t dim = 0; dim < intermediate_size; ++dim) {
        const float g = gate[static_cast<size_t>(dim)];
        activated[static_cast<size_t>(dim)] = (g / (1.0f + std::exp(-g))) * up[static_cast<size_t>(dim)];
    }

    std::vector<float>& mlp_out = session->scratch_mlp_out;
    mlp_out.resize(static_cast<size_t>(hidden_size));
    if (!linear_float_u16_blas(activated.data(), down_weight, mlp_out.data(), intermediate_size, hidden_size, dtype_code)) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < hidden_size; ++row) {
            mlp_out[static_cast<size_t>(row)] = dot_float_u16(
                activated.data(),
                down_weight + row * intermediate_size,
                intermediate_size,
                dtype_code
            );
        }
    }

    for (int64_t dim = 0; dim < hidden_size; ++dim) {
        out[dim] = write_u16(residual_after_attention[static_cast<size_t>(dim)] + mlp_out[static_cast<size_t>(dim)], dtype_code);
    }
    return 0;
}

extern "C" __declspec(dllexport) int kv_dense_layer_prefill_fp16(
    void* handle,
    int64_t layer,
    const uint16_t* hidden,
    const uint16_t* input_norm_weight,
    const uint16_t* post_norm_weight,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    const uint16_t* o_weight,
    const uint16_t* gate_weight,
    const uint16_t* up_weight,
    const uint16_t* down_weight,
    uint16_t* out,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    float rms_eps,
    float rope_theta
) {
    if (
        handle == nullptr || hidden == nullptr || input_norm_weight == nullptr || post_norm_weight == nullptr ||
        q_weight == nullptr || k_weight == nullptr || v_weight == nullptr || o_weight == nullptr ||
        gate_weight == nullptr || up_weight == nullptr || down_weight == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (seq_len <= 0 || hidden_size <= 0 || intermediate_size <= 0) {
        return 2;
    }
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (!valid_layer(session, layer)) {
        return 3;
    }
    LayerKvState& state = session->layers[static_cast<size_t>(layer)];
    if (state.committed_len + state.tentative_len + seq_len > session->max_seq_len) {
        return 4;
    }

    for (int64_t token = 0; token < seq_len; ++token) {
        const uint16_t* token_hidden = hidden + token * hidden_size;
        uint16_t* token_out = out + token * hidden_size;
        const int decode_code = kv_dense_layer_decode_fp16(
            handle,
            layer,
            token_hidden,
            input_norm_weight,
            post_norm_weight,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            gate_weight,
            up_weight,
            down_weight,
            token_out,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            rms_eps,
            rope_theta
        );
        if (decode_code != 0) {
            return 100 + decode_code;
        }
        const int commit_code = kv_commit(handle, 1);
        if (commit_code != 0) {
            return 200 + commit_code;
        }
    }
    return 0;
}

extern "C" __declspec(dllexport) int kv_dense_layer_decode_u16_ext(
    void* handle,
    int64_t layer,
    const uint16_t* hidden,
    const uint16_t* input_norm_weight,
    const uint16_t* post_norm_weight,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    const uint16_t* o_weight,
    const uint16_t* gate_weight,
    const uint16_t* up_weight,
    const uint16_t* down_weight,
    const uint16_t* q_bias,
    const uint16_t* k_bias,
    const uint16_t* v_bias,
    const uint16_t* q_norm_weight,
    const uint16_t* k_norm_weight,
    uint16_t* out,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    float rms_eps,
    float rope_theta
) {
    if (
        handle == nullptr || hidden == nullptr || input_norm_weight == nullptr || post_norm_weight == nullptr ||
        q_weight == nullptr || k_weight == nullptr || v_weight == nullptr || o_weight == nullptr ||
        gate_weight == nullptr || up_weight == nullptr || down_weight == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (hidden_size <= 0 || intermediate_size <= 0) {
        return 2;
    }
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (session == nullptr) {
        return 3;
    }
    const int dtype_code = session->dtype_code;

    std::vector<float>& input_norm = session->scratch_input_norm;
    input_norm.resize(static_cast<size_t>(hidden_size));
    rms_norm_one(hidden, input_norm_weight, input_norm.data(), hidden_size, rms_eps, dtype_code);
    std::vector<uint16_t>& input_norm_storage = session->scratch_input_norm_u16;
    std::vector<uint16_t>& attention_out = session->scratch_attention_out_u16;
    attention_out.resize(static_cast<size_t>(hidden_size));
    const int attention_code = kv_attention_decode_u16_ext(
        handle,
        layer,
        floats_to_u16_buffer(input_norm, input_norm_storage, dtype_code),
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        q_bias,
        k_bias,
        v_bias,
        q_norm_weight,
        k_norm_weight,
        attention_out.data(),
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        rope_theta,
        rms_eps
    );
    if (attention_code != 0) {
        return 100 + attention_code;
    }

    std::vector<float>& residual_after_attention = session->scratch_residual;
    residual_after_attention.resize(static_cast<size_t>(hidden_size));
    for (int64_t dim = 0; dim < hidden_size; ++dim) {
        residual_after_attention[static_cast<size_t>(dim)] =
            read_u16(hidden[dim], dtype_code) + read_u16(attention_out[static_cast<size_t>(dim)], dtype_code);
    }

    std::vector<uint16_t>& residual_storage = session->scratch_residual_u16;
    std::vector<float>& post_norm = session->scratch_post_norm;
    post_norm.resize(static_cast<size_t>(hidden_size));
    rms_norm_one(
        floats_to_u16_buffer(residual_after_attention, residual_storage, dtype_code),
        post_norm_weight,
        post_norm.data(),
        hidden_size,
        rms_eps,
        dtype_code
    );

    std::vector<uint16_t>& post_norm_storage = session->scratch_post_norm_u16;
    uint16_t* post_norm_ptr = floats_to_u16_buffer(post_norm, post_norm_storage, dtype_code);
    std::vector<float>& gate = session->scratch_gate;
    std::vector<float>& up = session->scratch_up;
    gate.resize(static_cast<size_t>(intermediate_size));
    up.resize(static_cast<size_t>(intermediate_size));
    linear_two_same_input(
        post_norm_ptr,
        gate_weight,
        up_weight,
        gate.data(),
        up.data(),
        hidden_size,
        intermediate_size,
        dtype_code
    );

    std::vector<float>& activated = session->scratch_activated;
    activated.resize(static_cast<size_t>(intermediate_size));
    for (int64_t dim = 0; dim < intermediate_size; ++dim) {
        const float g = gate[static_cast<size_t>(dim)];
        activated[static_cast<size_t>(dim)] = (g / (1.0f + std::exp(-g))) * up[static_cast<size_t>(dim)];
    }

    std::vector<float>& mlp_out = session->scratch_mlp_out;
    mlp_out.resize(static_cast<size_t>(hidden_size));
    if (!linear_float_u16_blas(activated.data(), down_weight, mlp_out.data(), intermediate_size, hidden_size, dtype_code)) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < hidden_size; ++row) {
            mlp_out[static_cast<size_t>(row)] = dot_float_u16(
                activated.data(),
                down_weight + row * intermediate_size,
                intermediate_size,
                dtype_code
            );
        }
    }

    for (int64_t dim = 0; dim < hidden_size; ++dim) {
        out[dim] = write_u16(residual_after_attention[static_cast<size_t>(dim)] + mlp_out[static_cast<size_t>(dim)], dtype_code);
    }
    return 0;
}

extern "C" __declspec(dllexport) int kv_dense_layer_decode_u16_ext_packed_rows8(
    void* handle,
    int64_t layer,
    const uint16_t* hidden,
    const uint16_t* input_norm_weight,
    const uint16_t* post_norm_weight,
    const uint16_t* q_weight_packed,
    const uint16_t* k_weight_packed,
    const uint16_t* v_weight_packed,
    const uint16_t* o_weight_packed,
    const uint16_t* gate_weight_packed,
    const uint16_t* up_weight_packed,
    const uint16_t* down_weight_packed,
    const uint16_t* q_bias,
    const uint16_t* k_bias,
    const uint16_t* v_bias,
    const uint16_t* q_norm_weight,
    const uint16_t* k_norm_weight,
    uint16_t* out,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    float rms_eps,
    float rope_theta
) {
    const bool previous = g_use_prepacked_rows8;
    g_use_prepacked_rows8 = true;
    const int code = kv_dense_layer_decode_u16_ext(
        handle,
        layer,
        hidden,
        input_norm_weight,
        post_norm_weight,
        q_weight_packed,
        k_weight_packed,
        v_weight_packed,
        o_weight_packed,
        gate_weight_packed,
        up_weight_packed,
        down_weight_packed,
        q_bias,
        k_bias,
        v_bias,
        q_norm_weight,
        k_norm_weight,
        out,
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        rms_eps,
        rope_theta
    );
    g_use_prepacked_rows8 = previous;
    return code;
}

extern "C" __declspec(dllexport) int kv_dense_layer_prefill_u16_ext(
    void* handle,
    int64_t layer,
    const uint16_t* hidden,
    const uint16_t* input_norm_weight,
    const uint16_t* post_norm_weight,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const uint16_t* v_weight,
    const uint16_t* o_weight,
    const uint16_t* gate_weight,
    const uint16_t* up_weight,
    const uint16_t* down_weight,
    const uint16_t* q_bias,
    const uint16_t* k_bias,
    const uint16_t* v_bias,
    const uint16_t* q_norm_weight,
    const uint16_t* k_norm_weight,
    uint16_t* out,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_attention_heads,
    int64_t num_key_value_heads,
    float rms_eps,
    float rope_theta
) {
    if (
        handle == nullptr || hidden == nullptr || input_norm_weight == nullptr || post_norm_weight == nullptr ||
        q_weight == nullptr || k_weight == nullptr || v_weight == nullptr || o_weight == nullptr ||
        gate_weight == nullptr || up_weight == nullptr || down_weight == nullptr || out == nullptr
    ) {
        return 1;
    }
    if (seq_len <= 0 || hidden_size <= 0 || intermediate_size <= 0) {
        return 2;
    }
    KvSession* session = reinterpret_cast<KvSession*>(handle);
    if (!valid_layer(session, layer)) {
        return 3;
    }
    LayerKvState& state = session->layers[static_cast<size_t>(layer)];
    if (state.committed_len + state.tentative_len + seq_len > session->max_seq_len) {
        return 4;
    }

    for (int64_t token = 0; token < seq_len; ++token) {
        const int decode_code = kv_dense_layer_decode_u16_ext(
            handle,
            layer,
            hidden + token * hidden_size,
            input_norm_weight,
            post_norm_weight,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            gate_weight,
            up_weight,
            down_weight,
            q_bias,
            k_bias,
            v_bias,
            q_norm_weight,
            k_norm_weight,
            out + token * hidden_size,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            rms_eps,
            rope_theta
        );
        if (decode_code != 0) {
            return 100 + decode_code;
        }
        const int commit_code = kv_commit(handle, 1);
        if (commit_code != 0) {
            return 200 + commit_code;
        }
    }
    return 0;
}
