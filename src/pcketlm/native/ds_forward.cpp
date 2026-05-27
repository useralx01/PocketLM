#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>
#include <immintrin.h>
#include <intrin.h>
#include <windows.h>

using DsTensorCallback = int (*)(
    int64_t layer_idx,
    int64_t layer_kind,
    int64_t weight_role,
    const void** out_ptr,
    int64_t* out_nbytes
);
using DsAttentionCallback = int (*)(
    int64_t layer_idx,
    const float* hidden_in,
    int64_t batch,
    int64_t hidden_dim,
    const float** out_ptr,
    int64_t* out_count
);
using DsDecodeCallback = int (*)(
    int64_t input_token_id,
    const float** out_ptr,
    int64_t* out_count
);
using DsPrefillCallback = int (*)(
    const int64_t* input_token_ids,
    int64_t n_tokens,
    const float** out_ptr,
    int64_t* out_count
);
using DsVerifyCallback = int (*)(
    const int64_t* candidate_token_ids,
    int64_t k,
    const float** out_ptr,
    int64_t* out_count
);
using Fp8MlpManyFn = int (*)(
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    const float*,
    float*,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t
);
using Fp8MlpManyWeightedFn = int (*)(
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    const float*,
    const float*,
    float*,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t
);

struct DsLayerRegistration {
    int64_t layer_idx = 0;
    int64_t layer_kind = 0;
    int64_t weight_role = 0;
    const void* fp8_ptr = nullptr;
    int64_t fp8_nbytes = 0;
    const void* scale_ptr = nullptr;
    int64_t scale_nbytes = 0;
};

struct DsSession {
    int64_t num_layers = 0;
    int64_t hidden_dim = 0;
    int64_t num_experts = 0;
    int64_t top_k = 0;
    void* fp8_pack_callback = nullptr;
    void* kv_session_handle = nullptr;
    int64_t monolithic_calls = 0;
    int64_t callback_invocations = 0;
    int64_t expert_invocations = 0;
    int64_t attention_invocations = 0;
    int64_t fused_attention_invocations = 0;
    int64_t layers_executed = 0;
    std::vector<DsLayerRegistration> registrations;
    std::vector<float> scratch_q_low;
    std::vector<float> scratch_q;
    std::vector<float> scratch_kv;
    std::vector<float> scratch_q_nope;
    std::vector<float> scratch_q_pe;
    std::vector<float> scratch_attention_heads;
};

static inline float fp16_to_fp32(uint16_t value) {
    const __m128i half = _mm_cvtsi32_si128(static_cast<int>(value));
    const __m128 full = _mm_cvtph_ps(half);
    return _mm_cvtss_f32(full);
}

static inline float bf16_to_fp32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

static inline float read_u16(uint16_t value, int64_t dtype_code) {
    return dtype_code == 1 ? bf16_to_fp32(value) : fp16_to_fp32(value);
}

static HMODULE load_sibling_dll(const char* name) {
    char self_path[MAX_PATH] = {0};
    HMODULE self_module = nullptr;
    if (GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&load_sibling_dll),
            &self_module
        ) && GetModuleFileNameA(self_module, self_path, MAX_PATH) > 0) {
        std::string path(self_path);
        const size_t slash = path.find_last_of("\\/");
        if (slash != std::string::npos) {
            const std::string full_name = path.substr(0, slash + 1) + name;
            return LoadLibraryA(full_name.c_str());
        }
    }
    return LoadLibraryA(name);
}

static Fp8MlpManyFn fp8_mlp_many_fn() {
    static HMODULE module = nullptr;
    static Fp8MlpManyFn fn = nullptr;
    if (fn != nullptr) {
        return fn;
    }
    if (module == nullptr) {
        module = load_sibling_dll("fp8_linear.dll");
    }
    if (module == nullptr) {
        return nullptr;
    }
    fn = reinterpret_cast<Fp8MlpManyFn>(GetProcAddress(module, "fp8_e4m3_block_mlp_many_f32"));
    return fn;
}

static Fp8MlpManyWeightedFn fp8_mlp_many_weighted_fn() {
    static HMODULE module = nullptr;
    static Fp8MlpManyWeightedFn fn = nullptr;
    if (fn != nullptr) {
        return fn;
    }
    if (module == nullptr) {
        module = load_sibling_dll("fp8_linear.dll");
    }
    if (module == nullptr) {
        return nullptr;
    }
    fn = reinterpret_cast<Fp8MlpManyWeightedFn>(GetProcAddress(module, "fp8_e4m3_block_mlp_many_weighted_f32"));
    return fn;
}

static std::string registration_key(int64_t layer_idx, int64_t layer_kind, int64_t weight_role) {
    return std::to_string(layer_idx) + ":" + std::to_string(layer_kind) + ":" + std::to_string(weight_role);
}

extern "C" __declspec(dllexport) int ds_cpu_has_avx2(void) {
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 7) {
        return 0;
    }
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) ? 1 : 0;
}

extern "C" __declspec(dllexport) void* ds_session_create(
    int64_t num_layers,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    void* fp8_pack_callback,
    void* kv_session_handle
) {
    if (num_layers <= 0 || hidden_dim <= 0 || num_experts < 0 || top_k < 0) {
        return nullptr;
    }
    DsSession* session = new (std::nothrow) DsSession();
    if (session == nullptr) {
        return nullptr;
    }
    session->num_layers = num_layers;
    session->hidden_dim = hidden_dim;
    session->num_experts = num_experts;
    session->top_k = top_k;
    session->fp8_pack_callback = fp8_pack_callback;
    session->kv_session_handle = kv_session_handle;
    return session;
}

extern "C" __declspec(dllexport) void ds_session_destroy(void* handle) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    delete session;
}

extern "C" __declspec(dllexport) int ds_session_register_layer(
    void* handle,
    int64_t layer_idx,
    int64_t layer_kind,
    int64_t weight_role,
    DsTensorCallback fp8_ptr_callback,
    DsTensorCallback scale_ptr_callback
) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr || layer_idx < 0 || layer_idx >= session->num_layers) {
        return 1;
    }
    if (fp8_ptr_callback == nullptr || scale_ptr_callback == nullptr) {
        return 2;
    }

    const void* fp8_ptr = nullptr;
    int64_t fp8_nbytes = 0;
    const int fp8_code = fp8_ptr_callback(layer_idx, layer_kind, weight_role, &fp8_ptr, &fp8_nbytes);
    session->callback_invocations += 1;
    if (fp8_code != 0 || fp8_ptr == nullptr || fp8_nbytes <= 0) {
        return 10 + fp8_code;
    }

    const void* scale_ptr = nullptr;
    int64_t scale_nbytes = 0;
    const int scale_code = scale_ptr_callback(layer_idx, layer_kind, weight_role, &scale_ptr, &scale_nbytes);
    session->callback_invocations += 1;
    if (scale_code != 0 || scale_ptr == nullptr || scale_nbytes <= 0) {
        return 20 + scale_code;
    }

    DsLayerRegistration registration;
    registration.layer_idx = layer_idx;
    registration.layer_kind = layer_kind;
    registration.weight_role = weight_role;
    registration.fp8_ptr = fp8_ptr;
    registration.fp8_nbytes = fp8_nbytes;
    registration.scale_ptr = scale_ptr;
    registration.scale_nbytes = scale_nbytes;
    session->registrations.push_back(registration);
    return 0;
}

extern "C" __declspec(dllexport) int64_t ds_monolithic_call_counter(void* handle) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return session->monolithic_calls;
}

extern "C" __declspec(dllexport) int64_t ds_callback_invocation_count(void* handle) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return session->callback_invocations;
}

extern "C" __declspec(dllexport) int64_t ds_expert_invocation_count(void* handle) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return session->expert_invocations;
}

extern "C" __declspec(dllexport) int64_t ds_attention_invocation_count(void* handle) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return session->attention_invocations;
}

extern "C" __declspec(dllexport) int64_t ds_fused_attention_invocation_count(void* handle) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return session->fused_attention_invocations;
}

extern "C" __declspec(dllexport) int64_t ds_layers_executed_count(void* handle) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return session->layers_executed;
}

extern "C" __declspec(dllexport) int64_t ds_registered_layer_count(void* handle) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return static_cast<int64_t>(session->registrations.size());
}

extern "C" __declspec(dllexport) int64_t ds_registered_fp8_nbytes(void* handle, int64_t index) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr || index < 0 || index >= static_cast<int64_t>(session->registrations.size())) {
        return -1;
    }
    return session->registrations[static_cast<size_t>(index)].fp8_nbytes;
}

extern "C" __declspec(dllexport) int64_t ds_registered_scale_nbytes(void* handle, int64_t index) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr || index < 0 || index >= static_cast<int64_t>(session->registrations.size())) {
        return -1;
    }
    return session->registrations[static_cast<size_t>(index)].scale_nbytes;
}

extern "C" __declspec(dllexport) int ds_moe_layer_forward_f32(
    void* handle,
    const float* expert_outputs,
    const float* route_weights,
    int64_t selected_count,
    int64_t batch,
    int64_t hidden_dim,
    float* hidden_out
) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (
        session == nullptr || expert_outputs == nullptr ||
        route_weights == nullptr || hidden_out == nullptr
    ) {
        return 1;
    }
    if (selected_count <= 0 || batch <= 0 || hidden_dim <= 0) {
        return 2;
    }
    const int64_t batch_width = batch * hidden_dim;
    for (int64_t offset = 0; offset < batch_width; ++offset) {
        float acc = 0.0f;
        for (int64_t expert = 0; expert < selected_count; ++expert) {
            acc += route_weights[expert] * expert_outputs[expert * batch_width + offset];
        }
        hidden_out[offset] = acc;
    }
    session->monolithic_calls += 1;
    session->expert_invocations += selected_count;
    return 0;
}

extern "C" __declspec(dllexport) int ds_attention_layer_forward_f32(
    void* handle,
    int64_t layer_idx,
    const float* hidden_in,
    int64_t batch,
    int64_t hidden_dim,
    DsAttentionCallback attention_callback,
    float* hidden_out
) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr || hidden_in == nullptr || attention_callback == nullptr || hidden_out == nullptr) {
        return 1;
    }
    if (layer_idx < 0 || layer_idx >= session->num_layers || batch <= 0 || hidden_dim <= 0) {
        return 2;
    }
    const float* callback_out = nullptr;
    int64_t callback_count = 0;
    const int code = attention_callback(layer_idx, hidden_in, batch, hidden_dim, &callback_out, &callback_count);
    session->callback_invocations += 1;
    if (code != 0 || callback_out == nullptr) {
        return 10 + code;
    }
    const int64_t expected = batch * hidden_dim;
    if (callback_count != expected) {
        return 20;
    }
    std::memcpy(hidden_out, callback_out, static_cast<size_t>(expected) * sizeof(float));
    session->monolithic_calls += 1;
    session->attention_invocations += 1;
    return 0;
}

extern "C" __declspec(dllexport) int ds_forward_decode_f32(
    void* handle,
    int64_t input_token_id,
    DsDecodeCallback decode_callback,
    float* output_logits,
    int64_t vocab_size
) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (session == nullptr || decode_callback == nullptr || output_logits == nullptr) {
        return 1;
    }
    if (vocab_size <= 0) {
        return 2;
    }
    const float* callback_out = nullptr;
    int64_t callback_count = 0;
    const int code = decode_callback(input_token_id, &callback_out, &callback_count);
    session->callback_invocations += 1;
    if (code != 0 || callback_out == nullptr) {
        return 10 + code;
    }
    if (callback_count != vocab_size) {
        return 20;
    }
    std::memcpy(output_logits, callback_out, static_cast<size_t>(vocab_size) * sizeof(float));
    session->monolithic_calls += 1;
    session->layers_executed += session->num_layers;
    return 0;
}

extern "C" __declspec(dllexport) int ds_forward_prefill_f32(
    void* handle,
    const int64_t* input_token_ids,
    int64_t n_tokens,
    DsPrefillCallback prefill_callback,
    float* output_final_logits,
    int64_t vocab_size
) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (
        session == nullptr || input_token_ids == nullptr || prefill_callback == nullptr ||
        output_final_logits == nullptr
    ) {
        return 1;
    }
    if (n_tokens <= 0 || vocab_size <= 0) {
        return 2;
    }
    const float* callback_out = nullptr;
    int64_t callback_count = 0;
    const int code = prefill_callback(input_token_ids, n_tokens, &callback_out, &callback_count);
    session->callback_invocations += 1;
    if (code != 0 || callback_out == nullptr) {
        return 10 + code;
    }
    if (callback_count != vocab_size) {
        return 20;
    }
    std::memcpy(output_final_logits, callback_out, static_cast<size_t>(vocab_size) * sizeof(float));
    session->monolithic_calls += 1;
    session->layers_executed += session->num_layers * n_tokens;
    return 0;
}

extern "C" __declspec(dllexport) int ds_forward_verify_f32(
    void* handle,
    const int64_t* candidate_token_ids,
    int64_t k,
    DsVerifyCallback verify_callback,
    float* output_logits_per_position,
    int64_t vocab_size
) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (
        session == nullptr || candidate_token_ids == nullptr || verify_callback == nullptr ||
        output_logits_per_position == nullptr
    ) {
        return 1;
    }
    if (k <= 0 || vocab_size <= 0) {
        return 2;
    }
    const float* callback_out = nullptr;
    int64_t callback_count = 0;
    const int code = verify_callback(candidate_token_ids, k, &callback_out, &callback_count);
    session->callback_invocations += 1;
    if (code != 0 || callback_out == nullptr) {
        return 10 + code;
    }
    if (callback_count != k * vocab_size) {
        return 20;
    }
    std::memcpy(output_logits_per_position, callback_out, static_cast<size_t>(callback_count) * sizeof(float));
    session->monolithic_calls += 1;
    session->layers_executed += session->num_layers * k;
    return 0;
}

static inline float dot_product_f32(const float* left, const float* right, int64_t count) {
    __m256 acc = _mm256_setzero_ps();
    int64_t index = 0;
    for (; index + 8 <= count; index += 8) {
        const __m256 a = _mm256_loadu_ps(left + index);
        const __m256 b = _mm256_loadu_ps(right + index);
        acc = _mm256_fmadd_ps(a, b, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; index < count; ++index) {
        sum += left[index] * right[index];
    }
    return sum;
}

extern "C" __declspec(dllexport) int ds_mla_attention_flash_forward(
    const float* q_nope,
    const float* q_pe,
    const float* kv_cache,
    const float* pe_cache,
    const float* wkv_b,
    float* output_heads,
    int64_t num_heads,
    int64_t cache_len,
    int64_t qk_nope_dim,
    int64_t qk_rope_dim,
    int64_t kv_lora_rank,
    int64_t v_head_dim,
    float softmax_scale
) {
    if (
        q_nope == nullptr || q_pe == nullptr || kv_cache == nullptr || pe_cache == nullptr ||
        wkv_b == nullptr || output_heads == nullptr
    ) {
        return 1;
    }
    if (
        num_heads <= 0 || cache_len <= 0 || qk_nope_dim < 0 || qk_rope_dim < 0 ||
        kv_lora_rank <= 0 || v_head_dim <= 0
    ) {
        return 2;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t head = 0; head < num_heads; ++head) {
        const float* q_nope_head = q_nope + head * qk_nope_dim;
        const float* q_pe_head = q_pe + head * qk_rope_dim;
        const float* wkv_head = wkv_b + head * (qk_nope_dim + v_head_dim) * kv_lora_rank;

        std::vector<float> q_abs(static_cast<size_t>(kv_lora_rank), 0.0f);
        for (int64_t latent = 0; latent < kv_lora_rank; ++latent) {
            float acc = 0.0f;
            for (int64_t dim = 0; dim < qk_nope_dim; ++dim) {
                acc += q_nope_head[dim] * wkv_head[dim * kv_lora_rank + latent];
            }
            q_abs[static_cast<size_t>(latent)] = acc;
        }

        std::vector<float> latent_acc(static_cast<size_t>(kv_lora_rank), 0.0f);
        float running_max = -INFINITY;
        float running_sum = 0.0f;
        for (int64_t pos = 0; pos < cache_len; ++pos) {
            const float* kv_row = kv_cache + pos * kv_lora_rank;
            const float* pe_row = pe_cache + pos * qk_rope_dim;
            float score = dot_product_f32(q_abs.data(), kv_row, kv_lora_rank);
            if (qk_rope_dim > 0) {
                score += dot_product_f32(q_pe_head, pe_row, qk_rope_dim);
            }
            score *= softmax_scale;
            const float new_max = running_max > score ? running_max : score;
            const float old_scale = std::isfinite(running_max) ? std::exp(running_max - new_max) : 0.0f;
            const float score_scale = std::exp(score - new_max);
            for (int64_t latent = 0; latent < kv_lora_rank; ++latent) {
                latent_acc[static_cast<size_t>(latent)] =
                    latent_acc[static_cast<size_t>(latent)] * old_scale + kv_row[latent] * score_scale;
            }
            running_sum = running_sum * old_scale + score_scale;
            running_max = new_max;
        }
        const float inv_sum = running_sum > 0.0f ? (1.0f / running_sum) : 0.0f;
        for (int64_t latent = 0; latent < kv_lora_rank; ++latent) {
            latent_acc[static_cast<size_t>(latent)] *= inv_sum;
        }

        float* out_head = output_heads + head * v_head_dim;
        const float* value_projection = wkv_head + qk_nope_dim * kv_lora_rank;
        for (int64_t out_dim = 0; out_dim < v_head_dim; ++out_dim) {
            out_head[out_dim] = dot_product_f32(
                latent_acc.data(),
                value_projection + out_dim * kv_lora_rank,
                kv_lora_rank
            );
        }
    }
    return 0;
}

static void linear_f32_rowmajor(
    const float* input,
    const float* weight,
    float* output,
    int64_t out_rows,
    int64_t in_cols
) {
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < out_rows; ++row) {
        output[row] = dot_product_f32(input, weight + row * in_cols, in_cols);
    }
}

static void rms_norm_inplace_f32(float* values, const float* norm_weight, int64_t dim, float eps) {
    float sum_sq = 0.0f;
    for (int64_t index = 0; index < dim; ++index) {
        sum_sq += values[index] * values[index];
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
    for (int64_t index = 0; index < dim; ++index) {
        values[index] = values[index] * inv_rms * norm_weight[index];
    }
}

static void apply_rope_groups_inplace_f32(
    float* values,
    int64_t group_count,
    int64_t rope_dim,
    const float* cos_values,
    const float* sin_values
) {
    if (rope_dim <= 0 || (rope_dim % 2) != 0) {
        return;
    }
    const int64_t pairs = rope_dim / 2;
    for (int64_t group = 0; group < group_count; ++group) {
        float* base = values + group * rope_dim;
        for (int64_t pair = 0; pair < pairs; ++pair) {
            const float x0 = base[pair * 2];
            const float x1 = base[pair * 2 + 1];
            const float c = cos_values[pair];
            const float s = sin_values[pair];
            base[pair * 2] = x0 * c - x1 * s;
            base[pair * 2 + 1] = x0 * s + x1 * c;
        }
    }
}

extern "C" __declspec(dllexport) int ds_attention_block_forward_f32(
    void* handle,
    const float* hidden_in,
    const float* q_a_weight,
    const float* q_b_weight,
    const float* kv_a_weight,
    const float* kv_b_weight,
    const float* o_weight,
    const float* q_norm_weight,
    const float* kv_norm_weight,
    const float* previous_kv_cache,
    const float* previous_pe_cache,
    const float* rope_cos,
    const float* rope_sin,
    float* next_kv_cache,
    float* next_pe_cache,
    float* hidden_out,
    int64_t hidden_dim,
    int64_t q_lora_rank,
    int64_t kv_lora_rank,
    int64_t num_heads,
    int64_t qk_nope_dim,
    int64_t qk_rope_dim,
    int64_t v_head_dim,
    int64_t previous_len,
    float rms_eps,
    float softmax_scale
) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (
        session == nullptr || hidden_in == nullptr || q_a_weight == nullptr || q_b_weight == nullptr ||
        kv_a_weight == nullptr || kv_b_weight == nullptr || o_weight == nullptr ||
        q_norm_weight == nullptr || kv_norm_weight == nullptr || rope_cos == nullptr || rope_sin == nullptr ||
        next_kv_cache == nullptr || next_pe_cache == nullptr || hidden_out == nullptr
    ) {
        return 1;
    }
    if (
        hidden_dim <= 0 || q_lora_rank <= 0 || kv_lora_rank <= 0 || num_heads <= 0 ||
        qk_nope_dim < 0 || qk_rope_dim < 0 || v_head_dim <= 0 || previous_len < 0
    ) {
        return 2;
    }
    if (previous_len > 0 && (previous_kv_cache == nullptr || previous_pe_cache == nullptr)) {
        return 3;
    }

    const int64_t head_q_width = qk_nope_dim + qk_rope_dim;
    const int64_t q_rows = num_heads * head_q_width;
    const int64_t kv_rows = kv_lora_rank + qk_rope_dim;
    const int64_t attention_width = num_heads * v_head_dim;
    const int64_t cache_len = previous_len + 1;

    std::vector<float>& q_low = session->scratch_q_low;
    std::vector<float>& q_nope = session->scratch_q_nope;
    std::vector<float>& q_pe = session->scratch_q_pe;
    std::vector<float>& attention_heads = session->scratch_attention_heads;
    q_low.assign(static_cast<size_t>(q_lora_rank), 0.0f);
    q_nope.assign(static_cast<size_t>(num_heads * qk_nope_dim), 0.0f);
    q_pe.assign(static_cast<size_t>(num_heads * qk_rope_dim), 0.0f);
    attention_heads.assign(static_cast<size_t>(attention_width), 0.0f);

    linear_f32_rowmajor(hidden_in, q_a_weight, q_low.data(), q_lora_rank, hidden_dim);
    rms_norm_inplace_f32(q_low.data(), q_norm_weight, q_lora_rank, rms_eps);

    #pragma omp parallel for schedule(static)
    for (int64_t head = 0; head < num_heads; ++head) {
        for (int64_t dim = 0; dim < qk_nope_dim; ++dim) {
            const int64_t row = head * head_q_width + dim;
            q_nope[static_cast<size_t>(head * qk_nope_dim + dim)] =
                dot_product_f32(q_low.data(), q_b_weight + row * q_lora_rank, q_lora_rank);
        }
        for (int64_t dim = 0; dim < qk_rope_dim; ++dim) {
            const int64_t row = head * head_q_width + qk_nope_dim + dim;
            q_pe[static_cast<size_t>(head * qk_rope_dim + dim)] =
                dot_product_f32(q_low.data(), q_b_weight + row * q_lora_rank, q_lora_rank);
        }
    }
    if (qk_rope_dim > 0) {
        apply_rope_groups_inplace_f32(q_pe.data(), num_heads, qk_rope_dim, rope_cos, rope_sin);
    }

    float* new_kv = next_kv_cache + previous_len * kv_lora_rank;
    float* new_pe = next_pe_cache + previous_len * qk_rope_dim;
    if (previous_len > 0) {
        std::memcpy(
            next_kv_cache,
            previous_kv_cache,
            static_cast<size_t>(previous_len * kv_lora_rank) * sizeof(float)
        );
        if (qk_rope_dim > 0) {
            std::memcpy(
                next_pe_cache,
                previous_pe_cache,
                static_cast<size_t>(previous_len * qk_rope_dim) * sizeof(float)
            );
        }
    }
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < kv_lora_rank; ++row) {
        new_kv[row] = dot_product_f32(hidden_in, kv_a_weight + row * hidden_dim, hidden_dim);
    }
    rms_norm_inplace_f32(new_kv, kv_norm_weight, kv_lora_rank, rms_eps);
    if (qk_rope_dim > 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < qk_rope_dim; ++row) {
            new_pe[row] = dot_product_f32(hidden_in, kv_a_weight + (kv_lora_rank + row) * hidden_dim, hidden_dim);
        }
        apply_rope_groups_inplace_f32(new_pe, 1, qk_rope_dim, rope_cos, rope_sin);
    }

    const int flash_code = ds_mla_attention_flash_forward(
        q_nope.data(),
        q_pe.data(),
        next_kv_cache,
        next_pe_cache,
        kv_b_weight,
        attention_heads.data(),
        num_heads,
        cache_len,
        qk_nope_dim,
        qk_rope_dim,
        kv_lora_rank,
        v_head_dim,
        softmax_scale
    );
    if (flash_code != 0) {
        return 100 + flash_code;
    }

    linear_f32_rowmajor(attention_heads.data(), o_weight, hidden_out, hidden_dim, attention_width);
    session->monolithic_calls += 1;
    session->attention_invocations += 1;
    session->fused_attention_invocations += 1;
    return 0;
}

extern "C" __declspec(dllexport) int ds_moe_layer_forward_fp8_f32(
    void* handle,
    const uint64_t* gate_weight_ptrs,
    const uint64_t* gate_scale_ptrs,
    const uint64_t* up_weight_ptrs,
    const uint64_t* up_scale_ptrs,
    const uint64_t* down_weight_ptrs,
    const uint64_t* down_scale_ptrs,
    const float* hidden,
    const float* route_weights,
    int64_t selected_count,
    int64_t batch,
    int64_t intermediate_rows,
    int64_t hidden_dim,
    int64_t gate_scale_cols,
    int64_t down_scale_cols,
    float* hidden_out
) {
    DsSession* session = reinterpret_cast<DsSession*>(handle);
    if (
        session == nullptr || gate_weight_ptrs == nullptr || gate_scale_ptrs == nullptr ||
        up_weight_ptrs == nullptr || up_scale_ptrs == nullptr ||
        down_weight_ptrs == nullptr || down_scale_ptrs == nullptr ||
        hidden == nullptr || route_weights == nullptr || hidden_out == nullptr
    ) {
        return 1;
    }
    if (
        selected_count <= 0 || batch <= 0 || intermediate_rows <= 0 ||
        hidden_dim <= 0 || gate_scale_cols <= 0 || down_scale_cols <= 0
    ) {
        return 2;
    }
    Fp8MlpManyWeightedFn weighted_fn = fp8_mlp_many_weighted_fn();
    if (weighted_fn != nullptr) {
        const int weighted_code = weighted_fn(
            gate_weight_ptrs,
            gate_scale_ptrs,
            up_weight_ptrs,
            up_scale_ptrs,
            down_weight_ptrs,
            down_scale_ptrs,
            hidden,
            route_weights,
            hidden_out,
            selected_count,
            batch,
            intermediate_rows,
            hidden_dim,
            gate_scale_cols,
            down_scale_cols
        );
        if (weighted_code == 0) {
            session->monolithic_calls += 1;
            session->expert_invocations += selected_count;
            return 0;
        }
    }
    Fp8MlpManyFn fn = fp8_mlp_many_fn();
    if (fn == nullptr) {
        return 3;
    }
    std::vector<float> expert_outputs(static_cast<size_t>(selected_count * batch * hidden_dim), 0.0f);
    const int mlp_code = fn(
        gate_weight_ptrs,
        gate_scale_ptrs,
        up_weight_ptrs,
        up_scale_ptrs,
        down_weight_ptrs,
        down_scale_ptrs,
        hidden,
        expert_outputs.data(),
        selected_count,
        batch,
        intermediate_rows,
        hidden_dim,
        gate_scale_cols,
        down_scale_cols
    );
    if (mlp_code != 0) {
        return 100 + mlp_code;
    }
    const int64_t batch_width = batch * hidden_dim;
    for (int64_t offset = 0; offset < batch_width; ++offset) {
        float acc = 0.0f;
        for (int64_t expert = 0; expert < selected_count; ++expert) {
            acc += route_weights[expert] * expert_outputs[expert * batch_width + offset];
        }
        hidden_out[offset] = acc;
    }
    session->monolithic_calls += 1;
    session->expert_invocations += selected_count;
    return 0;
}

extern "C" __declspec(dllexport) int ds_router_topk_u16(
    const uint16_t* hidden_u16,
    const uint16_t* router_weights_u16,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t k,
    int64_t dtype_code,
    int64_t* output_expert_ids,
    float* output_weights
) {
    if (
        hidden_u16 == nullptr || router_weights_u16 == nullptr ||
        output_expert_ids == nullptr || output_weights == nullptr
    ) {
        return 1;
    }
    if (hidden_dim <= 0 || num_experts <= 0 || k <= 0 || k > num_experts) {
        return 2;
    }

    std::vector<float> scores(static_cast<size_t>(num_experts), 0.0f);
    for (int64_t expert = 0; expert < num_experts; ++expert) {
        const uint16_t* row = router_weights_u16 + static_cast<size_t>(expert) * static_cast<size_t>(hidden_dim);
        float acc = 0.0f;
        for (int64_t col = 0; col < hidden_dim; ++col) {
            acc += read_u16(hidden_u16[static_cast<size_t>(col)], dtype_code) *
                read_u16(row[static_cast<size_t>(col)], dtype_code);
        }
        scores[static_cast<size_t>(expert)] = acc;
    }

    const float max_score = *std::max_element(scores.begin(), scores.end());
    double denom = 0.0;
    for (float& score : scores) {
        score = std::exp(score - max_score);
        denom += static_cast<double>(score);
    }
    if (denom <= 0.0) {
        return 3;
    }
    for (float& score : scores) {
        score = static_cast<float>(static_cast<double>(score) / denom);
    }

    std::vector<int64_t> ids(static_cast<size_t>(num_experts));
    for (int64_t expert = 0; expert < num_experts; ++expert) {
        ids[static_cast<size_t>(expert)] = expert;
    }
    std::partial_sort(
        ids.begin(),
        ids.begin() + static_cast<std::ptrdiff_t>(k),
        ids.end(),
        [&scores](int64_t left, int64_t right) {
            const float left_score = scores[static_cast<size_t>(left)];
            const float right_score = scores[static_cast<size_t>(right)];
            if (left_score == right_score) {
                return left < right;
            }
            return left_score > right_score;
        }
    );

    double selected_sum = 0.0;
    for (int64_t index = 0; index < k; ++index) {
        selected_sum += static_cast<double>(scores[static_cast<size_t>(ids[static_cast<size_t>(index)])]);
    }
    if (selected_sum <= 0.0) {
        return 4;
    }
    for (int64_t index = 0; index < k; ++index) {
        const int64_t expert_id = ids[static_cast<size_t>(index)];
        output_expert_ids[static_cast<size_t>(index)] = expert_id;
        output_weights[static_cast<size_t>(index)] =
            static_cast<float>(static_cast<double>(scores[static_cast<size_t>(expert_id)]) / selected_sum);
    }
    return 0;
}
