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

using DsTensorCallback = int (*)(
    int64_t layer_idx,
    int64_t layer_kind,
    int64_t weight_role,
    const void** out_ptr,
    int64_t* out_nbytes
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
    std::vector<DsLayerRegistration> registrations;
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
