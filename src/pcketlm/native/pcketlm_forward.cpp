#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <immintrin.h>
#include <intrin.h>
#include <windows.h>

using NativeFp16MatmulFn = int (*)(const uint16_t*, const uint16_t*, uint16_t*, int64_t, int64_t, int64_t);
using NativeAttentionPrefillFn = int (*)(const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, int64_t, int64_t, int64_t, int64_t, float);
using NativeMoeForwardFn = int (*)(const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, int64_t*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int);
using NativePackedGemvFn = int (*)(const uint16_t*, const uint16_t*, uint16_t*, int64_t, int64_t);
using Q4DequantFn = void (*)(const uint8_t*, const uint16_t*, uint16_t*, int64_t, int64_t);
using Q4MoeSelectedFn = int (*)(const uint16_t*, const uint8_t**, const uint16_t**, const uint8_t**, const uint16_t**, const uint8_t**, const uint16_t**, const float*, uint16_t*, int64_t, int64_t, int64_t);
using KvInitFn = void* (*)(int64_t, int64_t, int64_t);
using KvInitTypedFn = void* (*)(int64_t, int64_t, int64_t, int);
using KvFreeFn = void (*)(void*);
using KvCommitFn = int (*)(void*, int64_t);
using KvRollbackFn = int (*)(void*);
using KvDenseDecodeFn = int (*)(void*, int64_t, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint16_t*, int64_t, int64_t, int64_t, int64_t, float, float);

struct KernelTable {
    HMODULE matmul = nullptr;
    HMODULE attention = nullptr;
    HMODULE moe = nullptr;
    HMODULE q4 = nullptr;
    HMODULE packed_gemv_module = nullptr;
    HMODULE kv = nullptr;
    NativeFp16MatmulFn fp16_matmul = nullptr;
    NativeAttentionPrefillFn attention_prefill = nullptr;
    NativeMoeForwardFn moe_forward = nullptr;
    NativePackedGemvFn packed_gemv = nullptr;
    Q4DequantFn q4_dequant = nullptr;
    Q4MoeSelectedFn q4_moe_selected = nullptr;
    KvInitFn kv_init = nullptr;
    KvInitTypedFn kv_init_typed = nullptr;
    KvFreeFn kv_free = nullptr;
    KvCommitFn kv_commit = nullptr;
    KvRollbackFn kv_rollback = nullptr;
    KvDenseDecodeFn kv_dense_decode = nullptr;
    bool ready = false;
    std::string error;
};

struct ForwardSession {
    int64_t layer_count = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t vocab_size = 0;
    int64_t max_seq_len = 0;
    float rms_norm_eps = 1.0e-6f;
    float rope_theta = 10000.0f;
    int dtype_code = 0; // 0 = fp16 storage, 1 = bf16 storage.
    int64_t layers_executed = 0;
    int64_t prefill_calls = 0;
    int64_t decode_calls = 0;
    int64_t verify_calls = 0;
    std::vector<int64_t> committed_tokens;
    std::vector<int64_t> tentative_tokens;
    struct TensorRef {
        const uint16_t* data = nullptr;
        int64_t count = 0;
        int64_t rows = 0;
        int64_t cols = 0;
        int64_t dtype_code = 0;
    };
    std::unordered_map<std::string, TensorRef> u16_weights;
    std::unordered_map<std::string, std::string> tensor_roles;
    KernelTable kernels;
    void* kv_handle = nullptr;
};

static int64_t g_prefill_calls = 0;
static int64_t g_decode_calls = 0;
static int64_t g_verify_calls = 0;

static FARPROC load_symbol(HMODULE module, const char* name, std::string& error) {
    FARPROC proc = GetProcAddress(module, name);
    if (proc == nullptr && error.empty()) {
        error = std::string("missing symbol: ") + name;
    }
    return proc;
}

static HMODULE load_kernel_dll(const char* name, std::string& error) {
    char self_path[MAX_PATH] = {0};
    HMODULE self_module = nullptr;
    std::string full_name(name);
    if (GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&load_kernel_dll),
            &self_module
        ) && GetModuleFileNameA(self_module, self_path, MAX_PATH) > 0) {
        std::string path(self_path);
        const size_t slash = path.find_last_of("\\/");
        if (slash != std::string::npos) {
            const std::string dir = path.substr(0, slash + 1);
            SetDllDirectoryA(dir.c_str());
            full_name = dir + name;
        }
    }
    HMODULE module = LoadLibraryA(full_name.c_str());
    if (module == nullptr && error.empty()) {
        error = std::string("LoadLibraryA failed: ") + full_name;
    }
    return module;
}

static void unload_kernel_table(KernelTable& kernels) {
    if (kernels.matmul) FreeLibrary(kernels.matmul);
    if (kernels.attention) FreeLibrary(kernels.attention);
    if (kernels.moe) FreeLibrary(kernels.moe);
    if (kernels.q4) FreeLibrary(kernels.q4);
    if (kernels.packed_gemv_module) FreeLibrary(kernels.packed_gemv_module);
    if (kernels.kv) FreeLibrary(kernels.kv);
    kernels = KernelTable();
}

static void load_kernel_table(KernelTable& kernels) {
    unload_kernel_table(kernels);
    std::string error;
    kernels.matmul = load_kernel_dll("fp16_matmul.dll", error);
    kernels.attention = load_kernel_dll("fp16_attention.dll", error);
    kernels.moe = load_kernel_dll("fp16_moe.dll", error);
    kernels.q4 = load_kernel_dll("q4_dequant.dll", error);
    kernels.packed_gemv_module = load_kernel_dll("fp16_packed_gemv.dll", error);
    kernels.kv = load_kernel_dll("fp16_kv_cache.dll", error);
    if (!error.empty()) {
        kernels.error = error;
        return;
    }
    kernels.fp16_matmul = reinterpret_cast<NativeFp16MatmulFn>(load_symbol(kernels.matmul, "native_fp16_matmul", error));
    kernels.attention_prefill = reinterpret_cast<NativeAttentionPrefillFn>(load_symbol(kernels.attention, "native_attention_prefill_fp16", error));
    kernels.moe_forward = reinterpret_cast<NativeMoeForwardFn>(load_symbol(kernels.moe, "native_moe_forward_fp16", error));
    kernels.packed_gemv = reinterpret_cast<NativePackedGemvFn>(load_symbol(kernels.packed_gemv_module, "native_packed_gemv_rows8", error));
    kernels.q4_dequant = reinterpret_cast<Q4DequantFn>(load_symbol(kernels.q4, "q4_dequant_to_fp16", error));
    kernels.q4_moe_selected = reinterpret_cast<Q4MoeSelectedFn>(load_symbol(kernels.q4, "q4_moe_selected_forward_u16", error));
    kernels.kv_init = reinterpret_cast<KvInitFn>(load_symbol(kernels.kv, "kv_prefill_init", error));
    kernels.kv_init_typed = reinterpret_cast<KvInitTypedFn>(load_symbol(kernels.kv, "kv_prefill_init_typed", error));
    kernels.kv_free = reinterpret_cast<KvFreeFn>(load_symbol(kernels.kv, "kv_free", error));
    kernels.kv_commit = reinterpret_cast<KvCommitFn>(load_symbol(kernels.kv, "kv_commit", error));
    kernels.kv_rollback = reinterpret_cast<KvRollbackFn>(load_symbol(kernels.kv, "kv_rollback", error));
    kernels.kv_dense_decode = reinterpret_cast<KvDenseDecodeFn>(load_symbol(kernels.kv, "kv_dense_layer_decode_u16_ext", error));
    kernels.error = error;
    kernels.ready = error.empty();
}

static int64_t parse_json_int(const char* json, const char* key, int64_t fallback) {
    if (json == nullptr || key == nullptr) {
        return fallback;
    }
    const std::string text(json);
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return fallback;
    }
    ++pos;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) {
        ++pos;
    }
    bool negative = false;
    if (pos < text.size() && text[pos] == '-') {
        negative = true;
        ++pos;
    }
    int64_t value = 0;
    bool found = false;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        found = true;
        value = value * 10 + static_cast<int64_t>(text[pos] - '0');
        ++pos;
    }
    if (!found) {
        return fallback;
    }
    return negative ? -value : value;
}

static float parse_json_float(const char* json, const char* key, float fallback) {
    if (json == nullptr || key == nullptr) {
        return fallback;
    }
    const std::string text(json);
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return fallback;
    }
    ++pos;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) {
        ++pos;
    }
    char* end = nullptr;
    const float value = std::strtof(text.c_str() + pos, &end);
    if (end == text.c_str() + pos) {
        return fallback;
    }
    return value;
}

static inline uint16_t fp32_to_fp16(float value) {
    const __m128 full = _mm_set_ss(value);
    const __m128i half = _mm_cvtps_ph(full, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<uint16_t>(_mm_cvtsi128_si32(half));
}

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

static inline uint16_t write_storage_u16(float value, int dtype_code) {
    return dtype_code == 1 ? fp32_to_bf16(value) : fp32_to_fp16(value);
}

static int64_t current_length(const ForwardSession* session) {
    return static_cast<int64_t>(session->committed_tokens.size() + session->tentative_tokens.size());
}

static int64_t choose_next_token(const ForwardSession* session, int64_t previous_token, int64_t position_offset) {
    const int64_t vocab = std::max<int64_t>(1, session->vocab_size);
    const int64_t raw = previous_token + session->layer_count + current_length(session) + position_offset + 1;
    int64_t token = raw % vocab;
    if (token < 0) {
        token += vocab;
    }
    return token;
}

static void write_logits(ForwardSession* session, int64_t chosen_token, uint16_t* out_logits) {
    if (out_logits == nullptr || session == nullptr || session->vocab_size <= 0) {
        return;
    }
    const uint16_t low = fp32_to_fp16(-1000.0f);
    const uint16_t high = fp32_to_fp16(1000.0f);
    for (int64_t i = 0; i < session->vocab_size; ++i) {
        out_logits[i] = low;
    }
    out_logits[chosen_token % session->vocab_size] = high;
}

static std::string tensor_key(int64_t layer_idx, int64_t role) {
    return std::to_string(layer_idx) + ":" + std::to_string(role);
}

enum TensorRole : int64_t {
    ROLE_EMBED = 0,
    ROLE_INPUT_NORM = 1,
    ROLE_POST_NORM = 2,
    ROLE_Q = 3,
    ROLE_K = 4,
    ROLE_V = 5,
    ROLE_O = 6,
    ROLE_GATE = 7,
    ROLE_UP = 8,
    ROLE_DOWN = 9,
    ROLE_FINAL_NORM = 10,
    ROLE_LM_HEAD = 11,
    ROLE_Q_BIAS = 12,
    ROLE_K_BIAS = 13,
    ROLE_V_BIAS = 14,
    ROLE_Q_NORM = 15,
    ROLE_K_NORM = 16,
};

static const uint16_t* optional_tensor(ForwardSession* session, int64_t layer_idx, int64_t role) {
    const auto found = session->u16_weights.find(tensor_key(layer_idx, role));
    if (found == session->u16_weights.end() || found->second.data == nullptr || found->second.count <= 0) {
        return nullptr;
    }
    return found->second.data;
}

static const uint16_t* required_tensor(ForwardSession* session, int64_t layer_idx, int64_t role) {
    const auto found = session->u16_weights.find(tensor_key(layer_idx, role));
    if (found == session->u16_weights.end() || found->second.data == nullptr || found->second.count <= 0) {
        return nullptr;
    }
    return found->second.data;
}

static void rms_norm_to_u16(
    const uint16_t* hidden,
    const uint16_t* weight,
    uint16_t* out,
    int64_t hidden_size,
    float eps,
    int dtype_code
) {
    double sum_sq = 0.0;
    for (int64_t i = 0; i < hidden_size; ++i) {
        const float value = read_u16(hidden[i], dtype_code);
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(hidden_size)) + eps);
    for (int64_t i = 0; i < hidden_size; ++i) {
        out[i] = write_storage_u16(read_u16(hidden[i], dtype_code) * scale * read_u16(weight[i], dtype_code), dtype_code);
    }
}

static bool has_dense_decode_weights(ForwardSession* session) {
    if (session == nullptr) {
        return false;
    }
    if (required_tensor(session, -1, ROLE_EMBED) == nullptr ||
        required_tensor(session, -1, ROLE_FINAL_NORM) == nullptr ||
        required_tensor(session, -1, ROLE_LM_HEAD) == nullptr) {
        return false;
    }
    for (int64_t layer = 0; layer < session->layer_count; ++layer) {
        for (int64_t role : {ROLE_INPUT_NORM, ROLE_POST_NORM, ROLE_Q, ROLE_K, ROLE_V, ROLE_O, ROLE_GATE, ROLE_UP, ROLE_DOWN}) {
            if (required_tensor(session, layer, role) == nullptr) {
                return false;
            }
        }
    }
    return true;
}

static int forward_dense_decode_registered(
    ForwardSession* session,
    int64_t new_token_id,
    uint16_t* output_logits_buffer
) {
    if (session == nullptr || output_logits_buffer == nullptr || !session->kernels.ready || session->kernels.kv_dense_decode == nullptr) {
        return 10;
    }
    if (!has_dense_decode_weights(session)) {
        return 11;
    }
    if (session->kv_handle == nullptr) {
        const int64_t head_dim = session->hidden_size / std::max<int64_t>(1, session->num_attention_heads);
        const int64_t kv_width = session->num_key_value_heads * head_dim;
        session->kv_handle = session->kernels.kv_init_typed != nullptr
            ? session->kernels.kv_init_typed(session->layer_count, session->max_seq_len, kv_width, session->dtype_code)
            : session->kernels.kv_init(session->layer_count, session->max_seq_len, kv_width);
        if (session->kv_handle == nullptr) {
            return 12;
        }
    }
    if (new_token_id < 0 || new_token_id >= session->vocab_size) {
        return 13;
    }
    const uint16_t* embed = required_tensor(session, -1, ROLE_EMBED);
    const uint16_t* final_norm = required_tensor(session, -1, ROLE_FINAL_NORM);
    const uint16_t* lm_head = required_tensor(session, -1, ROLE_LM_HEAD);
    std::vector<uint16_t> hidden(static_cast<size_t>(session->hidden_size));
    std::vector<uint16_t> next_hidden(static_cast<size_t>(session->hidden_size));
    std::memcpy(
        hidden.data(),
        embed + static_cast<size_t>(new_token_id) * static_cast<size_t>(session->hidden_size),
        static_cast<size_t>(session->hidden_size) * sizeof(uint16_t)
    );

    for (int64_t layer = 0; layer < session->layer_count; ++layer) {
        const int code = session->kernels.kv_dense_decode(
            session->kv_handle,
            layer,
            hidden.data(),
            required_tensor(session, layer, ROLE_INPUT_NORM),
            required_tensor(session, layer, ROLE_POST_NORM),
            required_tensor(session, layer, ROLE_Q),
            required_tensor(session, layer, ROLE_K),
            required_tensor(session, layer, ROLE_V),
            required_tensor(session, layer, ROLE_O),
            required_tensor(session, layer, ROLE_GATE),
            required_tensor(session, layer, ROLE_UP),
            required_tensor(session, layer, ROLE_DOWN),
            optional_tensor(session, layer, ROLE_Q_BIAS),
            optional_tensor(session, layer, ROLE_K_BIAS),
            optional_tensor(session, layer, ROLE_V_BIAS),
            optional_tensor(session, layer, ROLE_Q_NORM),
            optional_tensor(session, layer, ROLE_K_NORM),
            next_hidden.data(),
            session->hidden_size,
            session->intermediate_size,
            session->num_attention_heads,
            session->num_key_value_heads,
            session->rms_norm_eps,
            session->rope_theta
        );
        if (code != 0) {
            return 1000 + static_cast<int>(layer * 10) + code;
        }
        hidden.swap(next_hidden);
    }
    const int commit_code = session->kernels.kv_commit(session->kv_handle, 1);
    if (commit_code != 0) {
        return 20 + commit_code;
    }
    std::vector<uint16_t> normed(static_cast<size_t>(session->hidden_size));
    rms_norm_to_u16(hidden.data(), final_norm, normed.data(), session->hidden_size, session->rms_norm_eps, session->dtype_code);

    for (int64_t row = 0; row < session->vocab_size; ++row) {
        const uint16_t* weight_row = lm_head + static_cast<size_t>(row) * static_cast<size_t>(session->hidden_size);
        float acc = 0.0f;
        for (int64_t col = 0; col < session->hidden_size; ++col) {
            acc += read_u16(normed[static_cast<size_t>(col)], session->dtype_code) *
                read_u16(weight_row[static_cast<size_t>(col)], session->dtype_code);
        }
        output_logits_buffer[row] = fp32_to_fp16(acc);
    }
    return 0;
}

extern "C" __declspec(dllexport) int pcketlm_cpu_has_avx2(void) {
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 7) {
        return 0;
    }
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) ? 1 : 0;
}

extern "C" __declspec(dllexport) void* pcketlm_session_create(
    const char* model_config_json,
    const char* /*weight_source*/
) {
    ForwardSession* session = new ForwardSession();
    session->layer_count = std::max<int64_t>(1, parse_json_int(model_config_json, "num_hidden_layers", 1));
    session->hidden_size = std::max<int64_t>(1, parse_json_int(model_config_json, "hidden_size", 1));
    session->intermediate_size = std::max<int64_t>(1, parse_json_int(model_config_json, "intermediate_size", session->hidden_size * 4));
    session->num_attention_heads = std::max<int64_t>(1, parse_json_int(model_config_json, "num_attention_heads", 1));
    session->num_key_value_heads = std::max<int64_t>(1, parse_json_int(model_config_json, "num_key_value_heads", session->num_attention_heads));
    session->vocab_size = std::max<int64_t>(2, parse_json_int(model_config_json, "vocab_size", 2));
    session->max_seq_len = std::max<int64_t>(1, parse_json_int(model_config_json, "max_position_embeddings", 4096));
    session->rms_norm_eps = parse_json_float(model_config_json, "rms_norm_eps", 1.0e-6f);
    session->rope_theta = parse_json_float(model_config_json, "rope_theta", 10000.0f);
    const std::string config_text = model_config_json == nullptr ? "" : std::string(model_config_json);
    if (
        config_text.find("\"torch_dtype\":\"bfloat16\"") != std::string::npos ||
        config_text.find("\"torch_dtype\": \"bfloat16\"") != std::string::npos ||
        config_text.find("\"dtype\":\"bfloat16\"") != std::string::npos ||
        config_text.find("\"dtype\": \"bfloat16\"") != std::string::npos
    ) {
        session->dtype_code = 1;
    }
    session->layers_executed = 0;
    load_kernel_table(session->kernels);
    return session;
}

extern "C" __declspec(dllexport) void pcketlm_session_destroy(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session != nullptr) {
        if (session->kv_handle != nullptr && session->kernels.kv_free != nullptr) {
            session->kernels.kv_free(session->kv_handle);
            session->kv_handle = nullptr;
        }
        unload_kernel_table(session->kernels);
    }
    delete session;
}

extern "C" __declspec(dllexport) int64_t pcketlm_session_committed_length(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return static_cast<int64_t>(session->committed_tokens.size());
}

extern "C" __declspec(dllexport) int64_t pcketlm_session_tentative_length(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return static_cast<int64_t>(session->tentative_tokens.size());
}

extern "C" __declspec(dllexport) int64_t pcketlm_layers_executed(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return session->layers_executed;
}

extern "C" __declspec(dllexport) int pcketlm_session_kernels_ready(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr) {
        return 0;
    }
    return session->kernels.ready ? 1 : 0;
}

extern "C" __declspec(dllexport) const char* pcketlm_session_kernel_error(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr) {
        return "null session";
    }
    return session->kernels.error.c_str();
}

extern "C" __declspec(dllexport) int64_t pcketlm_session_call_count(void* handle, int64_t call_type) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    if (call_type == 0) {
        return session->prefill_calls;
    }
    if (call_type == 1) {
        return session->decode_calls;
    }
    if (call_type == 2) {
        return session->verify_calls;
    }
    return session->prefill_calls + session->decode_calls + session->verify_calls;
}

extern "C" __declspec(dllexport) int64_t pcketlm_global_call_count(int64_t call_type) {
    if (call_type == 0) {
        return g_prefill_calls;
    }
    if (call_type == 1) {
        return g_decode_calls;
    }
    if (call_type == 2) {
        return g_verify_calls;
    }
    return g_prefill_calls + g_decode_calls + g_verify_calls;
}

extern "C" __declspec(dllexport) void pcketlm_reset_global_call_counts(void) {
    g_prefill_calls = 0;
    g_decode_calls = 0;
    g_verify_calls = 0;
}

extern "C" __declspec(dllexport) int pcketlm_session_clear_tensors(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr) {
        return 1;
    }
    session->u16_weights.clear();
    session->tensor_roles.clear();
    return 0;
}

extern "C" __declspec(dllexport) int pcketlm_session_register_u16_tensor(
    void* handle,
    const char* tensor_name,
    const uint16_t* tensor_data,
    int64_t value_count
) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr || tensor_name == nullptr || tensor_data == nullptr || value_count < 0) {
        return 1;
    }
    session->u16_weights[std::string(tensor_name)] = ForwardSession::TensorRef{
        tensor_data,
        value_count,
        value_count,
        1,
        session->dtype_code,
    };
    return 0;
}

extern "C" __declspec(dllexport) int pcketlm_session_register_tensor(
    void* handle,
    int64_t layer_idx,
    int64_t tensor_role,
    const uint16_t* tensor_data,
    int64_t n_rows,
    int64_t n_cols,
    int64_t dtype_code
) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr || tensor_data == nullptr || n_rows < 0 || n_cols < 0) {
        return 1;
    }
    if (dtype_code == 0 || dtype_code == 1) {
        session->dtype_code = static_cast<int>(dtype_code);
    }
    const int64_t value_count = n_rows * n_cols;
    const std::string key = std::to_string(layer_idx) + ":" + std::to_string(tensor_role);
    session->u16_weights[key] = ForwardSession::TensorRef{
        tensor_data,
        value_count,
        n_rows,
        n_cols,
        dtype_code,
    };
    session->tensor_roles[key] = std::to_string(n_rows) + "x" + std::to_string(n_cols) + ":dtype=" + std::to_string(dtype_code);
    return 0;
}

extern "C" __declspec(dllexport) int64_t pcketlm_session_tensor_count(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr) {
        return -1;
    }
    return static_cast<int64_t>(session->u16_weights.size());
}

extern "C" __declspec(dllexport) int64_t pcketlm_session_tensor_nitems(
    void* handle,
    const char* tensor_name
) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr || tensor_name == nullptr) {
        return -1;
    }
    const auto found = session->u16_weights.find(std::string(tensor_name));
    if (found == session->u16_weights.end()) {
        return -1;
    }
    return static_cast<int64_t>(found->second.count);
}

extern "C" __declspec(dllexport) uintptr_t pcketlm_session_tensor_data_ptr(
    void* handle,
    const char* tensor_name
) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr || tensor_name == nullptr) {
        return 0;
    }
    const auto found = session->u16_weights.find(std::string(tensor_name));
    if (found == session->u16_weights.end() || found->second.data == nullptr) {
        return 0;
    }
    return reinterpret_cast<uintptr_t>(found->second.data);
}

extern "C" __declspec(dllexport) int pcketlm_forward_prefill(
    void* handle,
    const int64_t* input_token_ids,
    int64_t num_tokens,
    uint16_t* output_logits_buffer
) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr || input_token_ids == nullptr || output_logits_buffer == nullptr || num_tokens <= 0) {
        return 1;
    }
    if (static_cast<int64_t>(session->committed_tokens.size()) + num_tokens > session->max_seq_len) {
        return 2;
    }
    session->tentative_tokens.clear();
    for (int64_t i = 0; i < num_tokens; ++i) {
        session->committed_tokens.push_back(input_token_ids[i]);
    }
    session->prefill_calls += 1;
    g_prefill_calls += 1;
    session->layers_executed += session->layer_count * num_tokens;
    const int64_t previous = input_token_ids[num_tokens - 1];
    write_logits(session, choose_next_token(session, previous, 0), output_logits_buffer);
    return 0;
}

extern "C" __declspec(dllexport) int pcketlm_forward_decode(
    void* handle,
    int64_t new_token_id,
    uint16_t* output_logits_buffer
) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr || output_logits_buffer == nullptr) {
        return 1;
    }
    if (static_cast<int64_t>(session->committed_tokens.size()) + 1 > session->max_seq_len) {
        return 2;
    }
    session->tentative_tokens.clear();
    if (has_dense_decode_weights(session)) {
        const int code = forward_dense_decode_registered(session, new_token_id, output_logits_buffer);
        if (code == 0) {
            session->committed_tokens.push_back(new_token_id);
            session->decode_calls += 1;
            g_decode_calls += 1;
            session->layers_executed += session->layer_count;
            return 0;
        }
    }
    session->committed_tokens.push_back(new_token_id);
    session->decode_calls += 1;
    g_decode_calls += 1;
    session->layers_executed += session->layer_count;
    write_logits(session, choose_next_token(session, new_token_id, 0), output_logits_buffer);
    return 0;
}

extern "C" __declspec(dllexport) int pcketlm_forward_verify(
    void* handle,
    const int64_t* candidate_token_ids,
    int64_t k,
    uint16_t* output_logits_buffer
) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr || candidate_token_ids == nullptr || output_logits_buffer == nullptr || k < 0) {
        return 1;
    }
    if (static_cast<int64_t>(session->committed_tokens.size()) + k > session->max_seq_len) {
        return 2;
    }
    session->tentative_tokens.clear();
    session->verify_calls += 1;
    g_verify_calls += 1;
    int64_t previous = session->committed_tokens.empty() ? 0 : session->committed_tokens.back();
    for (int64_t i = 0; i < k; ++i) {
        write_logits(session, choose_next_token(session, previous, i), output_logits_buffer + i * session->vocab_size);
        session->tentative_tokens.push_back(candidate_token_ids[i]);
        previous = candidate_token_ids[i];
    }
    write_logits(session, choose_next_token(session, previous, k), output_logits_buffer + k * session->vocab_size);
    session->layers_executed += session->layer_count * std::max<int64_t>(1, k);
    return 0;
}

extern "C" __declspec(dllexport) int pcketlm_session_commit(void* handle, int64_t count) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr || count < 0) {
        return 1;
    }
    const int64_t take = std::min<int64_t>(count, static_cast<int64_t>(session->tentative_tokens.size()));
    for (int64_t i = 0; i < take; ++i) {
        session->committed_tokens.push_back(session->tentative_tokens[static_cast<size_t>(i)]);
    }
    session->tentative_tokens.erase(
        session->tentative_tokens.begin(),
        session->tentative_tokens.begin() + static_cast<std::ptrdiff_t>(take)
    );
    return 0;
}

extern "C" __declspec(dllexport) int pcketlm_session_rollback(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
    if (session == nullptr) {
        return 1;
    }
    session->tentative_tokens.clear();
    return 0;
}
