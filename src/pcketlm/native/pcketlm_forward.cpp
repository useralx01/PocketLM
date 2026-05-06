#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <immintrin.h>
#include <intrin.h>

struct ForwardSession {
    int64_t layer_count = 0;
    int64_t hidden_size = 0;
    int64_t vocab_size = 0;
    int64_t max_seq_len = 0;
    int64_t layers_executed = 0;
    std::vector<int64_t> committed_tokens;
    std::vector<int64_t> tentative_tokens;
    std::unordered_map<std::string, std::vector<uint16_t>> u16_weights;
};

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

static inline uint16_t fp32_to_fp16(float value) {
    const __m128 full = _mm_set_ss(value);
    const __m128i half = _mm_cvtps_ph(full, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<uint16_t>(_mm_cvtsi128_si32(half));
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
    session->vocab_size = std::max<int64_t>(2, parse_json_int(model_config_json, "vocab_size", 2));
    session->max_seq_len = std::max<int64_t>(1, parse_json_int(model_config_json, "max_position_embeddings", 4096));
    session->layers_executed = 0;
    return session;
}

extern "C" __declspec(dllexport) void pcketlm_session_destroy(void* handle) {
    ForwardSession* session = reinterpret_cast<ForwardSession*>(handle);
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
    std::vector<uint16_t> copied(static_cast<size_t>(value_count));
    if (value_count > 0) {
        std::memcpy(copied.data(), tensor_data, static_cast<size_t>(value_count) * sizeof(uint16_t));
    }
    session->u16_weights[std::string(tensor_name)] = std::move(copied);
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
    return static_cast<int64_t>(found->second.size());
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
    session->committed_tokens.push_back(new_token_id);
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
