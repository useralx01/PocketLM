#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    std::vector<LayerKvState> layers;
};

static inline bool valid_layer(KvSession* session, int64_t layer) {
    return session != nullptr && layer >= 0 && layer < session->layer_count;
}

extern "C" __declspec(dllexport) void* kv_prefill_init(
    int64_t layer_count,
    int64_t max_seq_len,
    int64_t kv_width
) {
    if (layer_count <= 0 || max_seq_len <= 0 || kv_width <= 0) {
        return nullptr;
    }
    KvSession* session = new KvSession();
    session->layer_count = layer_count;
    session->max_seq_len = max_seq_len;
    session->kv_width = kv_width;
    session->layers.resize(static_cast<size_t>(layer_count));
    return session;
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
