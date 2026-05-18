#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <windows.h>

static HANDLE cached_read_handle(const char* path) {
    static std::mutex mutex;
    static std::unordered_map<std::string, HANDLE> handles;
    const std::string key(path);
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = handles.find(key);
    if (found != handles.end()) {
        return found->second;
    }
    HANDLE handle = CreateFileA(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (handle == INVALID_HANDLE_VALUE) {
        return INVALID_HANDLE_VALUE;
    }
    handles.emplace(key, handle);
    return handle;
}

extern "C" __declspec(dllexport) int native_read_tensor_bytes(
    const char* path,
    uint64_t absolute_offset,
    uint64_t nbytes,
    uint8_t* out
) {
    if (path == nullptr || out == nullptr) {
        return 1;
    }

    HANDLE handle = cached_read_handle(path);
    if (handle == INVALID_HANDLE_VALUE) {
        return 2;
    }

    uint64_t total_read = 0;
    while (total_read < nbytes) {
        const uint64_t remaining = nbytes - total_read;
        const DWORD chunk = remaining > static_cast<uint64_t>(0x7ffff000u)
            ? static_cast<DWORD>(0x7ffff000u)
            : static_cast<DWORD>(remaining);
        OVERLAPPED overlapped = {};
        const uint64_t read_offset = absolute_offset + total_read;
        overlapped.Offset = static_cast<DWORD>(read_offset & 0xffffffffu);
        overlapped.OffsetHigh = static_cast<DWORD>((read_offset >> 32) & 0xffffffffu);
        DWORD got = 0;
        const BOOL ok = ReadFile(handle, out + total_read, chunk, &got, &overlapped);
        total_read += static_cast<uint64_t>(got);
        if (!ok || got != chunk) {
            return 4;
        }
    }

    return 0;
}

extern "C" __declspec(dllexport) int native_copy_tensor_bytes(
    const uint8_t* source,
    uint64_t nbytes,
    uint8_t* out
) {
    if (source == nullptr || out == nullptr) {
        return 1;
    }
    std::memcpy(out, source, static_cast<size_t>(nbytes));
    return 0;
}
