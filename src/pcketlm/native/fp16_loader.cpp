#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" __declspec(dllexport) int native_read_tensor_bytes(
    const char* path,
    uint64_t absolute_offset,
    uint64_t nbytes,
    uint8_t* out
) {
    if (path == nullptr || out == nullptr) {
        return 1;
    }

    FILE* handle = nullptr;
    const errno_t open_error = fopen_s(&handle, path, "rb");
    if (open_error != 0 || handle == nullptr) {
        return 2;
    }

    if (_fseeki64(handle, static_cast<__int64>(absolute_offset), SEEK_SET) != 0) {
        fclose(handle);
        return 3;
    }

    uint64_t total_read = 0;
    while (total_read < nbytes) {
        const uint64_t remaining = nbytes - total_read;
        const size_t chunk = remaining > static_cast<uint64_t>(SIZE_MAX)
            ? SIZE_MAX
            : static_cast<size_t>(remaining);
        const size_t got = fread(out + total_read, 1, chunk, handle);
        total_read += static_cast<uint64_t>(got);
        if (got != chunk) {
            fclose(handle);
            return 4;
        }
    }

    fclose(handle);
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
