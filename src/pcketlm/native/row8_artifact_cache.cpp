#include <cstdint>
#include <cstdio>
#include <vector>
#include <windows.h>


struct Row8ArtifactTensor {
    std::vector<uint8_t> bytes;
};


struct Row8MappedFile {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    uint8_t* base = nullptr;
    uint64_t nbytes = 0;
};


extern "C" __declspec(dllexport) void* row8_artifact_load_tensor(
    const char* path,
    uint64_t offset,
    uint64_t nbytes
) {
    if (path == nullptr || nbytes == 0 || (nbytes % 2) != 0) {
        return nullptr;
    }
    FILE* file = nullptr;
    if (fopen_s(&file, path, "rb") != 0 || file == nullptr) {
        return nullptr;
    }
    if (_fseeki64(file, static_cast<__int64>(offset), SEEK_SET) != 0) {
        fclose(file);
        return nullptr;
    }
    Row8ArtifactTensor* tensor = new Row8ArtifactTensor();
    tensor->bytes.resize(static_cast<size_t>(nbytes));
    const size_t read_count = fread(tensor->bytes.data(), 1, static_cast<size_t>(nbytes), file);
    fclose(file);
    if (read_count != static_cast<size_t>(nbytes)) {
        delete tensor;
        return nullptr;
    }
    return tensor;
}


extern "C" __declspec(dllexport) void row8_artifact_free_tensor(void* handle) {
    Row8ArtifactTensor* tensor = reinterpret_cast<Row8ArtifactTensor*>(handle);
    delete tensor;
}


extern "C" __declspec(dllexport) const uint16_t* row8_artifact_tensor_data(void* handle) {
    Row8ArtifactTensor* tensor = reinterpret_cast<Row8ArtifactTensor*>(handle);
    if (tensor == nullptr || tensor->bytes.empty()) {
        return nullptr;
    }
    return reinterpret_cast<const uint16_t*>(tensor->bytes.data());
}


extern "C" __declspec(dllexport) uint64_t row8_artifact_tensor_nbytes(void* handle) {
    Row8ArtifactTensor* tensor = reinterpret_cast<Row8ArtifactTensor*>(handle);
    if (tensor == nullptr) {
        return 0;
    }
    return static_cast<uint64_t>(tensor->bytes.size());
}


extern "C" __declspec(dllexport) void* row8_artifact_map_file(const char* path) {
    if (path == nullptr) {
        return nullptr;
    }
    HANDLE file = CreateFileA(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (file == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0) {
        CloseHandle(file);
        return nullptr;
    }
    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        CloseHandle(file);
        return nullptr;
    }
    void* base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (base == nullptr) {
        CloseHandle(mapping);
        CloseHandle(file);
        return nullptr;
    }
    Row8MappedFile* mapped = new Row8MappedFile();
    mapped->file = file;
    mapped->mapping = mapping;
    mapped->base = reinterpret_cast<uint8_t*>(base);
    mapped->nbytes = static_cast<uint64_t>(size.QuadPart);
    return mapped;
}


extern "C" __declspec(dllexport) void row8_artifact_unmap_file(void* handle) {
    Row8MappedFile* mapped = reinterpret_cast<Row8MappedFile*>(handle);
    if (mapped == nullptr) {
        return;
    }
    if (mapped->base != nullptr) {
        UnmapViewOfFile(mapped->base);
    }
    if (mapped->mapping != nullptr) {
        CloseHandle(mapped->mapping);
    }
    if (mapped->file != INVALID_HANDLE_VALUE) {
        CloseHandle(mapped->file);
    }
    delete mapped;
}


extern "C" __declspec(dllexport) const uint16_t* row8_artifact_mapped_data(
    void* handle,
    uint64_t offset,
    uint64_t nbytes
) {
    Row8MappedFile* mapped = reinterpret_cast<Row8MappedFile*>(handle);
    if (mapped == nullptr || mapped->base == nullptr || (offset % 2) != 0 || (nbytes % 2) != 0) {
        return nullptr;
    }
    if (offset > mapped->nbytes || nbytes > mapped->nbytes || offset + nbytes > mapped->nbytes) {
        return nullptr;
    }
    return reinterpret_cast<const uint16_t*>(mapped->base + offset);
}


extern "C" __declspec(dllexport) uint64_t row8_artifact_mapped_nbytes(void* handle) {
    Row8MappedFile* mapped = reinterpret_cast<Row8MappedFile*>(handle);
    if (mapped == nullptr) {
        return 0;
    }
    return mapped->nbytes;
}
