#include <cstdint>
#include <cstdio>
#include <vector>


struct Row8ArtifactTensor {
    std::vector<uint8_t> bytes;
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
