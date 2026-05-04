#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int64_t ROW_BLOCK = 8;

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

static inline float read_u16(uint16_t value, int dtype_code) {
    return dtype_code == 1 ? bf16_to_fp32(value) : fp16_to_fp32(value);
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

extern "C" __declspec(dllexport) int native_pack_u16_rows8(
    const uint16_t* weight,
    uint16_t* packed,
    int64_t rows,
    int64_t cols
) {
    if (weight == nullptr || packed == nullptr) {
        return 1;
    }
    if (rows < 0 || cols < 0) {
        return 2;
    }
    const int64_t block_count = (rows + ROW_BLOCK - 1) / ROW_BLOCK;
    #pragma omp parallel for schedule(static)
    for (int64_t block = 0; block < block_count; ++block) {
        const int64_t row_base = block * ROW_BLOCK;
        uint16_t* out_block = packed + block * cols * ROW_BLOCK;
        for (int64_t col = 0; col < cols; ++col) {
            for (int64_t lane = 0; lane < ROW_BLOCK; ++lane) {
                const int64_t row = row_base + lane;
                out_block[col * ROW_BLOCK + lane] = row < rows ? weight[row * cols + col] : 0;
            }
        }
    }
    return 0;
}

extern "C" __declspec(dllexport) int native_packed_gemv_rows8(
    const uint16_t* hidden,
    const uint16_t* packed,
    float* out,
    int64_t rows,
    int64_t cols,
    int dtype_code
) {
    if (hidden == nullptr || packed == nullptr || out == nullptr) {
        return 1;
    }
    if (rows < 0 || cols < 0 || (dtype_code != 0 && dtype_code != 1)) {
        return 2;
    }
    configure_openmp_threads();
    const int64_t block_count = (rows + ROW_BLOCK - 1) / ROW_BLOCK;
    #pragma omp parallel for schedule(static)
    for (int64_t block = 0; block < block_count; ++block) {
        const int64_t row_base = block * ROW_BLOCK;
        const uint16_t* weight_block = packed + block * cols * ROW_BLOCK;
        __m256 acc = _mm256_setzero_ps();
        int64_t col = 0;
        for (; col < cols; ++col) {
            const __m256 w = load_u16_as_ps(weight_block + col * ROW_BLOCK, dtype_code);
            const __m256 h = _mm256_set1_ps(read_u16(hidden[col], dtype_code));
            acc = _mm256_fmadd_ps(h, w, acc);
        }
        float sums[ROW_BLOCK];
        _mm256_storeu_ps(sums, acc);
        for (int64_t lane = 0; lane < ROW_BLOCK; ++lane) {
            const int64_t row = row_base + lane;
            if (row < rows) {
                out[row] = sums[lane];
            }
        }
    }
    return 0;
}

