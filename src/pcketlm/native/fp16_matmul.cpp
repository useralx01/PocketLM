#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <vector>

#include "cblas.h"

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void openblas_set_num_threads(int num_threads);

static inline float fp16_to_fp32(uint16_t value) {
    const __m128i half = _mm_cvtsi32_si128(static_cast<int>(value));
    const __m128 full = _mm_cvtph_ps(half);
    return _mm_cvtss_f32(full);
}

static inline uint16_t fp32_to_fp16(float value) {
    const __m128 full = _mm_set_ss(value);
    const __m128i half = _mm_cvtps_ph(full, 0);
    return static_cast<uint16_t>(_mm_cvtsi128_si32(half));
}

static inline float bf16_to_fp32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
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

static inline float read_u16(uint16_t value, int dtype_code) {
    return dtype_code == 1 ? bf16_to_fp32(value) : fp16_to_fp32(value);
}

static inline float horizontal_sum_ps(__m256 values) {
    const __m128 low = _mm256_castps256_ps128(values);
    const __m128 high = _mm256_extractf128_ps(values, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline float dot_u16_u16(const uint16_t* left, const uint16_t* right, int64_t count, int dtype_code) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int64_t index = 0;
    for (; index + 32 <= count; index += 32) {
        acc0 = _mm256_fmadd_ps(load_u16_as_ps(left + index, dtype_code), load_u16_as_ps(right + index, dtype_code), acc0);
        acc1 = _mm256_fmadd_ps(load_u16_as_ps(left + index + 8, dtype_code), load_u16_as_ps(right + index + 8, dtype_code), acc1);
        acc2 = _mm256_fmadd_ps(load_u16_as_ps(left + index + 16, dtype_code), load_u16_as_ps(right + index + 16, dtype_code), acc2);
        acc3 = _mm256_fmadd_ps(load_u16_as_ps(left + index + 24, dtype_code), load_u16_as_ps(right + index + 24, dtype_code), acc3);
    }
    for (; index + 8 <= count; index += 8) {
        acc0 = _mm256_fmadd_ps(load_u16_as_ps(left + index, dtype_code), load_u16_as_ps(right + index, dtype_code), acc0);
    }
    float acc = horizontal_sum_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; index < count; ++index) {
        acc += read_u16(left[index], dtype_code) * read_u16(right[index], dtype_code);
    }
    return acc;
}

static void insert_topk(float value, int64_t token_id, float* top_logits, int64_t* top_ids, int64_t top_k) {
    int64_t slot = -1;
    float worst = INFINITY;
    for (int64_t index = 0; index < top_k; ++index) {
        if (top_ids[index] < 0) {
            slot = index;
            break;
        }
        if (top_logits[index] < worst) {
            worst = top_logits[index];
            slot = index;
        }
    }
    if (slot >= 0 && (top_ids[slot] < 0 || value > top_logits[slot])) {
        top_logits[slot] = value;
        top_ids[slot] = token_id;
    }
}

extern "C" __declspec(dllexport) int native_lm_head_topk_u16(
    const uint16_t* hidden,
    const uint16_t* weight,
    int64_t* out_ids,
    float* out_logits,
    int64_t rows,
    int64_t hidden_size,
    int64_t top_k,
    int dtype_code,
    int64_t token_offset
) {
    if (hidden == nullptr || weight == nullptr || out_ids == nullptr || out_logits == nullptr) {
        return 1;
    }
    if (rows <= 0 || hidden_size <= 0 || top_k <= 0 || (dtype_code != 0 && dtype_code != 1)) {
        return 2;
    }
    #ifdef _OPENMP
    const char* requested_threads = std::getenv("PCKETLM_NATIVE_THREADS");
    if (requested_threads != nullptr) {
        const int parsed = std::atoi(requested_threads);
        if (parsed > 0) {
            omp_set_num_threads(parsed);
        }
    }
    #endif
    for (int64_t index = 0; index < top_k; ++index) {
        out_ids[index] = -1;
        out_logits[index] = -INFINITY;
    }

    #pragma omp parallel
    {
        float* local_logits = new float[static_cast<size_t>(top_k)];
        int64_t* local_ids = new int64_t[static_cast<size_t>(top_k)];
        for (int64_t index = 0; index < top_k; ++index) {
            local_ids[index] = -1;
            local_logits[index] = -INFINITY;
        }
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            const float logit = dot_u16_u16(hidden, weight + row * hidden_size, hidden_size, dtype_code);
            insert_topk(logit, token_offset + row, local_logits, local_ids, top_k);
        }
        #pragma omp critical
        {
            for (int64_t index = 0; index < top_k; ++index) {
                if (local_ids[index] >= 0) {
                    insert_topk(local_logits[index], local_ids[index], out_logits, out_ids, top_k);
                }
            }
        }
        delete[] local_logits;
        delete[] local_ids;
    }
    return 0;
}

static int handwritten_fp16_matmul(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* c,
    int64_t m,
    int64_t n,
    int64_t k
) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        return 1;
    }
    if (m < 0 || n < 0 || k < 0) {
        return 2;
    }

    #ifdef _OPENMP
    const char* requested_threads = std::getenv("PCKETLM_NATIVE_THREADS");
    if (requested_threads != nullptr) {
        const int parsed = std::atoi(requested_threads);
        if (parsed > 0) {
            omp_set_num_threads(parsed);
        }
    }
    #endif

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < m; ++row) {
        int64_t col = 0;
        for (; col + 32 <= n; col += 32) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            for (int64_t inner = 0; inner < k; ++inner) {
                const float av = fp16_to_fp32(a[row * k + inner]);
                const __m256 avec = _mm256_set1_ps(av);
                const uint16_t* b_base = b + inner * n + col;
                const __m128i b_half0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_base));
                const __m128i b_half1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_base + 8));
                const __m128i b_half2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_base + 16));
                const __m128i b_half3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_base + 24));
                acc0 = _mm256_fmadd_ps(avec, _mm256_cvtph_ps(b_half0), acc0);
                acc1 = _mm256_fmadd_ps(avec, _mm256_cvtph_ps(b_half1), acc1);
                acc2 = _mm256_fmadd_ps(avec, _mm256_cvtph_ps(b_half2), acc2);
                acc3 = _mm256_fmadd_ps(avec, _mm256_cvtph_ps(b_half3), acc3);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + row * n + col), _mm256_cvtps_ph(acc0, 0));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + row * n + col + 8), _mm256_cvtps_ph(acc1, 0));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + row * n + col + 16), _mm256_cvtps_ph(acc2, 0));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + row * n + col + 24), _mm256_cvtps_ph(acc3, 0));
        }
        for (; col + 16 <= n; col += 16) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (int64_t inner = 0; inner < k; ++inner) {
                const float av = fp16_to_fp32(a[row * k + inner]);
                const __m256 avec = _mm256_set1_ps(av);
                const uint16_t* b_base = b + inner * n + col;
                const __m128i b_half0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_base));
                const __m128i b_half1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_base + 8));
                acc0 = _mm256_fmadd_ps(avec, _mm256_cvtph_ps(b_half0), acc0);
                acc1 = _mm256_fmadd_ps(avec, _mm256_cvtph_ps(b_half1), acc1);
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + row * n + col), _mm256_cvtps_ph(acc0, 0));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + row * n + col + 8), _mm256_cvtps_ph(acc1, 0));
        }
        for (; col + 8 <= n; col += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int64_t inner = 0; inner < k; ++inner) {
                const float av = fp16_to_fp32(a[row * k + inner]);
                const __m256 avec = _mm256_set1_ps(av);
                const __m128i b_half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + inner * n + col));
                const __m256 b_float = _mm256_cvtph_ps(b_half);
                acc = _mm256_fmadd_ps(avec, b_float, acc);
            }
            const __m128i packed = _mm256_cvtps_ph(acc, 0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + row * n + col), packed);
        }
        for (; col < n; ++col) {
            float acc = 0.0f;
            for (int64_t inner = 0; inner < k; ++inner) {
                acc += fp16_to_fp32(a[row * k + inner]) * fp16_to_fp32(b[inner * n + col]);
            }
            c[row * n + col] = fp32_to_fp16(acc);
        }
    }
    return 0;
}

static void configure_blas_threads() {
    const char* requested_threads = std::getenv("PCKETLM_BLAS_THREADS");
    if (requested_threads == nullptr || requested_threads[0] == '\0') {
        requested_threads = std::getenv("PCKETLM_NATIVE_THREADS");
    }
    if (requested_threads != nullptr && requested_threads[0] != '\0') {
        const int parsed = std::atoi(requested_threads);
        if (parsed > 0) {
            openblas_set_num_threads(parsed);
            return;
        }
    }
    #ifdef _OPENMP
    openblas_set_num_threads(std::max(1, std::min(4, omp_get_num_procs())));
    #else
    openblas_set_num_threads(1);
    #endif
}

static void fp16_matrix_to_fp32(const uint16_t* source, float* dest, int64_t count) {
    int64_t index = 0;
    for (; index + 8 <= count; index += 8) {
        const __m128i half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + index));
        _mm256_storeu_ps(dest + index, _mm256_cvtph_ps(half));
    }
    for (; index < count; ++index) {
        dest[index] = fp16_to_fp32(source[index]);
    }
}

static void fp32_matrix_to_fp16(const float* source, uint16_t* dest, int64_t count) {
    int64_t index = 0;
    for (; index + 8 <= count; index += 8) {
        const __m256 values = _mm256_loadu_ps(source + index);
        const __m128i half = _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest + index), half);
    }
    for (; index < count; ++index) {
        dest[index] = fp32_to_fp16(source[index]);
    }
}

extern "C" __declspec(dllexport) int native_fp16_matmul(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* c,
    int64_t m,
    int64_t n,
    int64_t k
) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        return 1;
    }
    if (m < 0 || n < 0 || k < 0) {
        return 2;
    }
    const char* disable_blas = std::getenv("PCKETLM_DISABLE_BLAS_GEMM");
    if (disable_blas != nullptr && (
        std::strcmp(disable_blas, "1") == 0 ||
        std::strcmp(disable_blas, "true") == 0 ||
        std::strcmp(disable_blas, "yes") == 0 ||
        std::strcmp(disable_blas, "on") == 0
    )) {
        return handwritten_fp16_matmul(a, b, c, m, n, k);
    }
    configure_blas_threads();

    const size_t a_count = static_cast<size_t>(m * k);
    const size_t b_count = static_cast<size_t>(k * n);
    const size_t c_count = static_cast<size_t>(m * n);
    std::vector<float> a32(a_count);
    std::vector<float> b32(b_count);
    std::vector<float> c32(c_count, 0.0f);

    fp16_matrix_to_fp32(a, a32.data(), static_cast<int64_t>(a_count));
    fp16_matrix_to_fp32(b, b32.data(), static_cast<int64_t>(b_count));
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<blasint>(m),
        static_cast<blasint>(n),
        static_cast<blasint>(k),
        1.0f,
        a32.data(),
        static_cast<blasint>(k),
        b32.data(),
        static_cast<blasint>(n),
        0.0f,
        c32.data(),
        static_cast<blasint>(n)
    );
    fp32_matrix_to_fp16(c32.data(), c, static_cast<int64_t>(c_count));
    return 0;
}
