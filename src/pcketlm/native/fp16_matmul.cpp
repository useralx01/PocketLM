#include <cstdint>
#include <cstdlib>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
        for (; col + 8 <= n; col += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int64_t inner = 0; inner < k; ++inner) {
                const float av = fp16_to_fp32(a[row * k + inner]);
                const __m256 avec = _mm256_set1_ps(av);
                const __m128i b_half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + inner * n + col));
                const __m256 b_float = _mm256_cvtph_ps(b_half);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(avec, b_float));
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
