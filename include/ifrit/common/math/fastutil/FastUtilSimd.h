#pragma once
#include "../../util/ApiConv.h"
#include <cstdint>

namespace Ifrit::Math::FastUtil {
#ifdef IFRIT_USE_SIMD_256
// Approximated Reciprocal with Newton-Raphson method
inline __m256 approxReciprocalNR_256(__m256 x) {
  // https://stackoverflow.com/questions/31555260/fast-vectorized-rsqrt-and-reciprocal-with-sse-avx-depending-on-precision
  __m256 res = _mm256_rcp_ps(x);
  __m256 muls = _mm256_mul_ps(x, _mm256_mul_ps(res, res));
  return res = _mm256_sub_ps(_mm256_add_ps(res, res), muls);
}

// Flip the sign of the float
inline __m256 flipFp32Sign_256(__m256 x) {
  // https://stackoverflow.com/questions/3361132/flipping-sign-on-packed-sse-floats
  return _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
}

// Fill all lanes with 1
inline __m256 fillAllOne_256() {
  auto p = _mm256_setzero_ps();
  return _mm256_cmp_ps(p, p, _CMP_EQ_OQ);
}

// Gather 32 floats from memory and store into 4 differents 256-bit registers
// With stride of 4.
inline void gather32Fp32_stride4_256(float *data, __m256 &r0, __m256 &r1,
                                     __m256 &r2, __m256 &r3) {
  // I do not know whether this is optimal
  constexpr bool useInstruction = false;
  if constexpr (useInstruction) {
    auto pos = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    r0 = _mm256_i32gather_ps(data, pos, 4);
    r1 = _mm256_i32gather_ps(data + 1, pos, 4);
    r2 = _mm256_i32gather_ps(data + 2, pos, 4);
    r3 = _mm256_i32gather_ps(data + 3, pos, 4);
  } else {
    __m256 sect0 = _mm256_loadu_ps(data);
    __m256 sect1 = _mm256_loadu_ps(data + 8);
    __m256 sect2 = _mm256_loadu_ps(data + 16);
    __m256 sect3 = _mm256_loadu_ps(data + 24);
    __m256 shuf0 = _mm256_shuffle_ps(sect0, sect1, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 shuf1 = _mm256_shuffle_ps(sect0, sect1, _MM_SHUFFLE(3, 1, 3, 1));
    __m256 shuf2 = _mm256_shuffle_ps(sect2, sect3, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 shuf3 = _mm256_shuffle_ps(sect2, sect3, _MM_SHUFFLE(3, 1, 3, 1));
    __m256i perm0 = _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7);
    __m256 p0 = _mm256_permutevar8x32_ps(shuf0, perm0);
    __m256 p1 = _mm256_permutevar8x32_ps(shuf1, perm0);
    __m256 p2 = _mm256_permutevar8x32_ps(shuf2, perm0);
    __m256 p3 = _mm256_permutevar8x32_ps(shuf3, perm0);
    r0 = _mm256_permute2f128_ps(p0, p2, 0x20);
    r2 = _mm256_permute2f128_ps(p0, p2, 0x31);
    r1 = _mm256_permute2f128_ps(p1, p3, 0x20);
    r3 = _mm256_permute2f128_ps(p1, p3, 0x31);
  }
}

#endif

inline float approxReciprocal(float x) {
#ifdef IFRIT_USE_SIMD_128
  // https://codereview.stackexchange.com/questions/259771/fast-reciprocal-1-x
  float r = _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
  r = r * (2 - r * x);
#else
  float r = 1.0f / x;
#endif
  return r;
}
inline float fastApproxReciprocal(float x) {
#ifdef IFRIT_USE_SIMD_128
  // https://codereview.stackexchange.com/questions/259771/fast-reciprocal-1-x
  float r = _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
#else
  float r = 1.0f / x;
#endif
  return r;
}

} // namespace Ifrit::Math::FastUtil