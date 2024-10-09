#pragma once
#include "core/definition/CoreExports.h"

namespace Ifrit::Math::FastUtil {
#ifdef IFRIT_USE_SIMD_256
	inline __m256 approxReciprocalNR_256(__m256 x) {
		// https://stackoverflow.com/questions/31555260/fast-vectorized-rsqrt-and-reciprocal-with-sse-avx-depending-on-precision
		__m256 res = _mm256_rcp_ps(x);
		__m256 muls = _mm256_mul_ps(x, _mm256_mul_ps(res, res));
		return res = _mm256_sub_ps(_mm256_add_ps(res, res), muls);
	}
	inline __m256 flipFp32Sign_256(__m256 x) {
		// https://stackoverflow.com/questions/3361132/flipping-sign-on-packed-sse-floats
		return _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
	}
	inline __m256 fillAllOne_256() {
		auto p = _mm256_setzero_ps();
		return _mm256_cmp_ps(p, p, _CMP_EQ_OQ);
	}

#endif
}