#include "core/definition/CoreExports.h"

namespace Ifrit::Math::FastUtil {
	inline uint64_t i32Pack(uint32_t x, uint32_t y) {
		return (uint64_t)x | ((uint64_t)y << 32);
	}
	inline uint32_t i32UnpackFromi64First(uint64_t x) {
		return (uint32_t)x;
	}
	inline uint32_t i32UnpackFromi64Second(uint64_t x) {
		return (uint32_t)(x >> 32);
	}
	inline float approxReciprocal(float x) {
		// https://codereview.stackexchange.com/questions/259771/fast-reciprocal-1-x
		float r = _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
		r = r * (2 - r * x);
		return r;
	}
}

