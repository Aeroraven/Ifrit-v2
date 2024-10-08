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
}