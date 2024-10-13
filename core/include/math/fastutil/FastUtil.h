#include "core/definition/CoreExports.h"
#ifdef _MSC_VER
#include <intrin.h>
#endif
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

	// This function aims to provide a optimized std fill for dword
	// Some times std::fill is not optimized for qword under MSVC compiler  (no rep stos or sse)
	template<class T>
	inline void memsetDword(T* src, T value, size_t counts) {
#ifdef _MSC_VER
		if constexpr (sizeof(T) == 4) {
			static_assert(sizeof(unsigned long) == 4, "Unexpected size of unsigned long");
			__stosd((unsigned long*)src, std::bit_cast<unsigned long, T>(value), counts);
		}
		else {
			std::fill(src, src + counts, value);
		}
#else
		std::fill(src, src + counts, value);
#endif
	}
}

