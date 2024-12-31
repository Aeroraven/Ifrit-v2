
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "../../util/ApiConv.h"
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <bit>
#include <cstdint>

namespace Ifrit::Math::FastUtil {
inline uint64_t i32Pack(uint32_t x, uint32_t y) {
  return (uint64_t)x | ((uint64_t)y << 32);
}
inline uint32_t i32UnpackFromi64First(uint64_t x) { return (uint32_t)x; }
inline uint32_t i32UnpackFromi64Second(uint64_t x) {
  return (uint32_t)(x >> 32);
}

// This function aims to provide an optimized std fill for dword
// Some times std::fill is not optimized for qword under MSVC compiler  (no rep
// stos or sse)
template <class T> inline void memsetDword(T *src, T value, size_t counts) {
#ifdef _MSC_VER
  if constexpr (sizeof(T) == 4) {
    static_assert(sizeof(unsigned long) == 4,
                  "Unexpected size of unsigned long");
    __stosd((unsigned long *)src, std::bit_cast<unsigned long, T>(value),
            counts);
  } else {
    std::fill(src, src + counts, value);
  }
#else
  std::fill(src, src + counts, value);
#endif
}

// Returns the number of leading 0-bits in x, starting at the most significant
// bit position. If x is 0, the result is undefined.
template <class T> inline int qclz(T x) {
#ifdef _MSC_VER
  if constexpr (sizeof(T) == 4) {
    unsigned long r = 0;
    _BitScanReverse(&r, x);
    return 31 - r;
  } else if constexpr (sizeof(T) == 8) {
    unsigned long r = 0;
    _BitScanReverse64(&r, x);
    return 63 - r;
  } else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Unsupported size for clz");
  }
#else
  return __builtin_clz(x);
#endif
}

// Returns the log2 of x, rounded down. If x is 0, the result is undefined.
template <class T> inline int qlog2(T x) {
  return (((sizeof(T) * 8)) - 1) - qclz(x);
}

} // namespace Ifrit::Math::FastUtil
