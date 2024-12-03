
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
#include <cstdint>

namespace Ifrit::Math::FastUtil {
inline uint64_t i32Pack(uint32_t x, uint32_t y) {
  return (uint64_t)x | ((uint64_t)y << 32);
}
inline uint32_t i32UnpackFromi64First(uint64_t x) { return (uint32_t)x; }
inline uint32_t i32UnpackFromi64Second(uint64_t x) {
  return (uint32_t)(x >> 32);
}

// This function aims to provide a optimized std fill for dword
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
} // namespace Ifrit::Math::FastUtil
