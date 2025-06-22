
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

#pragma once
#include "fastutil/FastUtil.h"

namespace Ifrit::Math
{
    /// Returns the number of trailing 0 - bits in x, starting at the most significant
    //  bit position. If x is 0, the result is undefined.
    template <class T> inline i32 CountTrailingZero(T x)
    {
#ifdef _MSC_VER
        if IF_CONSTEXPR (sizeof(T) == 4)
        {
            unsigned long r = 0;
            _BitScanReverse(&r, x);
            return r;
        }
        else if IF_CONSTEXPR (sizeof(T) == 8)
        {
            unsigned long r = 0;
            _BitScanReverse64(&r, x);
            return r;
        }
        else
        {
            static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Unsupported size for ctz");
        }
#else
        return __builtin_ctz(x);
#endif
    }

    // Packing a 32-bit integer into a 64-bit integer. With ordered comparison.
    IF_FORCEINLINE u64 OrderedPack32(u32 x, u32 y)
    {
        if (x < y)
        {
            return (u64)x | ((u64)y << 32);
        }
        else
        {
            return (u64)y | ((u64)x << 32);
        }
    }
} // namespace Ifrit::Math