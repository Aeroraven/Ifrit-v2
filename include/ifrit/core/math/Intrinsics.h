
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
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/base/IfritBase.h"
#ifdef _MSC_VER
    #include <intrin.h>
#endif

namespace Ifrit::Math
{
    inline u64                           IntegerPack32(u32 x, u32 y) { return (u64)x | ((u64)y << 32); }
    inline u32                           IntegerUnpack32From64First(u64 x) { return (u32)x; }
    inline u32                           IntegerUnpack32From64Second(u64 x) { return (u32)(x >> 32); }

    // This function aims to provide an optimized std fill for dword
    // Some times std::fill is not optimized for qword under MSVC compiler  (no rep
    // stos or sse)
    template <class T> IF_CONSTEXPR void MemSetDword(T* src, T value, size_t counts)
    {
#ifdef _MSC_VER
        if IF_CONSTEXPR (sizeof(T) == 4)
        {
            static_assert(sizeof(unsigned long) == 4, "Unexpected size of unsigned long");
            //__stosd((unsigned long*)src, std::bit_cast<unsigned long, T>(value), counts);
            std::fill(src, src + counts, value);
        }
        else
        {
            std::fill(src, src + counts, value);
        }
#else
        std::fill(src, src + counts, value);
#endif
    }

    // Returns the number of leading 0-bits in x, starting at the most significant
    // bit position. If x is 0, the result is undefined.
    template <class T> inline i32 CountLeadingZero(T x)
    {
#ifdef _MSC_VER
        if IF_CONSTEXPR (sizeof(T) == 4)
        {
            unsigned long r = 0;
            _BitScanReverse(&r, x);
            return 31 - r;
        }
        else if IF_CONSTEXPR (sizeof(T) == 8)
        {
            unsigned long r = 0;
            _BitScanReverse64(&r, x);
            return 63 - r;
        }
        else
        {
            static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Unsupported size for clz");
        }
#else
        return __builtin_clz(x);
#endif
    }

    // Returns the log2 of x, rounded down. If x is 0, the result is undefined.
    template <class T> inline u32 IntegerLog2(T x) { return (((sizeof(T) * 8)) - 1) - CountLeadingZero(x); }

    // Returns the log2 of x, rounded up. If x is 0, the result is undefined.
    template <class T> inline u32 IntegerLog2RoundUp(T x)
    {
        if (x <= 1) IF_UNLIKELY
            return 0;
        return (sizeof(T) * 8) - CountLeadingZero(x - 1);
    }

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