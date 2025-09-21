#pragma once
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/typing/Traits.h"
#include <bit>
#ifdef _MSC_VER
    #include <intrin.h>
#endif

namespace Ifrit
{
    // This function aims to provide an optimized std fill for dword
    // Some times std::fill is not optimized for qword under MSVC compiler  (no rep
    // stos or sse)
    template <typename T>
        requires IConceptIsDecayed<T>
    constexpr IF_FORCEINLINE void MemSet(T* src, T value, size_t counts)
    {
#ifdef _MSC_VER
        if constexpr (std::is_constant_evaluated())
        {
            std::fill(src, src + counts, value);
            return;
        }
        if constexpr (sizeof(T) == 4)
        {
            static_assert(sizeof(unsigned long) == 4, "Unexpected size of unsigned long");
            __stosd((unsigned long*)src, std::bit_cast<unsigned long, T>(value), counts);
        }
        else if constexpr (sizeof(T) == 2)
        {
            static_assert(sizeof(unsigned short) == 2, "Unexpected size of unsigned short");
            __stosw((unsigned short*)src, std::bit_cast<unsigned short, T>(value), counts);
        }
        else if constexpr (sizeof(T) == 1)
        {
            static_assert(sizeof(unsigned char) == 1, "Unexpected size of unsigned char");
            __stosb((unsigned char*)src, std::bit_cast<unsigned char, T>(value), counts);
        }
        else if constexpr (sizeof(T) == 8)
        {
            using ull = unsigned long long;
            static_assert(sizeof(ull) == 8, "Unexpected size of unsigned long long");
            __stosq((ull*)src, std::bit_cast<ull, T>(value), counts);
        }
        else
        {
            std::fill(src, src + counts, value);
        }
#else
        std::fill(src, src + counts, value);
#endif
    }

    template <typename T, usize N>
        requires IConceptIsDecayed<T>
    constexpr IF_FORCEINLINE void MemSet(Array<T, N>& arr, T value)
    {
        MemSet(arr.data(), value, N);
    }

    // Returns the number of leading 0-bits in x, starting at the most significant
    // bit position. If x is 0, the result is undefined.
    template <IScalar T>
        requires(IConceptSizeofIs<T, 4> || IConceptSizeofIs<T, 8>)
    IF_NODISCARD constexpr inline i32 CountLeadingZero(T x)
    {
        if (std::is_constant_evaluated())
        {
            if (x == 0) IF_UNLIKELY
                return sizeof(T) * 8;
            i32 count = 0;
            for (i32 i = sizeof(T) * 8 - 1; i >= 0; --i)
            {
                if ((x & (T(1) << i)) != 0)
                {
                    break;
                }
                ++count;
            }
            return count;
        }
        else
        {

#ifdef _MSC_VER
            if constexpr (IConceptSizeofIs<T, 4>)
            {
                unsigned long r = 0;
                _BitScanReverse(&r, x);
                return 31 - r;
            }
            else if constexpr (IConceptSizeofIs<T, 8>)
            {
                unsigned long r = 0;
                _BitScanReverse64(&r, x);
                return 63 - r;
            }
            return 0;
#else
            return __builtin_clz(x);
#endif
        }
    }

    /// Returns the number of trailing 0 - bits in x, starting at the most significant
    //  bit position. If x is 0, the result is undefined.
    template <IScalar T>
        requires(IConceptSizeofIs<T, 4> || IConceptSizeofIs<T, 8>)
    IF_NODISCARD constexpr inline i32 CountTrailingZero(T x)
    {
        if (std::is_constant_evaluated())
        {
            if (x == 0) IF_UNLIKELY
                return sizeof(T) * 8;
            i32 count = 0;
            for (i32 i = 0; i < sizeof(T) * 8; ++i)
            {
                if ((x & (T(1) << i)) != 0)
                {
                    break;
                }
                ++count;
            }
            return count;
        }
        else
        {
#ifdef _MSC_VER
            if constexpr (IConceptSizeofIs<T, 4>)
            {
                unsigned long r = 0;
                _BitScanReverse(&r, x);
                return r;
            }
            else if constexpr (IConceptSizeofIs<T, 8>)
            {
                unsigned long r = 0;
                _BitScanReverse64(&r, x);
                return r;
            }
            return 0;
#else
            return __builtin_ctz(x);
#endif
        }
    }

    template <IIntegral T>
        requires(IConceptSizeofIs<T, 4> || IConceptSizeofIs<T, 8>)
    IF_NODISCARD inline constexpr u32 IntegerLog2(T x)
    {
        return (((sizeof(T) * 8)) - 1) - CountLeadingZero(x);
    }

    template <IIntegral T>
        requires(IConceptSizeofIs<T, 4> || IConceptSizeofIs<T, 8>)
    inline u32 IntegerLog2RoundUp(T x)
    {
        if (x <= 1) IF_UNLIKELY
            return 0;
        return (sizeof(T) * 8) - CountLeadingZero(x - 1);
    }

    // Packing a 32-bit integer into a 64-bit integer. With ordered comparison.
    template <typename T, typename U>
        requires IConceptIsSame<T, u32> && IConceptIsSame<U, u32>
    constexpr IF_FORCEINLINE u64 OrderedPack32(T x, U y) noexcept
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

} // namespace Ifrit