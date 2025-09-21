#pragma once
#include "ifrit/core/typing/Traits.h"

namespace Ifrit
{
    template <IEnum T>
    IF_NODISCARD constexpr IF_FORCEINLINE TTraitUnderlyingType<T> GetEnumUnderlyingValue(T e) noexcept
    {
        return static_cast<TTraitUnderlyingType<T>>(e);
    }

    template <IEnum T, typename U>
        requires IConceptConversionGuarded<U, TTraitUnderlyingType<T>>
    IF_NODISCARD constexpr IF_FORCEINLINE bool HasFlagBit(U value, T flag) noexcept
    {
        using V = TTraitUnderlyingType<T>;
        return (static_cast<V>(value) & static_cast<V>(flag)) != 0;
    }

    template <IEnum T, typename U>
        requires IConceptConversionGuarded<U, TTraitUnderlyingType<T>>
    IF_NODISCARD constexpr IF_FORCEINLINE U SetFlagBit(U value, T flag) noexcept
    {
        using V = TTraitUnderlyingType<T>;
        return static_cast<U>(static_cast<V>(value) | static_cast<V>(flag));
    }

    template <IEnum T, typename U>
        requires IConceptConversionGuarded<U, TTraitUnderlyingType<T>>
    IF_NODISCARD constexpr IF_FORCEINLINE U ClearFlagBit(U value, T flag) noexcept
    {
        using V = TTraitUnderlyingType<T>;
        return static_cast<U>(static_cast<V>(value) & ~static_cast<V>(flag));
    }

} // namespace Ifrit
