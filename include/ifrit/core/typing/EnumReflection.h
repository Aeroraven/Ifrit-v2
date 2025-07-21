
#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "magicenum/include/magic_enum/magic_enum.hpp"

namespace Ifrit
{
    template <typename T IF_REQUIRES(std::is_enum_v<T>)> IF_CONSTEXPR IF_FORCEINLINE String GetEnumName(T value)
    {
        String ret = String{ magic_enum::enum_name(value) };
        return ret;
    }

    template <typename T IF_REQUIRES(std::is_enum_v<T>)>
    IF_CONSTEXPR IF_FORCEINLINE typename std::underlying_type<T>::type GetEnumUnderlyingValue(T e) noexcept
    {
        return static_cast<typename std::underlying_type<T>::type>(e);
    }

    template <typename T IF_REQUIRES(std::is_enum_v<T>)> IF_CONSTEXPR IF_FORCEINLINE T GetEnumFromName(StringView name)
    {
        auto enumValue = magic_enum::enum_cast<T>(name);
        return enumValue.value();
    }

} // namespace Ifrit
