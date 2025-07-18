
#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "magicenum/include/magic_enum/magic_enum.hpp"

namespace Ifrit
{
    template <typename T IF_REQUIRES(std::is_enum_v<T>)> IF_FORCEINLINE String GetEnumName(T value)
    {
        String ret = String{ magic_enum::enum_name(value) };
        return ret;
    }

    template <typename E> IF_CONSTEXPR typename std::underlying_type<E>::type GetEnumUnderlyingValue(E e) noexcept
    {
        return static_cast<typename std::underlying_type<E>::type>(e);
    }

} // namespace Ifrit