#pragma once

#ifdef __cplusplus
    #include <array>
    #include <functional>
    #include <memory>
    #include <string>
    #include <vector>
    #include <span>
    #include <concepts>
    #include <type_traits>
#endif

#include "ifrit/core/base/IfritBasicAlias.h"

#ifdef __cplusplus
namespace Ifrit
{
    template <typename T, u32 V> using Array     = std::array<T, V>;
    template <typename T> using VecView          = std::span<T>;
    template <typename T> using Vec              = std::vector<T>;
    template <typename T> using Ref              = std::shared_ptr<T>;
    template <typename T> using Owner            = std::unique_ptr<T>;
    template <typename T> using Fn               = std::function<T>;
    template <typename T, typename U> using Pair = std::pair<T, U>;
    using String                                 = std::string;
    using StringView                             = std::string_view;
    using IntPtr                                 = std::intptr_t;

    using ZString  = char*;
    using CZString = const char*;

    template <typename T, typename... Args>
        requires std::is_constructible_v<T, Args...>
    IF_FORCEINLINE Ref<T> MakeRef(Args&&... args)
    {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
        requires std::is_constructible_v<T, Args...>
    IF_FORCEINLINE Owner<T> MakeOwner(Args&&... args)
    {
        return std::make_unique<T>(std::forward<Args>(args)...);
    }
} // namespace Ifrit
#endif
