#pragma once

#include <optional>

namespace Ifrit
{
    template <typename T> using TOptional = std::optional<T>;

    inline constexpr auto      NullOpt = std::nullopt;

    template <typename T> bool OptionalNotEmpty(const TOptional<T>& opt) { return opt.has_value(); }
} // namespace Ifrit