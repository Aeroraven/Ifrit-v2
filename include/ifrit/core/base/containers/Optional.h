#pragma once

#include <optional>

namespace Ifrit
{
    template <typename T> using TOptional = std::optional<T>;

    inline constexpr auto NullOpt = std::nullopt;
} // namespace Ifrit