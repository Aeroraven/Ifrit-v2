
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
#include "ifrit/core/typing/Traits.h"
#include <tuple>

namespace Ifrit
{
    // hash operator for std::pair
    template <IHashable T1, IHashable T2> struct PairwiseHash
    {
        usize operator()(const std::pair<T1, T2>& p) const
        {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ h2;
        }
    };

    template <typename... Types, u32 InitSeed = 0u>
    IF_NODISCARD IF_FORCEINLINE constexpr usize HashCombineImpl(TArgType<Types>... args) noexcept
    {
        usize seed = InitSeed;
        (...,
            (seed ^= std::hash<TEnumDecayedType<Types>>{}(static_cast<TEnumDecayedType<Types>>(args)) + 0x9e3779b9
                    + (seed << 6) + (seed >> 2)));
        return seed;
    }

    template <typename... Types, u32 InitSeed = 0u>
    IF_NODISCARD IF_FORCEINLINE constexpr usize HashCombine(Types&&... args) noexcept
    {
        return HashCombineImpl<Types...>(std::forward<Types>(args)...);
    }

} // namespace Ifrit