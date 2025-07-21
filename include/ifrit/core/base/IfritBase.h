
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
#include "ifrit/core/base/IfritBasicAlias.h"

#ifdef __cplusplus
namespace Ifrit
{
    template <typename T, u32 V> using Array                          = std::array<T, V>;
    template <typename T> using VecView                               = std::span<T>;
    template <typename T> using Vec                                   = std::vector<T>;
    template <typename T> using Ref                                   = std::shared_ptr<T>;
    template <typename T> using Owner                                 = std::unique_ptr<T>;
    template <typename T> using Set                                   = std::set<T>;
    template <typename T> using HashSet                               = std::unordered_set<T>;
    template <typename K, typename V> using Map                       = std::map<K, V>;
    template <typename K, typename V> using HashMap                   = std::unordered_map<K, V>;
    template <typename K, typename V, typename H> using CustomHashMap = std::unordered_map<K, V, H>;
    template <typename T> using Fn                                    = std::function<T>;
    template <typename T> using Atomic                                = std::atomic<T>;
    template <typename T, typename U> using Pair                      = std::pair<T, U>;
    using String                                                      = std::string;
    using StringView                                                  = std::string_view;
    template <typename T> using Queue                                 = std::queue<T>;
    using IntPtr                                                      = std::intptr_t;

    template <typename T, typename... Args> IF_FORCEINLINE Ref<T> MakeRef(Args&&... args)
    {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }

    template <typename T, typename... Args> IF_FORCEINLINE Owner<T> MakeOwner(Args&&... args)
    {
        return std::make_unique<T>(std::forward<Args>(args)...);
    }
} // namespace Ifrit
#endif
