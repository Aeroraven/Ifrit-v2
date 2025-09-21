#pragma once

#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>

namespace Ifrit
{
    template <typename K, typename V> using TMap                                = std::map<K, V>;
    template <typename K, typename V, typename H = std::hash<K>> using THashMap = std::unordered_map<K, V, H>;
    template <typename T> using TSet                                            = std::set<T>;
    template <typename T, typename H = std::hash<T>> using THashSet             = std::unordered_set<T, H>;
} // namespace Ifrit