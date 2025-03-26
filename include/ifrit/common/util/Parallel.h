#pragma once
#include <algorithm>
#include <execution>
#include <functional>
#include <ranges>
namespace Ifrit::Common::Utility
{
    template <class T>
    void UnorderedFor(T start, T end, std::function<void(T)> func)
    {
        auto rng = std::ranges::views::iota(start, end);
        std::for_each(std::execution::par_unseq, rng.begin(), rng.end(), [func](T x) { func(x); });
    }

    template <class T>
    void SequentialFor(int start, int end, std::function<void(T)> func)
    {
        for (int i = start; i < end; i++)
        {
            func(i);
        }
    }
} // namespace Ifrit::Common::Utility