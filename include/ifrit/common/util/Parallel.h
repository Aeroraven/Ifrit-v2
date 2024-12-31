#pragma once
#include <algorithm>
#include <execution>
#include <functional>
#include <ranges>
namespace Ifrit::Common::Utility {
template <class T>
void unordered_for(T start, T end, std::function<void(T)> func) {
  auto rng = std::ranges::views::iota(start, end);
  std::for_each(std::execution::par_unseq, rng.begin(), rng.end(),
                [func](T x) { func(x); });
}
} // namespace Ifrit::Common::Utility