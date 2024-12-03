#pragma once
#include <array>
#include <cstdint>
namespace Ifrit::Math::ConstFunc {

template <typename T, typename U, uint32_t N>
constexpr std::array<U, N> convertArray(const std::array<T, N> &arr) {
  std::array<U, N> result;
  for (uint32_t i = 0; i < N; ++i) {
    result[i] = static_cast<U>(arr[i]);
  }
  return result;
}

template <typename T, uint32_t N>
constexpr std::array<T, N> uniformSampleIncl(T min, T max) {
  std::array<T, N> result;
  for (uint32_t i = 0; i < N; ++i) {
    float u = 1.0f * i / (N - 1);
    result[i] = min + u * (max - min);
  }
  return result;
}

// For N bins, and corresponding N values, give a point X to sample
// Then, it samples the value at the bin that X falls into B[t]<=X<B[t+1]
// and linearly interpolate the value between B[t] and B[t+1]
template <typename T, uint32_t N>
constexpr T binLerp(const std::array<T, N> bins, const std::array<T, N> values,
                    T x) {
  if (x <= bins[0]) {
    return values[0];
  }
  if (x >= bins[N - 1]) {
    return values[N - 1];
  }
  for (uint32_t i = 0; i < N - 1; ++i) {
    if (x >= bins[i] && x < bins[i + 1]) {
      T t = (x - bins[i]) / (bins[i + 1] - bins[i]);
      return values[i] * (1.0f - t) + values[i + 1] * t;
    }
  }
  return values[0];
}

// Integer division, rounding up
constexpr auto divRoundUp(auto a, auto b) { return (a + b - 1) / b; }

} // namespace Ifrit::Math::ConstFunc