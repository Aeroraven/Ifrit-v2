#pragma once
#include <array>
namespace Ifrit::Core {

struct RendererConsts {
  static constexpr std::array<float, 8> cHalton2 = {
      0.5f, 0.25f, 0.75f, 0.125f, 0.625f, 0.375f, 0.875f, 0.0625f};
  static constexpr std::array<float, 8> cHalton3 = {
      0.3333333333333333f, 0.6666666666666666f, 0.1111111111111111f,
      0.4444444444444444f, 0.7777777777777778f, 0.2222222222222222f,
      0.5555555555555556f, 0.8888888888888888f};
};

} // namespace Ifrit::Core