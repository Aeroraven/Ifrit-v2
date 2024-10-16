#pragma once
#include <cstdint>
#include <common/core/ApiConv.h>
namespace Ifrit::Presentation::Backend {
class IFRIT_APIDECL BackendProvider {
public:
  virtual ~BackendProvider() = default;
  virtual void draw() = 0;
  virtual void updateTexture(const float* image, int channels,int width,int height) = 0;
  virtual void setViewport(int32_t x, int32_t y, int32_t width,
                           int32_t height) = 0;
};
} // namespace Ifrit::Presentation::Backend