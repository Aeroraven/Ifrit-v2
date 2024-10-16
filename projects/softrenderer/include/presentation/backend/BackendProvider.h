#pragma once
#include "core/data/Image.h"
#include "core/definition/CoreExports.h"
namespace Ifrit::Presentation::Backend {
class BackendProvider {
public:
  virtual ~BackendProvider() = default;
  virtual void draw() = 0;
  virtual void updateTexture(const Ifrit::Engine::SoftRenderer::Core::Data::ImageF32 &image) = 0;
  virtual void setViewport(int32_t x, int32_t y, int32_t width,
                           int32_t height) = 0;
};
} // namespace Ifrit::Presentation::Backend