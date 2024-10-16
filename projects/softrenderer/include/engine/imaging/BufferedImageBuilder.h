#pragma once
#include "core/definition/CoreExports.h"
#include "engine/imaging/BufferedImage.h"

namespace Ifrit::Engine::SoftRenderer::Imaging {
class IFRIT_APIDECL BufferedImageBuilder {
public:
  BufferedImageBuilder() = default;
  ~BufferedImageBuilder() = default;

  virtual std::shared_ptr<BufferedImage> build(const IfritImageCreateInfo &pCI);
};
} // namespace Ifrit::Engine::SoftRenderer::Imaging