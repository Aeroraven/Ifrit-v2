#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/imaging/BufferedImage.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging {
class IFRIT_APIDECL BufferedImageBuilder {
public:
  BufferedImageBuilder() = default;
  ~BufferedImageBuilder() = default;

  virtual std::shared_ptr<BufferedImage> build(const IfritImageCreateInfo &pCI);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging