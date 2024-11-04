#pragma once
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/imaging/BufferedImage.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging {
class LayeredBufferedImage {
private:
  std::vector<std::shared_ptr<BufferedImage>> layers;

public:
  LayeredBufferedImage() = default;
  ~LayeredBufferedImage() = default;

  void addLayer(std::shared_ptr<BufferedImage> layer);
  inline BufferedImage &getLayer(int index) const { return *layers[index]; }
  inline int getLayerCount() const {
    using namespace Ifrit::Common::Utility;
    return size_cast<int>(layers.size());
  }
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging