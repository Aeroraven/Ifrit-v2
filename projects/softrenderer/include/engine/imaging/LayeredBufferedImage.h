#pragma once
#include "core/definition/CoreExports.h"
#include "engine/imaging/BufferedImage.h"

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Imaging {
class LayeredBufferedImage {
private:
  std::vector<std::shared_ptr<BufferedImage>> layers;

public:
  LayeredBufferedImage() = default;
  ~LayeredBufferedImage() = default;

  void addLayer(std::shared_ptr<BufferedImage> layer);
  inline BufferedImage &getLayer(int index) const { return *layers[index]; }
  inline int getLayerCount() const { return layers.size(); }
};
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Imaging