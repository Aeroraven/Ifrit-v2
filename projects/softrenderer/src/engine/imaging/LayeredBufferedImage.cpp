#include "engine/imaging/LayeredBufferedImage.h"

namespace Ifrit::Engine::SoftRenderer::Imaging {
void LayeredBufferedImage::addLayer(std::shared_ptr<BufferedImage> layer) {
  layers.push_back(layer);
}
} // namespace Ifrit::Engine::SoftRenderer::Imaging