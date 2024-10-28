#include "ifrit/softgraphics/engine/imaging/LayeredBufferedImage.h"

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Imaging {
void LayeredBufferedImage::addLayer(std::shared_ptr<BufferedImage> layer) {
  layers.push_back(layer);
}
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Imaging