#include "ifrit/softgraphics/engine/imaging/LayeredBufferedImage.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging {
void LayeredBufferedImage::addLayer(std::shared_ptr<BufferedImage> layer) {
  layers.push_back(layer);
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging