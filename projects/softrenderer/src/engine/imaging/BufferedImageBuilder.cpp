#include "engine/imaging/BufferedImageBuilder.h"

namespace Ifrit::Engine::Imaging {
std::shared_ptr<BufferedImage>
BufferedImageBuilder::build(const IfritImageCreateInfo &pCI) {
  BufferedImageCreateInfo intCI;
  intCI.width = pCI.extent.width;
  intCI.height = pCI.extent.height;
  intCI.depth = pCI.extent.depth;
  if (pCI.format == IfritFormat::IF_FORMAT_R32G32B32A32_SFLOAT) {
    intCI.format = BufferedImageFormat::FORMAT_R8G8B8A8_UINT;
    intCI.channels = 4;
  } else if (pCI.format == IfritFormat::IF_FORMAT_R8G8B8A8_UINT) {
    intCI.format = BufferedImageFormat::FORMAT_R32G32B32A32_SFLOAT;
    intCI.channels = 4;
  }
  if (pCI.tilingMode == IfritImageTiling::IF_IMAGE_TILING_LINEAR) {
    intCI.tiling = BufferedImageTiling::TILING_LINEAR;
  } else if (pCI.tilingMode == IfritImageTiling::IF_IMAGE_TILING_OPTIMAL) {
    intCI.tiling = BufferedImageTiling::TILING_TILED;
  }

  // Create the image
  if (intCI.tiling == BufferedImageTiling::TILING_LINEAR) {
    if (intCI.format == BufferedImageFormat::FORMAT_R8G8B8A8_UINT) {
      return std::make_shared<BufferedImageLinear<uint8_t>>(intCI);
    } else if (intCI.format ==
               BufferedImageFormat::FORMAT_R32G32B32A32_SFLOAT) {
      return std::make_shared<BufferedImageLinear<float>>(intCI);
    }
  } else if (intCI.tiling == BufferedImageTiling::TILING_TILED) {
    if (intCI.format == BufferedImageFormat::FORMAT_R8G8B8A8_UINT) {
      return std::make_shared<BufferedImageTiled<uint8_t>>(intCI);
    } else if (intCI.format ==
               BufferedImageFormat::FORMAT_R32G32B32A32_SFLOAT) {
      return std::make_shared<BufferedImageTiled<float>>(intCI);
    }
  }
}
} // namespace Ifrit::Engine::Imaging