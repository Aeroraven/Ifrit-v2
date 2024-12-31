
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#include "ifrit/softgraphics/engine/imaging/BufferedImageBuilder.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging {
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
  return nullptr;
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging