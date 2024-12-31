
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


#include "ifrit/softgraphics/engine/imaging/BufferedImageSampler.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging {
inline void processWarpMode2D(IfritSamplerAddressMode mode, float &u,
                              float &v) {
  switch (mode) {
  case IfritSamplerAddressMode::IF_SAMPLER_ADDRESS_MODE_REPEAT:
    u = u - floor(u);
    v = v - floor(v);
    break;
  case IfritSamplerAddressMode::IF_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
    u = u - floor(u);
    v = v - floor(v);
    if (int(floor(u)) % 2 == 1)
      u = 1.0f - u;
    if (int(floor(v)) % 2 == 1)
      v = 1.0f - v;
    break;
  case IfritSamplerAddressMode::IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
    u = std::clamp(u, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);
    break;
  case IfritSamplerAddressMode::IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
    u = std::clamp(u, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);
    break;
  case IfritSamplerAddressMode::IF_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE:
    u = std::clamp(u, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);
    break;
  }
}
void BufferedImageSampler::sample2DDirect(float u, float v, int lod,
                                          iint2 offset,
                                          const LayeredBufferedImage &image,
                                          void *pixel) const {
  float wpU = u, wpV = v;
  processWarpMode2D(pCI.addressModeU, wpU, wpV);
  int width = image.getLayer(lod).getWidth();
  int height = image.getLayer(lod).getHeight();
  int x = int(wpU * (width - 1));
  int y = int(wpV * (height - 1));
  image.getLayer(lod).getPixel2D(x + offset.x, y + offset.y, pixel);
}
void BufferedImageSampler::sample3DDirect(float u, float v, float w, int lod,
                                          iint3 offset,
                                          const LayeredBufferedImage &image,
                                          void *pixel) const {
  float wpU = u, wpV = v, wpW = w;
  processWarpMode2D(pCI.addressModeU, wpU, wpV);
  processWarpMode2D(pCI.addressModeV, wpV, wpW);
  int width = image.getLayer(lod).getWidth();
  int height = image.getLayer(lod).getHeight();
  int depth = image.getLayer(lod).getDepth();
  int x = int(wpU * (width - 1));
  int y = int(wpV * (height - 1));
  int z = int(wpW * (depth - 1));
  image.getLayer(lod).getPixel3D(x + offset.x, y + offset.y, z + offset.z,
                                 pixel);
}
void BufferedImageSampler::sample2DLod(float u, float v, float lod,
                                       iint2 offset,
                                       const LayeredBufferedImage &image,
                                       void *pixel) const {
  auto minLod = int(floor(lod));
  auto maxLod = int(ceil(lod));
  sample2DDirect(u, v, minLod, offset, image, pixel);
}
void BufferedImageSampler::sample3DLod(float u, float v, float w, float lod,
                                       iint3 offset,
                                       const LayeredBufferedImage &image,
                                       void *pixel) const {
  auto minLod = int(floor(lod));
  auto maxLod = int(ceil(lod));
  sample3DDirect(u, v, w, minLod, offset, image, pixel);
}
void BufferedImageSampler::sample2DLodSi(float u, float v, float lod,
                                         iint2 offset, void *pixel) const {
  sample2DLod(u, v, lod, offset, sImg, pixel);
}
void BufferedImageSampler::sample3DLodSi(float u, float v, float w, float lod,
                                         iint3 offset, void *pixel) const {
  sample3DLod(u, v, w, lod, offset, sImg, pixel);
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging
