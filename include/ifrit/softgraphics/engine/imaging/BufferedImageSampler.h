
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


#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/Structures.h"
#include "ifrit/softgraphics/engine/imaging/LayeredBufferedImage.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging {
class BufferedImageSampler {
private:
  const IfritSamplerT pCI;
  const LayeredBufferedImage &sImg;

public:
  // BufferedImageSampler(const IfritSamplerT& createInfo): pCI(createInfo) {};
  BufferedImageSampler(const IfritSamplerT &createInfo,
                       const LayeredBufferedImage &sI)
      : pCI(createInfo), sImg(sI){};
  ~BufferedImageSampler() = default;

  void sample2DDirect(float u, float v, int lod, iint2 offset,
                      const LayeredBufferedImage &image, void *pixel) const;
  void sample3DDirect(float u, float v, float w, int lod, iint3 offset,
                      const LayeredBufferedImage &image, void *pixel) const;

  void sample2DLod(float u, float v, float lod, iint2 offset,
                   const LayeredBufferedImage &image, void *pixel) const;
  void sample3DLod(float u, float v, float w, float lod, iint3 offset,
                   const LayeredBufferedImage &image, void *pixel) const;

  void sample2DLodSi(float u, float v, float lod, iint2 offset,
                     void *pixel) const;
  void sample3DLodSi(float u, float v, float w, float lod, iint3 offset,
                     void *pixel) const;
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging