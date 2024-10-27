#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/Structures.h"
#include "ifrit/softgraphics/engine/imaging/LayeredBufferedImage.h"

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Imaging {
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
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Imaging