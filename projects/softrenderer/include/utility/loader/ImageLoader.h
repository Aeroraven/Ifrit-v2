#pragma once
#include "core/definition/CoreExports.h"

namespace Ifrit::Engine::SoftRenderer::Utility::Loader {
class ImageLoader {
public:
  void loadRGBA(const char *fileName, std::vector<float> *bufferOut,
                int *height, int *width);
};
} // namespace Ifrit::Engine::SoftRenderer::Utility::Loader