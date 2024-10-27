#pragma once
#include "core/definition/CoreExports.h"

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Utility::Loader {
class ImageLoader {
public:
  void loadRGBA(const char *fileName, std::vector<float> *bufferOut,
                int *height, int *width);
};
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::Utility::Loader