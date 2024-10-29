#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader {
class ImageLoader {
public:
  void loadRGBA(const char *fileName, std::vector<float> *bufferOut,
                int *height, int *width);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader