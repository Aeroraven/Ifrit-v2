#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/VertexBuffer.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::MeshletBuilder {
struct IFRIT_APIDECL Meshlet {
  VertexBuffer vbufs;
  std::vector<int> ibufs; // TODO: exporting c-style array
};
}; // namespace Ifrit::GraphicsBackend::SoftGraphics::MeshletBuilder