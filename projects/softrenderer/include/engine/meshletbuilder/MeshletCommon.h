#pragma once
#include "./core/definition/CoreExports.h"
#include "./engine/base/VertexBuffer.h"

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::MeshletBuilder {
struct IFRIT_APIDECL Meshlet {
  VertexBuffer vbufs;
  std::vector<int> ibufs; // TODO: exporting c-style array
};
}; // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::MeshletBuilder