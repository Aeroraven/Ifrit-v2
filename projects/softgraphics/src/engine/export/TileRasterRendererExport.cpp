
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


#include "ifrit/softgraphics/engine/export/TileRasterRendererExport.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/bufferman/BufferManager.h"
#include "ifrit/softgraphics/engine/tileraster/TileRasterRenderer.h"

#define IFRIT_TRNS Ifrit::GraphicsBackend::SoftGraphics::TileRaster
#define IFRIT_BASENS Ifrit::GraphicsBackend::SoftGraphics
#define IFRIT_TRTP                                                             \
  Ifrit::GraphicsBackend::SoftGraphics::LibraryExport::TileRasterRendererWrapper

using namespace Ifrit::GraphicsBackend::SoftGraphics;
using namespace Ifrit::GraphicsBackend::SoftGraphics::LibraryExport;
using namespace Ifrit::GraphicsBackend::SoftGraphics::TileRaster;

namespace Ifrit::GraphicsBackend::SoftGraphics::LibraryExport {
struct TileRasterRendererWrapper {
  std::shared_ptr<IFRIT_TRNS::TileRasterRenderer> renderer;
  std::vector<std::unique_ptr<IFRIT_BASENS::ShaderBase>> allocatedFuncWrappers;
  std::unique_ptr<std::vector<int>> allocatedIndexBuffer;
};
class VertexShaderFunctionalWrapper : virtual public VertexShader {
public:
  VertexShaderFunctionalPtr func = nullptr;
  virtual void execute(const void *const *input, ifloat4 *outPos,
                       ifloat4 *const *outVaryings) override {
    if (func)
      func(input, outPos, outVaryings);
  }
};
class FragmentShaderFunctionalWrapper : virtual public FragmentShader {
public:
  FragmentShaderFunctionalPtr func = nullptr;
  virtual void execute(const void *varyings, void *colorOutput,
                       float *fragmentDepth) override {
    if (func)
      func(varyings, colorOutput, fragmentDepth);
  }
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::LibraryExport
IFRIT_APIDECL_COMPAT IFRIT_TRTP *IFRIT_APICALL iftrCreateInstance()
    IFRIT_EXPORT_COMPAT_NOTHROW {
  auto hInst = new TileRasterRendererWrapper();
  hInst->renderer = std::make_shared<IFRIT_TRNS::TileRasterRenderer>();
  return hInst;
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
iftrDestroyInstance(IFRIT_TRTP *hInstance) IFRIT_EXPORT_COMPAT_NOTHROW {
  delete hInstance;
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindFrameBuffer(
    IFRIT_TRTP *hInstance,
    IFRIT_BASENS::FrameBuffer *frameBuffer) IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->bindFrameBuffer(*frameBuffer);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindVertexBuffer(
    IFRIT_TRTP *hInstance, const IFRIT_BASENS::VertexBuffer *vertexBuffer)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->bindVertexBuffer(*vertexBuffer);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindIndexBuffer(
    IFRIT_TRTP *hInstance, void *indexBuffer) IFRIT_EXPORT_COMPAT_NOTHROW {
  auto p = reinterpret_cast<BufferManager::IfritBuffer *>(indexBuffer);
  hInstance->renderer->bindIndexBuffer(*p);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindVertexShaderFunc(
    IFRIT_TRTP *hInstance, IFRIT_BASENS::VertexShaderFunctionalPtr func,
    IFRIT_BASENS::VaryingDescriptor *vsOutDescriptors)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  auto vsInst = std::make_unique<VertexShaderFunctionalWrapper>();
  vsInst->func = func;
  hInstance->renderer->bindVertexShaderLegacy(*vsInst, *vsOutDescriptors);
  hInstance->allocatedFuncWrappers.push_back(std::move(vsInst));
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindFragmentShaderFunc(
    IFRIT_TRTP *hInstance, IFRIT_BASENS::FragmentShaderFunctionalPtr func)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  auto fsInst = std::make_unique<FragmentShaderFunctionalWrapper>();
  fsInst->func = func;
  hInstance->renderer->bindFragmentShader(*fsInst);
  hInstance->allocatedFuncWrappers.push_back(std::move(fsInst));
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetBlendFunc(
    IFRIT_TRTP *hInstance, IFRIT_BASENS::IfritColorAttachmentBlendState *state)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->setBlendFunc(*state);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrSetDepthFunc(
    IFRIT_TRTP *hInstance,
    IFRIT_BASENS::IfritCompareOp state) IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->setDepthFunc(state);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrOptsetForceDeterministic(
    IFRIT_TRTP *hInstance, int opt) IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->optsetForceDeterministic(opt);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrOptsetDepthTestEnable(
    IFRIT_TRTP *hInstance, int opt) IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->optsetDepthTestEnable(opt);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
iftrDrawLegacy(IFRIT_TRTP *hInstance, int numVertices,
               int clearFramebuffer) IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->drawElements(numVertices, clearFramebuffer);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrClear(IFRIT_TRTP *hInstance)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->clear();
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrInit(IFRIT_TRTP *hInstance)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->init();
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrTest(void (*p)(int))
    IFRIT_EXPORT_COMPAT_NOTHROW {
  p(114514);
}

// Update v1
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindVertexShader(
    IFRIT_TRTP *hInstance, void *pVertexShader) IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->bindVertexShader(*(VertexShader *)pVertexShader);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL iftrBindFragmentShader(
    IFRIT_TRTP *hInstance, void *pFragmentShader) IFRIT_EXPORT_COMPAT_NOTHROW {
  hInstance->renderer->bindFragmentShader(*(FragmentShader *)pFragmentShader);
}

#undef IFRIT_TRTP
#undef IFRIT_BASENS
#undef IFRIT_TRNS