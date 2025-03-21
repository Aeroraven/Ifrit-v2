
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

#include "ifrit/core/renderer/AyanamiRenderer.h"
#include "ifrit/core/renderer/framegraph/FrameGraph.h"
#include "ifrit/core/renderer/util/RenderingUtils.h"

namespace Ifrit::Core {
using namespace Ifrit::Common::Utility;
using namespace Ifrit::Core::RenderingUtil;

struct AyanamiRendererResources {
  using DrawPass = GraphicsBackend::Rhi::RhiGraphicsPass;
  using ComputePass = GraphicsBackend::Rhi::RhiComputePass;

  FrameGraphCompiler m_fgCompiler;
  FrameGraphExecutor m_fgExecutor;

  DrawPass *m_debugPass = nullptr;
};

IFRIT_APIDECL void AyanamiRenderer::initRenderer() { m_resources = new AyanamiRendererResources(); }
IFRIT_APIDECL AyanamiRenderer::~AyanamiRenderer() {
  if (m_resources)
    delete m_resources;
}

IFRIT_APIDECL void AyanamiRenderer::prepareResources(RenderTargets *renderTargets, const RendererConfig &config) {
  auto rhi = m_app->getRhiLayer();
  if (m_resources->m_debugPass == nullptr) {
    auto rtFmt = renderTargets->getFormat();
    m_resources->m_debugPass = createGraphicsPass(rhi, "Ayanami/Ayanami.CopyPass.vert.glsl",
                                                  "Ayanami/Ayanami.CopyPass.frag.glsl", 0, 0, rtFmt);
  }
}

IFRIT_APIDECL void AyanamiRenderer::setupAndRunFrameGraph(RenderTargets *renderTargets, const GPUCmdBuffer *cmd) {
  auto rhi = m_app->getRhiLayer();
  FrameGraph fg;
  auto resRenderTargets = fg.addResource("RenderTargets");
  auto passDebug = fg.addPass("DebugPass", FrameGraphPassType::Graphics, {}, {resRenderTargets}, {});

  fg.setImportedResource(resRenderTargets, renderTargets->getColorAttachment(0)->getRenderTarget(), {0, 0, 1, 1});
  fg.setExecutionFunction(passDebug, [&]() {
    char emptyPushConst[4] = {0, 0, 0, 0};
    enqueueFullScreenPass(cmd, rhi, m_resources->m_debugPass, renderTargets, {}, emptyPushConst, 0);
  });
  auto compiledFg = m_resources->m_fgCompiler.compile(fg);
  m_resources->m_fgExecutor.executeInSingleCmd(cmd, compiledFg);
}

IFRIT_APIDECL std::unique_ptr<AyanamiRenderer::GPUCommandSubmission>
AyanamiRenderer::render(Scene *scene, Camera *camera, RenderTargets *renderTargets, const RendererConfig &config,
                        const std::vector<GPUCommandSubmission *> &cmdToWait) {

  prepareImmutableResources();
  prepareResources(renderTargets, config);
  auto rhi = m_app->getRhiLayer();
  auto dq = rhi->getQueue(GraphicsBackend::Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto task =
      dq->runAsyncCommand([&](const GPUCmdBuffer *cmd) { setupAndRunFrameGraph(renderTargets, cmd); }, cmdToWait, {});
  return task;
}

} // namespace Ifrit::Core