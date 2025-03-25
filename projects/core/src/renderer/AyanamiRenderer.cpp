
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

#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/core/renderer/ayanami/AyanamiSceneAggregator.h"

namespace Ifrit::Core {
using namespace Ifrit::Common::Utility;
using namespace Ifrit::Core::RenderingUtil;
using namespace Ifrit::Core::Ayanami;

struct AyanamiRendererResources {
  using DrawPass = GraphicsBackend::Rhi::RhiGraphicsPass;
  using ComputePass = GraphicsBackend::Rhi::RhiComputePass;
  using GPUTexture = GraphicsBackend::Rhi::RhiTextureRef;
  using GPUBindId = GraphicsBackend::Rhi::RhiBindlessIdRef;

  GPUTexture m_raymarchOutput = nullptr;
  Ref<GPUBindId> m_raymarchOutputUAVBindId;
  Ref<GPUBindId> m_raymarchOutputSRVBindId;

  FrameGraphCompiler m_fgCompiler;
  FrameGraphExecutor m_fgExecutor;
  Uref<AyanamiSceneAggregator> m_sceneAggregator;

  ComputePass *m_raymarchPass = nullptr;
  DrawPass *m_debugPass = nullptr;
};

IFRIT_APIDECL void AyanamiRenderer::initRenderer() {
  m_resources = new AyanamiRendererResources();
  m_resources->m_sceneAggregator = std::make_unique<AyanamiSceneAggregator>(m_app->getRhiLayer());
}
IFRIT_APIDECL AyanamiRenderer::~AyanamiRenderer() {
  if (m_resources)
    delete m_resources;
}

IFRIT_APIDECL void AyanamiRenderer::prepareResources(RenderTargets *renderTargets, const RendererConfig &config) {
  auto rhi = m_app->getRhiLayer();
  auto width = renderTargets->getRenderArea().width;
  auto height = renderTargets->getRenderArea().height;

  // Passes
  if (m_resources->m_debugPass == nullptr) {
    auto rtFmt = renderTargets->getFormat();
    m_resources->m_debugPass = createGraphicsPass(rhi, "Ayanami/Ayanami.CopyPass.vert.glsl",
                                                  "Ayanami/Ayanami.CopyPass.frag.glsl", 0, 1, rtFmt);
  }
  if (m_resources->m_raymarchPass == nullptr) {
    m_resources->m_raymarchPass = createComputePass(rhi, "Ayanami/Ayanami.RayMarch.comp.glsl", 0, 5);
  }

  // Resources
  if (m_resources->m_raymarchOutput == nullptr) {
    auto sampler = m_immRes.m_linearSampler;
    m_resources->m_raymarchOutput =
        rhi->createTexture2D("Ayanami_Raymarch", width, height, RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT,
                             RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT);
    m_resources->m_raymarchOutputUAVBindId = rhi->registerUAVImage(m_resources->m_raymarchOutput.get(), {0, 0, 1, 1});
    m_resources->m_raymarchOutputSRVBindId =
        rhi->registerCombinedImageSampler(m_resources->m_raymarchOutput.get(), sampler.get());
  }
}

IFRIT_APIDECL void AyanamiRenderer::setupAndRunFrameGraph(PerFrameData &perframe, RenderTargets *renderTargets,
                                                          const GPUCmdBuffer *cmd) {
  auto rtWidth = renderTargets->getRenderArea().width;
  auto rtHeight = renderTargets->getRenderArea().height;

  auto rhi = m_app->getRhiLayer();
  FrameGraph fg;
  auto resRaymarchOutput = fg.addResource("RaymarchOutput");
  auto resRenderTargets = fg.addResource("RenderTargets");

  auto passRaymarch = fg.addPass("RaymarchPass", FrameGraphPassType::Compute, {}, {resRaymarchOutput}, {});
  auto passDebug = fg.addPass("DebugPass", FrameGraphPassType::Graphics, {resRaymarchOutput}, {resRenderTargets}, {});

  fg.setImportedResource(resRaymarchOutput, m_resources->m_raymarchOutput.get(), {0, 0, 1, 1});
  fg.setImportedResource(resRenderTargets, renderTargets->getColorAttachment(0)->getRenderTarget(), {0, 0, 1, 1});

  fg.setExecutionFunction(passRaymarch, [&]() {
    struct RayMarchPc {
      u32 perframeId;
      u32 descId;
      u32 output;
      u32 rtH;
      u32 rtW;
    } pc;
    pc.rtH = rtHeight;
    pc.rtW = rtWidth;
    pc.output = m_resources->m_raymarchOutputUAVBindId->getActiveId();
    pc.descId = m_resources->m_sceneAggregator->getGatheredBufferId();
    pc.perframeId = perframe.m_views[0].m_viewBufferId->getActiveId();

    m_resources->m_raymarchPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
      cmd->setPushConst(m_resources->m_raymarchPass, 0, sizeof(RayMarchPc), &pc);
      cmd->dispatch(Math::ConstFunc::divRoundUp(rtWidth, 8), Math::ConstFunc::divRoundUp(rtHeight, 8), 1);
    });
    m_resources->m_raymarchPass->run(cmd, 0);
  });

  fg.setExecutionFunction(passDebug, [&]() {
    struct DispPc {
      u32 raymarchOutput;
    } pc;
    pc.raymarchOutput = m_resources->m_raymarchOutputSRVBindId->getActiveId();
    enqueueFullScreenPass(cmd, rhi, m_resources->m_debugPass, renderTargets, {}, &pc, 1);
  });
  auto compiledFg = m_resources->m_fgCompiler.compile(fg);
  m_resources->m_fgExecutor.executeInSingleCmd(cmd, compiledFg);
}

IFRIT_APIDECL std::unique_ptr<AyanamiRenderer::GPUCommandSubmission>
AyanamiRenderer::render(Scene *scene, Camera *camera, RenderTargets *renderTargets, const RendererConfig &config,
                        const std::vector<GPUCommandSubmission *> &cmdToWait) {
  if (m_perScenePerframe.count(scene) == 0) {
    m_perScenePerframe[scene] = PerFrameData();
  }
  m_config = &config;
  auto &perframeData = m_perScenePerframe[scene];

  prepareImmutableResources();
  prepareResources(renderTargets, config);

  SceneCollectConfig sceneConfig;
  sceneConfig.projectionTranslateX = 0;
  sceneConfig.projectionTranslateY = 0;
  collectPerframeData(perframeData, scene, camera, GraphicsShaderPassType::Opaque, renderTargets, sceneConfig);
  prepareDeviceResources(perframeData, renderTargets);
  m_resources->m_sceneAggregator->collectScene(scene);

  auto rhi = m_app->getRhiLayer();
  auto dq = rhi->getQueue(GraphicsBackend::Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto task = dq->runAsyncCommand(
      [&](const GPUCmdBuffer *cmd) { setupAndRunFrameGraph(perframeData, renderTargets, cmd); }, cmdToWait, {});
  return task;
}

} // namespace Ifrit::Core