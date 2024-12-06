
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

#include "ifrit/core/renderer/SyaroRenderer.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/core/renderer/RendererUtil.h"
#include <algorithm>
#include <bit>

#include "ifrit.shader/Syaro/Syaro.SharedConst.h"

using namespace Ifrit::GraphicsBackend::Rhi;

namespace Ifrit::Core {

struct GPUHiZDesc {
  uint32_t m_width;
  uint32_t m_height;
};

std::vector<Ifrit::GraphicsBackend::Rhi::RhiResourceBarrier>
registerUAVBarriers(
    const std::vector<Ifrit::GraphicsBackend::Rhi::RhiBuffer *> &buffers,
    const std::vector<Ifrit::GraphicsBackend::Rhi::RhiTexture *> &textures) {
  std::vector<Ifrit::GraphicsBackend::Rhi::RhiResourceBarrier> barriers;
  for (auto &buffer : buffers) {
    Ifrit::GraphicsBackend::Rhi::RhiUAVBarrier barrier;
    barrier.m_type = Ifrit::GraphicsBackend::Rhi::RhiResourceType::Buffer;
    barrier.m_buffer = buffer;
    Ifrit::GraphicsBackend::Rhi::RhiResourceBarrier resBarrier;
    resBarrier.m_type = Ifrit::GraphicsBackend::Rhi::RhiBarrierType::UAVAccess;
    resBarrier.m_uav = barrier;
    barriers.push_back(resBarrier);
  }
  for (auto &texture : textures) {
    Ifrit::GraphicsBackend::Rhi::RhiUAVBarrier barrier;
    barrier.m_type = Ifrit::GraphicsBackend::Rhi::RhiResourceType::Texture;
    barrier.m_texture = texture;
    Ifrit::GraphicsBackend::Rhi::RhiResourceBarrier resBarrier;
    resBarrier.m_type = Ifrit::GraphicsBackend::Rhi::RhiBarrierType::UAVAccess;
    resBarrier.m_uav = barrier;
    barriers.push_back(resBarrier);
  }
  return barriers;
}

IFRIT_APIDECL PerFrameData::PerViewData &
SyaroRenderer::getPrimaryView(PerFrameData &perframeData) {
  for (auto &view : perframeData.m_views) {
    if (view.m_viewType == PerFrameData::ViewType::Primary) {
      return view;
    }
  }
  throw std::runtime_error("Primary view not found");
  return perframeData.m_views[0];
}
IFRIT_APIDECL void SyaroRenderer::createTimer() {
  auto rhi = m_app->getRhiLayer();
  auto timer = rhi->createDeviceTimer();
  m_timer = timer;
}

IFRIT_APIDECL SyaroRenderer::GPUShader *SyaroRenderer::createShaderFromFile(
    const std::string &shaderPath, const std::string &entry,
    GraphicsBackend::Rhi::RhiShaderStage stage) {
  auto rhi = m_app->getRhiLayer();
  std::string shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  auto path = shaderBasePath + "/Syaro/" + shaderPath;
  auto shaderCode = Ifrit::Common::Utility::readTextFile(path);
  std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
  return rhi->createShader(shaderPath, shaderCodeVec, entry, stage,
                           RhiShaderSourceType::GLSLCode);
}
IFRIT_APIDECL void SyaroRenderer::setupTextureShowPass() {
  auto rhi = m_app->getRhiLayer();

  auto vsShader = createShaderFromFile("Syaro.FullScreenCopy.vert.glsl", "main",
                                       RhiShaderStage::Vertex);
  auto fsShader = createShaderFromFile("Syaro.FullScreenCopy.frag.glsl", "main",
                                       RhiShaderStage::Fragment);

  m_textureShowPass = rhi->createGraphicsPass();
  m_textureShowPass->setVertexShader(vsShader);
  m_textureShowPass->setPixelShader(fsShader);
  m_textureShowPass->setNumBindlessDescriptorSets(1); // Only one texture

  Ifrit::GraphicsBackend::Rhi::RhiRenderTargetsFormat rtFmt;
  rtFmt.m_colorFormats = {RhiImageFormat::RHI_FORMAT_B8G8R8A8_SRGB};
  rtFmt.m_depthFormat = RhiImageFormat::RHI_FORMAT_D32_SFLOAT;
  m_textureShowPass->setRenderTargetFormat(rtFmt);
}

IFRIT_APIDECL void SyaroRenderer::setupPostprocessPassAndTextures() {
  // passes
  m_acesToneMapping =
      std::make_unique<PostprocessPassCollection::PostFxAcesToneMapping>(m_app);
  // tex and samplers
  m_postprocTexSampler = m_app->getRhiLayer()->createTrivialSampler();
}

IFRIT_APIDECL void SyaroRenderer::createPostprocessTextures(uint32_t width,
                                                            uint32_t height) {
  auto rhi = m_app->getRhiLayer();
  auto rtFmt = RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT;
  if (m_postprocTex.find({width, height}) != m_postprocTex.end()) {
    return;
  }
  for (uint32_t i = 0; i < 2; i++) {
    auto tex = rhi->createRenderTargetTexture(width, height, rtFmt, 0);
    auto colorRT =
        rhi->createRenderTarget(tex.get(), {{0.0f, 0.0f, 0.0f, 1.0f}},
                                RhiRenderTargetLoadOp::Clear, 0, 0);
    auto rts = rhi->createRenderTargets();

    m_postprocTex[{width, height}][i] = tex;
    m_postprocTexId[{width, height}][i] = rhi->registerCombinedImageSampler(
        tex.get(), m_postprocTexSampler.get());
    m_postprocColorRT[{width, height}][i] = colorRT;
    rts->setColorAttachments({colorRT.get()});
    m_postprocRTs[{width, height}][i] = rts;
  }
}

IFRIT_APIDECL void
SyaroRenderer::renderToneMapping(PerFrameData &perframeData,
                                 RenderTargets *renderTargets,
                                 const GPUCmdBuffer *cmd) {
  auto primaryView = getPrimaryView(perframeData);
  auto rhi = m_app->getRhiLayer();
  auto width = primaryView.m_viewData.m_renderWidth;
  auto height = primaryView.m_viewData.m_renderHeight;
  auto srcPostprocTex = m_postprocTexId[{width, height}][0].get();
  m_acesToneMapping->renderPostFx(cmd, renderTargets, srcPostprocTex);
}

IFRIT_APIDECL void SyaroRenderer::setupPbrAtmosphereRenderer() {
  m_atmosphereRenderer = std::make_shared<PbrAtmosphereRenderer>(m_app);
  auto rhi = m_app->getRhiLayer();
  auto csShader = createShaderFromFile("Syaro.PbrAtmoRender.comp.glsl", "main",
                                       RhiShaderStage::Compute);
  m_atmospherePass = rhi->createComputePass();
  m_atmospherePass->setComputeShader(csShader);
  m_atmospherePass->setNumBindlessDescriptorSets(0);
  m_atmospherePass->setPushConstSize(sizeof(uint32_t) * 15);
}

IFRIT_APIDECL void
SyaroRenderer::setupDeferredShadingPass(RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();

  // This seems to be a bit of redundant code
  // The rhi backend now can reference the pipeline with similar CI

  PipelineAttachmentConfigs paCfg;
  auto rtCfg = renderTargets->getFormat();
  paCfg.m_colorFormats = {cTAAFormat};
  paCfg.m_depthFormat = RhiImageFormat::RHI_FORMAT_UNDEFINED;

  rtCfg.m_colorFormats = paCfg.m_colorFormats;
  rtCfg.m_depthFormat = RhiImageFormat::RHI_FORMAT_UNDEFINED;

  DrawPass *pass = nullptr;
  if (m_deferredShadingPass.find(paCfg) != m_deferredShadingPass.end()) {
    pass = m_deferredShadingPass[paCfg];
  } else {
    pass = rhi->createGraphicsPass();
    auto vsShader = createShaderFromFile("Syaro.DeferredShading.vert.glsl",
                                         "main", RhiShaderStage::Vertex);
    auto fsShader = createShaderFromFile("Syaro.DeferredShading.frag.glsl",
                                         "main", RhiShaderStage::Fragment);
    pass->setVertexShader(vsShader);
    pass->setPixelShader(fsShader);
    pass->setNumBindlessDescriptorSets(3);
    pass->setPushConstSize(3 * sizeof(uint32_t));
    pass->setRenderTargetFormat(rtCfg);
    m_deferredShadingPass[paCfg] = pass;
  }
}

IFRIT_APIDECL void SyaroRenderer::setupTAAPass(RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();

  PipelineAttachmentConfigs paCfg;
  auto rtCfg = renderTargets->getFormat();
  paCfg.m_colorFormats = {cTAAFormat,
                          RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT};
  paCfg.m_depthFormat = rtCfg.m_depthFormat;
  rtCfg.m_colorFormats = paCfg.m_colorFormats;

  DrawPass *pass = nullptr;
  if (m_taaPass.find(paCfg) != m_taaPass.end()) {
    pass = m_taaPass[paCfg];
  } else {
    pass = rhi->createGraphicsPass();
    auto vsShader = createShaderFromFile("Syaro.TAA.vert.glsl", "main",
                                         RhiShaderStage::Vertex);
    auto fsShader = createShaderFromFile("Syaro.TAA.frag.glsl", "main",
                                         RhiShaderStage::Fragment);
    pass->setVertexShader(vsShader);
    pass->setPixelShader(fsShader);
    pass->setNumBindlessDescriptorSets(2);
    pass->setPushConstSize(sizeof(uint32_t) * 5);
    pass->setRenderTargetFormat(rtCfg);
    m_taaPass[paCfg] = pass;
  }
}

IFRIT_APIDECL void
SyaroRenderer::renderDeferredShading(PerFrameData &perframeData,
                                     RenderTargets *renderTargets,
                                     const GPUCmdBuffer *cmd) {
  auto rhi = m_app->getRhiLayer();
  setupDeferredShadingPass(renderTargets);

  auto curRT = perframeData.m_taaHistory[perframeData.m_frameId % 2].m_rts;

  PipelineAttachmentConfigs paCfg;
  auto rtCfg = curRT->getFormat();
  paCfg.m_colorFormats = rtCfg.m_colorFormats;
  paCfg.m_depthFormat = RhiImageFormat::RHI_FORMAT_UNDEFINED;

  auto pass = m_deferredShadingPass[paCfg];
  auto &primaryView = getPrimaryView(perframeData);
  struct DeferPushConst {
    uint32_t shadowMapDataRef;
    uint32_t numShadowMaps;
    uint32_t depthTexRef;
  } pc;
  pc.numShadowMaps = perframeData.m_views.size() - 1;
  pc.shadowMapDataRef =
      perframeData.m_shadowData.m_allShadowDataId->getActiveId();
  pc.depthTexRef = primaryView.m_visDepthId->getActiveId();

  pass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->attachBindlessReferenceGraphics(pass, 1,
                                                perframeData.m_gbufferDescFrag);
    ctx->m_cmd->attachBindlessReferenceGraphics(
        pass, 2, perframeData.m_gbufferDepthDesc);
    ctx->m_cmd->attachBindlessReferenceGraphics(pass, 3,
                                                primaryView.m_viewBindlessRef);
    ctx->m_cmd->attachVertexBufferView(
        *rhi->getFullScreenQuadVertexBufferView());
    ctx->m_cmd->attachVertexBuffers(
        0, {rhi->getFullScreenQuadVertexBuffer().get()});
    ctx->m_cmd->setPushConst(pass, 0, sizeof(DeferPushConst), &pc);
    ctx->m_cmd->drawInstanced(3, 1, 0, 0);
  });
  cmd->beginScope("Syaro: Deferred Shading");
  pass->run(cmd, curRT.get(), 0);
  cmd->endScope();
}

void SyaroRenderer::renderTAAResolve(PerFrameData &perframeData,
                                     RenderTargets *renderTargets,
                                     const GPUCmdBuffer *cmd) {
  auto rhi = m_app->getRhiLayer();
  auto taaRT = rhi->createRenderTargets();
  auto taaCurTargetTex =
      perframeData.m_taaHistory[perframeData.m_frameId % 2].m_colorRT;
  auto taaCurTarget = rhi->createRenderTarget(
      taaCurTargetTex.get(), {}, RhiRenderTargetLoadOp::Clear, 0, 0);

  auto primaryView = getPrimaryView(perframeData);
  auto width = primaryView.m_viewData.m_renderWidth;
  auto height = primaryView.m_viewData.m_renderHeight;

  createPostprocessTextures(width, height);
  auto taaRenderTarget = m_postprocColorRT[{width, height}][0].get();

  taaRT->setColorAttachments({taaCurTarget.get(), taaRenderTarget});
  taaRT->setDepthStencilAttachment(renderTargets->getDepthStencilAttachment());
  taaRT->setRenderArea(renderTargets->getRenderArea());

  setupTAAPass(renderTargets);
  auto rtCfg = taaRT->getFormat();
  PipelineAttachmentConfigs paCfg;
  paCfg.m_colorFormats = rtCfg.m_colorFormats;
  paCfg.m_depthFormat = rtCfg.m_depthFormat;

  auto pass = m_taaPass[paCfg];
  auto renderArea = renderTargets->getRenderArea();
  uint32_t jitterX = std::bit_cast<uint32_t, float>(perframeData.m_taaJitterX);
  uint32_t jitterY = std::bit_cast<uint32_t, float>(perframeData.m_taaJitterY);
  uint32_t pconst[5] = {
      perframeData.m_frameId,
      renderArea.width,
      renderArea.height,
      jitterX,
      jitterY,
  };
  pass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->attachBindlessReferenceGraphics(pass, 1,
                                                perframeData.m_taaHistoryDesc);
    ctx->m_cmd->attachBindlessReferenceGraphics(
        pass, 2, perframeData.m_gbufferDepthDesc);
    ctx->m_cmd->setPushConst(pass, 0, sizeof(uint32_t) * 3, pconst);
    ctx->m_cmd->attachVertexBufferView(
        *rhi->getFullScreenQuadVertexBufferView());
    ctx->m_cmd->attachVertexBuffers(
        0, {rhi->getFullScreenQuadVertexBuffer().get()});
    ctx->m_cmd->drawInstanced(3, 1, 0, 0);
  });
  cmd->beginScope("Syaro: TAA Resolve");
  pass->run(cmd, taaRT.get(), 0);
  cmd->endScope();
}

IFRIT_APIDECL void SyaroRenderer::setupVisibilityPass() {
  auto rhi = m_app->getRhiLayer();
  auto tsShader = createShaderFromFile("Syaro.VisBuffer.task.glsl", "main",
                                       RhiShaderStage::Task);
  auto msShader = createShaderFromFile("Syaro.VisBuffer.mesh.glsl", "main",
                                       RhiShaderStage::Mesh);
  auto fsShader = createShaderFromFile("Syaro.VisBuffer.frag.glsl", "main",
                                       RhiShaderStage::Fragment);

  m_visibilityPass = rhi->createGraphicsPass();
  m_visibilityPass->setTaskShader(tsShader);
  m_visibilityPass->setMeshShader(msShader);
  m_visibilityPass->setPixelShader(fsShader);
  m_visibilityPass->setNumBindlessDescriptorSets(3);
  m_visibilityPass->setPushConstSize(sizeof(uint32_t));

  Ifrit::GraphicsBackend::Rhi::RhiRenderTargetsFormat rtFmt;
  rtFmt.m_colorFormats = {RhiImageFormat::RHI_FORMAT_R32_UINT};
  rtFmt.m_depthFormat = RhiImageFormat::RHI_FORMAT_D32_SFLOAT;
  m_visibilityPass->setRenderTargetFormat(rtFmt);

  m_depthOnlyVisibilityPass = rhi->createGraphicsPass();
  m_depthOnlyVisibilityPass->setTaskShader(tsShader);
  m_depthOnlyVisibilityPass->setMeshShader(msShader);
  m_depthOnlyVisibilityPass->setNumBindlessDescriptorSets(3);
  m_depthOnlyVisibilityPass->setPushConstSize(sizeof(uint32_t));

  Ifrit::GraphicsBackend::Rhi::RhiRenderTargetsFormat depthOnlyRtFmt;
  depthOnlyRtFmt.m_colorFormats = {};
  depthOnlyRtFmt.m_depthFormat = RhiImageFormat::RHI_FORMAT_D32_SFLOAT;
  m_depthOnlyVisibilityPass->setRenderTargetFormat(depthOnlyRtFmt);
}

IFRIT_APIDECL void SyaroRenderer::setupInstanceCullingPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.InstanceCulling.comp.glsl", "main",
                                     RhiShaderStage::Compute);

  m_instanceCullingPass = rhi->createComputePass();
  m_instanceCullingPass->setComputeShader(shader);
  m_instanceCullingPass->setNumBindlessDescriptorSets(4);
  m_instanceCullingPass->setPushConstSize(sizeof(uint32_t) * 2);
}

IFRIT_APIDECL void SyaroRenderer::setupPersistentCullingPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.PersistentCulling.comp.glsl",
                                     "main", RhiShaderStage::Compute);

  m_persistentCullingPass = rhi->createComputePass();
  m_persistentCullingPass->setComputeShader(shader);
  m_persistentCullingPass->setNumBindlessDescriptorSets(5);
  m_persistentCullingPass->setPushConstSize(sizeof(uint32_t));

  m_indirectDrawBuffer = rhi->createIndirectMeshDrawBufferDevice(
      1, Ifrit::GraphicsBackend::Rhi::RhiBufferUsage::
             RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  m_indirectDrawBufferId = rhi->registerStorageBuffer(m_indirectDrawBuffer);
  m_persistCullDesc = rhi->createBindlessDescriptorRef();
  m_persistCullDesc->addStorageBuffer(m_indirectDrawBuffer, 0);
}

IFRIT_APIDECL void SyaroRenderer::setupHiZPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.HiZ.comp.glsl", "main",
                                     RhiShaderStage::Compute);

  m_hizPass = rhi->createComputePass();
  m_hizPass->setComputeShader(shader);
  m_hizPass->setNumBindlessDescriptorSets(1);
}

IFRIT_APIDECL void SyaroRenderer::setupSinglePassHiZPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.SinglePassHiZ.comp.glsl", "main",
                                     RhiShaderStage::Compute);
  m_singlePassHiZPass = rhi->createComputePass();
  m_singlePassHiZPass->setComputeShader(shader);
  m_singlePassHiZPass->setNumBindlessDescriptorSets(1);
  m_singlePassHiZPass->setPushConstSize(sizeof(uint32_t) * 5);
}
IFRIT_APIDECL void SyaroRenderer::setupEmitDepthTargetsPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.EmitDepthTarget.comp.glsl", "main",
                                     RhiShaderStage::Compute);

  m_emitDepthTargetsPass = rhi->createComputePass();
  m_emitDepthTargetsPass->setComputeShader(shader);
  m_emitDepthTargetsPass->setNumBindlessDescriptorSets(4);
  m_emitDepthTargetsPass->setPushConstSize(2 * sizeof(uint32_t));
}

IFRIT_APIDECL void SyaroRenderer::setupMaterialClassifyPass() {
  auto rhi = m_app->getRhiLayer();
  // Count pass
  if constexpr (true) {
    auto shader = createShaderFromFile("Syaro.ClassifyMaterial.Count.comp.glsl",
                                       "main", RhiShaderStage::Compute);
    m_matclassCountPass = rhi->createComputePass();
    m_matclassCountPass->setComputeShader(shader);
    m_matclassCountPass->setNumBindlessDescriptorSets(1);
    m_matclassCountPass->setPushConstSize(sizeof(uint32_t) * 3);
  }
  // Reserve pass
  if constexpr (true) {
    auto shader =
        createShaderFromFile("Syaro.ClassifyMaterial.Reserve.comp.glsl", "main",
                             RhiShaderStage::Compute);
    m_matclassReservePass = rhi->createComputePass();
    m_matclassReservePass->setComputeShader(shader);
    m_matclassReservePass->setNumBindlessDescriptorSets(1);
    m_matclassReservePass->setPushConstSize(sizeof(uint32_t) * 3);
  }
  // Scatter pass
  if constexpr (true) {
    auto shader =
        createShaderFromFile("Syaro.ClassifyMaterial.Scatter.comp.glsl", "main",
                             RhiShaderStage::Compute);
    m_matclassScatterPass = rhi->createComputePass();
    m_matclassScatterPass->setComputeShader(shader);
    m_matclassScatterPass->setNumBindlessDescriptorSets(1);
    m_matclassScatterPass->setPushConstSize(sizeof(uint32_t) * 3);
  }
  // Debug pass
  if constexpr (true) {
    auto shader = createShaderFromFile("Syaro.ClassifyMaterial.Debug.comp.glsl",
                                       "main", RhiShaderStage::Compute);
    m_matclassDebugPass = rhi->createComputePass();
    m_matclassDebugPass->setComputeShader(shader);
    m_matclassDebugPass->setNumBindlessDescriptorSets(1);
    m_matclassDebugPass->setPushConstSize(sizeof(uint32_t) * 3);
  }
}

IFRIT_APIDECL void
SyaroRenderer::materialClassifyBufferSetup(PerFrameData &perframeData,
                                           RenderTargets *renderTargets) {
  auto numMaterials = perframeData.m_enabledEffects.size();
  auto rhi = m_app->getRhiLayer();
  auto renderArea = renderTargets->getRenderArea();
  auto width = renderArea.width + renderArea.x;
  auto height = renderArea.height + renderArea.y;
  auto totalSize = width * height;
  bool needRecreate = false;
  bool needRecreateMat = false;
  bool needRecreatePixel = false;
  if (perframeData.m_matClassSupportedNumMaterials < numMaterials ||
      perframeData.m_matClassCountBuffer == nullptr) {
    needRecreate = true;
    needRecreateMat = true;
  }
  if (perframeData.m_matClassSupportedNumPixels < totalSize) {
    needRecreate = true;
    needRecreatePixel = true;
  }
  if (!needRecreate) {
    return;
  }
  if (needRecreateMat) {
    perframeData.m_matClassSupportedNumMaterials = numMaterials;
    auto createSize = cMatClassCounterBufferSizeBase +
                      cMatClassCounterBufferSizeMult * numMaterials;
    perframeData.m_matClassCountBuffer = rhi->createStorageBufferDevice(
        createSize, RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);

    perframeData.m_matClassIndirectDispatchBuffer =
        rhi->createStorageBufferDevice(
            sizeof(uint32_t) * 4 * numMaterials,
            RhiBufferUsage::RHI_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  }
  if (needRecreatePixel) {
    perframeData.m_matClassSupportedNumPixels = totalSize;
    perframeData.m_matClassFinalBuffer = rhi->createStorageBufferDevice(
        totalSize * sizeof(uint32_t),
        RhiBufferUsage::RHI_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    perframeData.m_matClassPixelOffsetBuffer = rhi->createStorageBufferDevice(
        totalSize * sizeof(uint32_t),
        RhiBufferUsage::RHI_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    perframeData.m_matClassDebug = rhi->createRenderTargetTexture(
        width, height, RhiImageFormat::RHI_FORMAT_R32_UINT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT);
    perframeData.m_matClassDebug = rhi->createRenderTargetTexture(
        width, height, RhiImageFormat::RHI_FORMAT_R32_UINT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT);
  }

  if (needRecreate) {
    perframeData.m_matClassDesc = rhi->createBindlessDescriptorRef();
    perframeData.m_matClassDesc->addUAVImage(
        perframeData.m_velocityMaterial.get(), {0, 0, 1, 1}, 0);
    perframeData.m_matClassDesc->addStorageBuffer(
        perframeData.m_matClassCountBuffer, 1);
    perframeData.m_matClassDesc->addStorageBuffer(
        perframeData.m_matClassFinalBuffer, 2);
    perframeData.m_matClassDesc->addStorageBuffer(
        perframeData.m_matClassPixelOffsetBuffer, 3);
    perframeData.m_matClassDesc->addStorageBuffer(
        perframeData.m_matClassIndirectDispatchBuffer, 4);
    perframeData.m_matClassDesc->addUAVImage(perframeData.m_matClassDebug.get(),
                                             {0, 0, 1, 1}, 5);

    perframeData.m_matClassBarrier.clear();
    RhiResourceBarrier barrierCountBuffer;
    barrierCountBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierCountBuffer.m_uav.m_buffer = perframeData.m_matClassCountBuffer;
    barrierCountBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierFinalBuffer;
    barrierFinalBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierFinalBuffer.m_uav.m_buffer = perframeData.m_matClassFinalBuffer;
    barrierFinalBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierPixelOffsetBuffer;
    barrierPixelOffsetBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierPixelOffsetBuffer.m_uav.m_buffer =
        perframeData.m_matClassPixelOffsetBuffer;
    barrierPixelOffsetBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierIndirectDispatchBuffer;
    barrierIndirectDispatchBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierIndirectDispatchBuffer.m_uav.m_buffer =
        perframeData.m_matClassIndirectDispatchBuffer;
    barrierIndirectDispatchBuffer.m_uav.m_type = RhiResourceType::Buffer;

    perframeData.m_matClassBarrier.push_back(barrierCountBuffer);
    perframeData.m_matClassBarrier.push_back(barrierFinalBuffer);
    perframeData.m_matClassBarrier.push_back(barrierPixelOffsetBuffer);
    perframeData.m_matClassBarrier.push_back(barrierIndirectDispatchBuffer);
  }
}

IFRIT_APIDECL void
SyaroRenderer::depthTargetsSetup(PerFrameData &perframeData,
                                 RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();
  auto rtArea = renderTargets->getRenderArea();
  if (perframeData.m_velocityMaterial != nullptr)
    return;
  perframeData.m_velocityMaterial = rhi->createRenderTargetTexture(
      rtArea.width + rtArea.x, rtArea.height + rtArea.y,
      RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT,
      RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);
  perframeData.m_velocityMaterialDesc = rhi->createBindlessDescriptorRef();
  perframeData.m_velocityMaterialDesc->addUAVImage(
      perframeData.m_velocityMaterial.get(), {0, 0, 1, 1}, 0);

  auto &primaryView = getPrimaryView(perframeData);
  perframeData.m_velocityMaterialDesc->addCombinedImageSampler(
      primaryView.m_visibilityBuffer.get(),
      perframeData.m_visibilitySampler.get(), 1);

  // For gbuffer, depth is required to reconstruct position
  perframeData.m_gbufferDepthDesc = rhi->createBindlessDescriptorRef();
  perframeData.m_gbufferDepthSampler = rhi->createTrivialSampler();
  perframeData.m_gbufferDepthDesc->addCombinedImageSampler(
      perframeData.m_velocityMaterial.get(),
      perframeData.m_gbufferDepthSampler.get(), 0);
  perframeData.m_gbufferDepthIdX = rhi->registerCombinedImageSampler(
      primaryView.m_visPassDepth, primaryView.m_visDepthSampler.get());
}

IFRIT_APIDECL void
SyaroRenderer::recreateInstanceCullingBuffers(PerFrameData &perframe,
                                              uint32_t newMaxInstances) {
  for (uint32_t i = 0; i < perframe.m_views.size(); i++) {
    auto &view = perframe.m_views[i];
    if (view.m_maxSupportedInstances == 0 ||
        view.m_maxSupportedInstances < newMaxInstances) {
      auto rhi = m_app->getRhiLayer();
      view.m_maxSupportedInstances = newMaxInstances;
      view.m_instCullDiscardObj =
          rhi->createStorageBufferDevice(newMaxInstances * sizeof(uint32_t), 0);
      view.m_instCullPassedObj =
          rhi->createStorageBufferDevice(newMaxInstances * sizeof(uint32_t), 0);
      view.m_persistCullIndirectDispatch = rhi->createStorageBufferDevice(
          sizeof(uint32_t) * 12,
          Ifrit::GraphicsBackend::Rhi::RhiBufferUsage::
                  RHI_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
              Ifrit::GraphicsBackend::Rhi::RhiBufferUsage::
                  RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
      view.m_instCullDesc = rhi->createBindlessDescriptorRef();
      view.m_instCullDesc->addStorageBuffer(view.m_instCullDiscardObj, 0);
      view.m_instCullDesc->addStorageBuffer(view.m_instCullPassedObj, 1);
      view.m_instCullDesc->addStorageBuffer(view.m_persistCullIndirectDispatch,
                                            2);

      // create barriers
      view.m_persistCullBarrier.clear();
      view.m_persistCullBarrier = registerUAVBarriers(
          {view.m_instCullDiscardObj, view.m_instCullPassedObj,
           view.m_persistCullIndirectDispatch},
          {});

      view.m_visibilityBarrier.clear();
      view.m_visibilityBarrier = registerUAVBarriers(
          {view.m_allFilteredMeshlets, view.m_allFilteredMeshletsCount}, {});
    }
  }
}

IFRIT_APIDECL void SyaroRenderer::renderEmitDepthTargets(
    PerFrameData &perframeData, SyaroRenderer::RenderTargets *renderTargets,
    const GPUCmdBuffer *cmd) {
  auto rhi = m_app->getRhiLayer();
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);

  auto &primaryView = getPrimaryView(perframeData);
  m_emitDepthTargetsPass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        auto primaryView = getPrimaryView(perframeData);

        ctx->m_cmd->imageBarrier(primaryView.m_visPassDepth,
                                 RhiResourceState::DepthStencilRenderTarget,
                                 RhiResourceState::Common, {0, 0, 1, 1});
        ctx->m_cmd->imageBarrier(
            perframeData.m_velocityMaterial.get(), RhiResourceState::Undefined,
            RhiResourceState::UAVStorageImage, {0, 0, 1, 1});
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_emitDepthTargetsPass, 1, primaryView.m_viewBindlessRef);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_emitDepthTargetsPass, 2,
            perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_emitDepthTargetsPass, 3, primaryView.m_allFilteredMeshletsDesc);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_emitDepthTargetsPass, 4, perframeData.m_velocityMaterialDesc);
        uint32_t pcData[2] = {primaryView.m_viewData.m_renderWidth,
                              primaryView.m_viewData.m_renderHeight};
        ctx->m_cmd->setPushConst(m_emitDepthTargetsPass, 0,
                                 sizeof(uint32_t) * 2, &pcData[0]);
        uint32_t wgX =
            (pcData[0] + cEmitDepthGroupSizeX - 1) / cEmitDepthGroupSizeX;
        uint32_t wgY =
            (pcData[1] + cEmitDepthGroupSizeY - 1) / cEmitDepthGroupSizeY;
        ctx->m_cmd->dispatch(wgX, wgY, 1);
        ctx->m_cmd->imageBarrier(perframeData.m_velocityMaterial.get(),
                                 RhiResourceState::UAVStorageImage,
                                 RhiResourceState::UAVStorageImage,
                                 {0, 0, 1, 1});
        ctx->m_cmd->imageBarrier(
            primaryView.m_visPassDepth, RhiResourceState::Common,
            RhiResourceState::DepthStencilRenderTarget, {0, 0, 1, 1});
      });

  cmd->globalMemoryBarrier();
  cmd->beginScope("Syaro: Emit Depth Targets");
  m_emitDepthTargetsPass->run(cmd, 0);
  cmd->endScope();
}
IFRIT_APIDECL void SyaroRenderer::renderTwoPassOcclCulling(
    CullingPass cullPass, PerFrameData &perframeData,
    RenderTargets *renderTargets, const GPUCmdBuffer *cmd,
    PerFrameData::ViewType filteredViewType, uint32_t idx) {
  auto rhi = m_app->getRhiLayer();
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);

  int pcData[2] = {0, 1};

  std::unique_ptr<SyaroRenderer::GPUCommandSubmission> lastTask = nullptr;
  uint32_t k = idx;
  if (k == ~0u) {
    for (k = 0; k < perframeData.m_views.size(); k++) {
      if (filteredViewType == perframeData.m_views[k].m_viewType) {
        break;
      }
    }
  }
  if (filteredViewType != perframeData.m_views[k].m_viewType) {
    return;
  }
  if (k != 0 && cullPass != CullingPass::First) {
    cmd->globalMemoryBarrier();
  }
  auto &perView = perframeData.m_views[k];
  auto numObjs = perframeData.m_allInstanceData.m_objectData.size();
  int pcDataInstCull[4] = {0, numObjs, 1, numObjs};
  m_instanceCullingPass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        if (cullPass == CullingPass::First) {
          ctx->m_cmd->uavBufferClear(perView.m_persistCullIndirectDispatch, 0);
        }
        ctx->m_cmd->uavBufferBarrier(perView.m_persistCullIndirectDispatch);
        ctx->m_cmd->attachBindlessReferenceCompute(m_instanceCullingPass, 1,
                                                   perView.m_viewBindlessRef);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_instanceCullingPass, 2,
            perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
        ctx->m_cmd->attachBindlessReferenceCompute(m_instanceCullingPass, 3,
                                                   perView.m_instCullDesc);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_instanceCullingPass, 4, perView.m_spHiZData.m_hizDesc);

        if (cullPass == CullingPass::First) {
          ctx->m_cmd->setPushConst(m_instanceCullingPass, 0,
                                   sizeof(uint32_t) * 2, &pcDataInstCull[0]);
          auto tgx =
              (numObjs + SyaroConfig::cInstanceCullingThreadGroupSizeX - 1) /
              SyaroConfig::cInstanceCullingThreadGroupSizeX;
          ctx->m_cmd->dispatch(tgx, 1, 1);
        } else if (cullPass == CullingPass::Second) {
          ctx->m_cmd->setPushConst(m_instanceCullingPass, 0,
                                   sizeof(uint32_t) * 2, &pcDataInstCull[2]);
          ctx->m_cmd->dispatchIndirect(perView.m_persistCullIndirectDispatch,
                                       3 * sizeof(uint32_t));
        }
      });

  m_persistentCullingPass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        ctx->m_cmd->resourceBarrier(perView.m_persistCullBarrier);
        if (cullPass == CullingPass::First) {
          ctx->m_cmd->uavBufferClear(perView.m_allFilteredMeshletsCount, 0);
          ctx->m_cmd->uavBufferBarrier(perView.m_allFilteredMeshletsCount);
          ctx->m_cmd->uavBufferClear(m_indirectDrawBuffer, 0);
          ctx->m_cmd->uavBufferBarrier(m_indirectDrawBuffer);
        } else {
          ctx->m_cmd->uavBufferBarrier(perView.m_allFilteredMeshletsCount);
          ctx->m_cmd->uavBufferBarrier(m_indirectDrawBuffer);
        }
        // bind view buffer
        ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 1,
                                                   perView.m_viewBindlessRef);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_persistentCullingPass, 2,
            perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
        ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 3,
                                                   m_persistCullDesc);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_persistentCullingPass, 4, perView.m_allFilteredMeshletsDesc);
        ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 5,
                                                   perView.m_instCullDesc);
        if (cullPass == CullingPass::First) {
          ctx->m_cmd->setPushConst(m_persistentCullingPass, 0, sizeof(uint32_t),
                                   &pcData[0]);
          ctx->m_cmd->dispatchIndirect(perView.m_persistCullIndirectDispatch,
                                       0);
        } else if (cullPass == CullingPass::Second) {
          ctx->m_cmd->setPushConst(m_persistentCullingPass, 0, sizeof(uint32_t),
                                   &pcData[1]);
          ctx->m_cmd->dispatchIndirect(perView.m_persistCullIndirectDispatch,
                                       6 * sizeof(uint32_t));
        }
      });

  auto &visPass = (filteredViewType == PerFrameData::ViewType::Primary)
                      ? m_visibilityPass
                      : m_depthOnlyVisibilityPass;
  visPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    // bind view buffer
    ctx->m_cmd->attachBindlessReferenceGraphics(visPass, 1,
                                                perView.m_viewBindlessRef);
    ctx->m_cmd->attachBindlessReferenceGraphics(
        visPass, 2, perframeData.m_allInstanceData.m_batchedObjBufRef);
    ctx->m_cmd->setCullMode(RhiCullMode::Back);
    ctx->m_cmd->attachBindlessReferenceGraphics(
        visPass, 3, perView.m_allFilteredMeshletsDesc);
    if (cullPass == CullingPass::First) {
      ctx->m_cmd->setPushConst(visPass, 0, sizeof(uint32_t), &pcData[0]);
      ctx->m_cmd->drawMeshTasksIndirect(perView.m_allFilteredMeshletsCount,
                                        sizeof(uint32_t) * 3, 1, 0);
    } else {
      ctx->m_cmd->setPushConst(visPass, 0, sizeof(uint32_t), &pcData[1]);
      ctx->m_cmd->drawMeshTasksIndirect(perView.m_allFilteredMeshletsCount,
                                        sizeof(uint32_t) * 0, 1, 0);
    }
  });

  // auto &primaryView = getPrimaryView(perframeData);
  auto width = perView.m_viewData.m_renderWidth;
  auto height = perView.m_viewData.m_renderHeight;
  uint32_t pushConst[5] = {perView.m_spHiZData.m_hizWidth,
                           perView.m_spHiZData.m_hizHeight, width, height,
                           perView.m_spHiZData.m_hizIters};
  m_singlePassHiZPass->setRecordFunction([&](RhiRenderPassContext *ctx) {
    ctx->m_cmd->attachBindlessReferenceCompute(m_singlePassHiZPass, 1,
                                               perView.m_spHiZData.m_hizDesc);
    ctx->m_cmd->setPushConst(m_singlePassHiZPass, 0, sizeof(uint32_t) * 5,
                             &pushConst[0]);
    auto tgX =
        (perView.m_spHiZData.m_hizWidth + cSPHiZTileSize - 1) / cSPHiZTileSize;
    auto tgY =
        (perView.m_spHiZData.m_hizHeight + cSPHiZTileSize - 1) / cSPHiZTileSize;
    ctx->m_cmd->dispatch(tgX, tgY, 1);
  });

  cmd->beginScope("Syaro: Instance Culling Pass");
  m_instanceCullingPass->run(cmd, 0);
  cmd->endScope();
  cmd->beginScope("Syaro: Persistent Culling Pass");
  m_persistentCullingPass->run(cmd, 0);
  cmd->endScope();
  cmd->globalMemoryBarrier();
  auto visPassRd = (filteredViewType == PerFrameData::ViewType::Primary)
                       ? m_visibilityPass
                       : m_depthOnlyVisibilityPass;
  if (cullPass == CullingPass::First) {
    // PERFORMANCE BOTTLNECK
    cmd->beginScope("Syaro: Visibility Pass, First");
    visPassRd->run(cmd, perView.m_visRTs.get(), 0);
    cmd->endScope();
  } else {
    // we wont clear the visibility buffer in the second pass
    cmd->beginScope("Syaro: Visibility Pass, Second");
    visPassRd->run(cmd, perView.m_visRTs2.get(), 0);
    cmd->endScope();
  }
  cmd->globalMemoryBarrier();
  cmd->beginScope("Syaro: HiZ Pass");
  m_singlePassHiZPass->run(cmd, 0);
  cmd->endScope();
}

IFRIT_APIDECL void
SyaroRenderer::renderMaterialClassify(PerFrameData &perframeData,
                                      RenderTargets *renderTargets,
                                      const GPUCmdBuffer *cmd) {
  auto rhi = m_app->getRhiLayer();
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto totalMaterials = perframeData.m_enabledEffects.size();

  auto renderArea = renderTargets->getRenderArea();
  auto width = renderArea.width + renderArea.x;
  auto height = renderArea.height + renderArea.y;
  uint32_t pcData[3] = {width, height, totalMaterials};

  constexpr uint32_t pTileWidth =
      cMatClassQuadSize * cMatClassGroupSizeCountScatterX;
  constexpr uint32_t pTileHeight =
      cMatClassQuadSize * cMatClassGroupSizeCountScatterY;

  // Counting
  auto wgX = (width + pTileWidth - 1) / pTileWidth;
  auto wgY = (height + pTileHeight - 1) / pTileHeight;
  m_matclassCountPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->uavBufferClear(perframeData.m_matClassCountBuffer, 0);
    ctx->m_cmd->uavBufferBarrier(perframeData.m_matClassCountBuffer);

    ctx->m_cmd->attachBindlessReferenceCompute(m_matclassCountPass, 1,
                                               perframeData.m_matClassDesc);
    ctx->m_cmd->setPushConst(m_matclassCountPass, 0, sizeof(uint32_t) * 3,
                             &pcData[0]);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
  });

  // Reserving
  auto wgX2 = (totalMaterials + cMatClassGroupSizeReserveX - 1) /
              cMatClassGroupSizeReserveX;
  m_matclassReservePass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        ctx->m_cmd->attachBindlessReferenceCompute(m_matclassReservePass, 1,
                                                   perframeData.m_matClassDesc);
        ctx->m_cmd->setPushConst(m_matclassReservePass, 0, sizeof(uint32_t) * 3,
                                 &pcData[0]);
        ctx->m_cmd->dispatch(wgX2, 1, 1);
      });

  // Scatter
  m_matclassScatterPass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        ctx->m_cmd->attachBindlessReferenceCompute(m_matclassScatterPass, 1,
                                                   perframeData.m_matClassDesc);
        ctx->m_cmd->setPushConst(m_matclassScatterPass, 0, sizeof(uint32_t) * 3,
                                 &pcData[0]);
        ctx->m_cmd->dispatch(wgX, wgY, 1);
      });

  // Start rendering
  cmd->globalMemoryBarrier();
  cmd->beginScope("Syaro: Material Classification");
  m_matclassCountPass->run(cmd, 0);
  cmd->resourceBarrier(perframeData.m_matClassBarrier);
  m_matclassReservePass->run(cmd, 0);
  cmd->resourceBarrier(perframeData.m_matClassBarrier);
  m_matclassScatterPass->run(cmd, 0);
  cmd->endScope();
}

IFRIT_APIDECL void SyaroRenderer::setupDefaultEmitGBufferPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.EmitGBuffer.Default."
                                     "comp.glsl",
                                     "main", RhiShaderStage::Compute);
  m_defaultEmitGBufferPass = rhi->createComputePass();
  m_defaultEmitGBufferPass->setComputeShader(shader);
  m_defaultEmitGBufferPass->setNumBindlessDescriptorSets(6);
  m_defaultEmitGBufferPass->setPushConstSize(sizeof(uint32_t) * 3);
}

IFRIT_APIDECL void SyaroRenderer::hizBufferSetup(PerFrameData &perframeData,
                                                 RenderTargets *renderTargets) {
  if (perframeData.m_views.size() > 1) {
    // throw std::runtime_error("Multiple views not supported for HiZ buffer");
  }
  for (uint32_t k = 0; k < perframeData.m_views.size(); k++) {
    auto &perView = perframeData.m_views[k];
    auto renderArea = renderTargets->getRenderArea();
    auto width = perView.m_viewData.m_renderWidth;
    auto height = perView.m_viewData.m_renderHeight;
    bool cond = (perView.m_hizTexture == nullptr);
    if (!cond && (perView.m_hizTexture->getWidth() != width ||
                  perView.m_hizTexture->getHeight() != height)) {
      cond = true;
    }
    if (!cond) {
      return;
    }
    auto rhi = m_app->getRhiLayer();
    auto maxMip = int(std::floor(std::log2(std::max(width, height))) + 1);
    perView.m_hizTexture = rhi->createRenderTargetMipTexture(
        width, height, maxMip, RhiImageFormat::RHI_FORMAT_R32_SFLOAT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);

    uint32_t rWidth = renderArea.width + renderArea.x;
    uint32_t rHeight = renderArea.height + renderArea.y;
    perView.m_hizDepthSampler = rhi->createTrivialSampler();

    for (int i = 0; i < maxMip; i++) {
      auto desc = rhi->createBindlessDescriptorRef();
      auto hizTexSize = rhi->createStorageBufferDevice(
          sizeof(GPUHiZDesc),
          RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
      GPUHiZDesc hizDesc = {rWidth, rHeight};
      auto stagedBuf = rhi->createStagedSingleBuffer(hizTexSize);
      auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
      tq->runSyncCommand([&](const RhiCommandBuffer *cmd) {
        stagedBuf->cmdCopyToDevice(cmd, &hizDesc, sizeof(GPUHiZDesc), 0);
      });
      rWidth = std::max(1u, rWidth / 2);
      rHeight = std::max(1u, rHeight / 2);
      desc->addCombinedImageSampler(perView.m_visPassDepth,
                                    perView.m_hizDepthSampler.get(), 0);
      desc->addUAVImage(perView.m_hizTexture.get(),
                        {static_cast<uint32_t>(std::max(0, i - 1)), 0, 1, 1},
                        1);
      desc->addUAVImage(perView.m_hizTexture.get(),
                        {static_cast<uint32_t>(i), 0, 1, 1}, 2);
      desc->addStorageBuffer(hizTexSize, 3);
      perView.m_hizDescs.push_back(desc);
    }
    perView.m_hizIter = maxMip;

    // For hiz-testing, we need to
    // create a descriptor for the hiz
    // texture Seems UAV/Storage image
    // does not support mip-levels
    for (int i = 0; i < maxMip; i++) {
      perView.m_hizTestMips.push_back(rhi->registerUAVImage(
          perView.m_hizTexture.get(), {static_cast<uint32_t>(i), 0, 1, 1}));
      perView.m_hizTestMipsId.push_back(
          perView.m_hizTestMips.back()->getActiveId());
    }
    perView.m_hizTestMipsBuffer = rhi->createStorageBufferDevice(
        sizeof(uint32_t) * maxMip,
        RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
    auto staged = rhi->createStagedSingleBuffer(perView.m_hizTestMipsBuffer);
    tq->runSyncCommand([&](const RhiCommandBuffer *cmd) {
      staged->cmdCopyToDevice(cmd, perView.m_hizTestMipsId.data(),
                              sizeof(uint32_t) * maxMip, 0);
    });
    perView.m_hizTestDesc = rhi->createBindlessDescriptorRef();
    perView.m_hizTestDesc->addStorageBuffer(perView.m_hizTestMipsBuffer, 0);
  }
}

IFRIT_APIDECL void
SyaroRenderer::sphizBufferSetup(PerFrameData &perframeData,
                                RenderTargets *renderTargets) {
  for (uint32_t k = 0; k < perframeData.m_views.size(); k++) {
    auto &perView = perframeData.m_views[k];
    auto renderArea = renderTargets->getRenderArea();
    auto width = perView.m_viewData.m_renderWidth;
    auto height = perView.m_viewData.m_renderHeight;
    bool cond = (perView.m_spHiZData.m_hizTexture == nullptr);

    // Make width and heights power of 2
    auto rWidth = 1 << (int)std::ceil(std::log2(width));
    auto rHeight = 1 << (int)std::ceil(std::log2(height));
    if (!cond && (perView.m_spHiZData.m_hizTexture->getWidth() != rWidth ||
                  perView.m_spHiZData.m_hizTexture->getHeight() != rHeight)) {
      cond = true;
    }
    if (!cond) {
      return;
    }
    auto rhi = m_app->getRhiLayer();
    auto maxMip = int(std::floor(std::log2(std::max(rWidth, rHeight))) + 1);
    perView.m_spHiZData.m_hizTexture = rhi->createRenderTargetMipTexture(
        rWidth, rHeight, maxMip, RhiImageFormat::RHI_FORMAT_R32_SFLOAT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);
    perView.m_spHiZData.m_hizSampler = rhi->createTrivialSampler();

    // The first bit is for spd atomics
    perView.m_spHiZData.m_hizRefs.push_back(0);
    for (int i = 0; i < maxMip; i++) {
      auto bindlessId =
          rhi->registerUAVImage(perView.m_spHiZData.m_hizTexture.get(),
                                {static_cast<uint32_t>(i), 0, 1, 1});
      perView.m_spHiZData.m_hizRefs.push_back(bindlessId->getActiveId());
    }

    perView.m_spHiZData.m_hizRefBuffer = rhi->createStorageBufferDevice(
        sizeof(uint32_t) * (perView.m_spHiZData.m_hizRefs.size()),
        RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);

    perView.m_spHiZData.m_hizAtomics = rhi->createStorageBufferDevice(
        sizeof(uint32_t) * (perView.m_spHiZData.m_hizRefs.size()),
        RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);

    auto staged =
        rhi->createStagedSingleBuffer(perView.m_spHiZData.m_hizRefBuffer);
    auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
    tq->runSyncCommand([&](const RhiCommandBuffer *cmd) {
      staged->cmdCopyToDevice(
          cmd, perView.m_spHiZData.m_hizRefs.data(),
          sizeof(uint32_t) * perView.m_spHiZData.m_hizRefs.size(), 0);
    });

    perView.m_spHiZData.m_hizDesc = rhi->createBindlessDescriptorRef();
    perView.m_spHiZData.m_hizDesc->addStorageBuffer(
        perView.m_spHiZData.m_hizRefBuffer, 1);
    perView.m_spHiZData.m_hizDesc->addCombinedImageSampler(
        perView.m_visPassDepth, perView.m_spHiZData.m_hizSampler.get(), 0);
    perView.m_spHiZData.m_hizDesc->addStorageBuffer(
        perView.m_spHiZData.m_hizAtomics, 2);

    perView.m_spHiZData.m_hizIters = maxMip;
    perView.m_spHiZData.m_hizWidth = rWidth;
    perView.m_spHiZData.m_hizHeight = rHeight;
  }
}

IFRIT_APIDECL void
SyaroRenderer::visibilityBufferSetup(PerFrameData &perframeData,
                                     RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();
  auto rtArea = renderTargets->getRenderArea();

  for (uint32_t k = 0; k < perframeData.m_views.size(); k++) {
    auto &perView = perframeData.m_views[k];
    bool createCond = (perView.m_visibilityBuffer == nullptr);
    if (!createCond) {
      auto visHeight = perView.m_visibilityBuffer->getHeight();
      auto visWidth = perView.m_visibilityBuffer->getWidth();
      auto rtSize = renderTargets->getRenderArea();
      createCond = (visHeight != rtSize.height + rtSize.x ||
                    visWidth != rtSize.width + rtSize.y);
    }
    auto visHeight = perView.m_viewData.m_renderHeight;
    auto visWidth = perView.m_viewData.m_renderWidth;
    if (visHeight == 0 || visWidth == 0) {
      if (perView.m_viewType == PerFrameData::ViewType::Primary) {
        // use render target size
        visHeight = rtArea.height + rtArea.y;
        visWidth = rtArea.width + rtArea.x;
      } else if (perView.m_viewType == PerFrameData::ViewType::Shadow) {
        visHeight = m_renderConfig.m_defaultShadowMapSize;
        visWidth = m_renderConfig.m_defaultShadowMapSize;
      }
    }
    // TODO: all shadow passes do not need visibility buffer, but
    //  depth buffer is required for shadow passes

    if (!createCond) {
      return;
    }
    // It seems nanite's paper uses
    // R32G32 for mesh visibility, but
    // I wonder the depth is implicitly
    // calculated from the depth buffer
    // so here I use R32 for visibility
    // buffer
    auto visBuffer = rhi->createRenderTargetTexture(
        visWidth, visHeight, PerFrameData::c_visibilityFormat,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);
    auto visDepth = rhi->createDepthRenderTexture(visWidth, visHeight);
    auto visDepthSampler = rhi->createTrivialSampler();
    perView.m_visibilityBuffer = visBuffer;

    // first pass rts
    perView.m_visPassDepth = visDepth;
    perView.m_visDepthSampler = visDepthSampler;
    perView.m_visDepthId = rhi->registerCombinedImageSampler(
        perView.m_visPassDepth, perView.m_visDepthSampler.get());

    perView.m_visDepthRT = rhi->createRenderTargetDepthStencil(
        visDepth, {{}, 1.0f}, RhiRenderTargetLoadOp::Clear);

    perView.m_visRTs = rhi->createRenderTargets();
    if (perView.m_viewType == PerFrameData::ViewType::Primary) {
      perView.m_visColorRT = rhi->createRenderTarget(
          visBuffer.get(), {{0, 0, 0, 0}}, RhiRenderTargetLoadOp::Clear, 0, 0);
      perView.m_visRTs->setColorAttachments({perView.m_visColorRT.get()});
    } else {
      // shadow passes do not need color attachment
      perView.m_visRTs->setColorAttachments({});
    }
    perView.m_visRTs->setDepthStencilAttachment(perView.m_visDepthRT.get());
    perView.m_visRTs->setRenderArea(renderTargets->getRenderArea());
    if (perView.m_viewType == PerFrameData::ViewType::Shadow) {
      perView.m_visRTs->setRenderArea({0, 0,
                                       m_renderConfig.m_defaultShadowMapSize,
                                       m_renderConfig.m_defaultShadowMapSize});
    }

    // second pass rts

    perView.m_visDepthRT2 = rhi->createRenderTargetDepthStencil(
        visDepth, {{}, 1.0f}, RhiRenderTargetLoadOp::Load);
    perView.m_visRTs2 = rhi->createRenderTargets();
    if (perView.m_viewType == PerFrameData::ViewType::Primary) {
      perView.m_visColorRT2 = rhi->createRenderTarget(
          visBuffer.get(), {{0, 0, 0, 0}}, RhiRenderTargetLoadOp::Load, 0, 0);
      perView.m_visRTs2->setColorAttachments({perView.m_visColorRT2.get()});
    } else {
      // shadow passes do not need color attachment
      perView.m_visRTs2->setColorAttachments({});
    }
    perView.m_visRTs2->setDepthStencilAttachment(perView.m_visDepthRT2.get());
    perView.m_visRTs2->setRenderArea(renderTargets->getRenderArea());
    if (perView.m_viewType == PerFrameData::ViewType::Shadow) {
      perView.m_visRTs2->setRenderArea({0, 0,
                                        m_renderConfig.m_defaultShadowMapSize,
                                        m_renderConfig.m_defaultShadowMapSize});
    }

    // Then a sampler
    perframeData.m_visibilitySampler = rhi->createTrivialSampler();
    perframeData.m_visShowCombinedRef = rhi->createBindlessDescriptorRef();
    perframeData.m_visShowCombinedRef->addCombinedImageSampler(
        perView.m_visibilityBuffer.get(),
        perframeData.m_visibilitySampler.get(), 0);
  }
}

IFRIT_APIDECL void
SyaroRenderer::renderDefaultEmitGBuffer(PerFrameData &perframeData,
                                        RenderTargets *renderTargets,
                                        const GPUCmdBuffer *cmd) {
  auto numMaterials = perframeData.m_enabledEffects.size();
  auto rtArea = renderTargets->getRenderArea();
  m_defaultEmitGBufferPass->setRecordFunction([&](const RhiRenderPassContext
                                                      *ctx) {
    // first transition all
    // gbuffer textures to
    // UAV/Common
    ctx->m_cmd->imageBarrier(
        perframeData.m_gbuffer.m_albedo_materialFlags.get(),
        RhiResourceState::Undefined, RhiResourceState::Common, {0, 0, 1, 1});
    ctx->m_cmd->imageBarrier(perframeData.m_gbuffer.m_normal_smoothness.get(),
                             RhiResourceState::Undefined,
                             RhiResourceState::Common, {0, 0, 1, 1});
    ctx->m_cmd->imageBarrier(perframeData.m_gbuffer.m_emissive.get(),
                             RhiResourceState::Undefined,
                             RhiResourceState::Common, {0, 0, 1, 1});
    ctx->m_cmd->imageBarrier(perframeData.m_gbuffer.m_specular_occlusion.get(),
                             RhiResourceState::Undefined,
                             RhiResourceState::Common, {0, 0, 1, 1});
    ctx->m_cmd->imageBarrier(perframeData.m_gbuffer.m_shadowMask.get(),
                             RhiResourceState::Undefined,
                             RhiResourceState::Common, {0, 0, 1, 1});

    // Clear gbuffer textures
    ctx->m_cmd->clearUAVImageFloat(
        perframeData.m_gbuffer.m_albedo_materialFlags.get(), {0, 0, 1, 1},
        {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->clearUAVImageFloat(
        perframeData.m_gbuffer.m_normal_smoothness.get(), {0, 0, 1, 1},
        {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->clearUAVImageFloat(perframeData.m_gbuffer.m_emissive.get(),
                                   {0, 0, 1, 1}, {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->clearUAVImageFloat(
        perframeData.m_gbuffer.m_specular_occlusion.get(), {0, 0, 1, 1},
        {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->clearUAVImageFloat(perframeData.m_gbuffer.m_shadowMask.get(),
                                   {0, 0, 1, 1}, {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->resourceBarrier(perframeData.m_gbuffer.m_gbufferBarrier);

    // For each material, make
    // one dispatch
    uint32_t pcData[3] = {0, rtArea.width + rtArea.x, rtArea.height + rtArea.y};
    auto &primaryView = getPrimaryView(perframeData);
    for (int i = 0; i < numMaterials; i++) {
      ctx->m_cmd->attachBindlessReferenceCompute(m_defaultEmitGBufferPass, 1,
                                                 perframeData.m_matClassDesc);
      ctx->m_cmd->attachBindlessReferenceCompute(
          m_defaultEmitGBufferPass, 2, perframeData.m_gbuffer.m_gbufferDesc);
      ctx->m_cmd->attachBindlessReferenceCompute(m_defaultEmitGBufferPass, 3,
                                                 primaryView.m_viewBindlessRef);
      ctx->m_cmd->attachBindlessReferenceCompute(
          m_defaultEmitGBufferPass, 4,
          perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
      ctx->m_cmd->attachBindlessReferenceCompute(
          m_defaultEmitGBufferPass, 5, primaryView.m_allFilteredMeshletsDesc);
      ctx->m_cmd->attachBindlessReferenceCompute(
          m_defaultEmitGBufferPass, 6, perframeData.m_velocityMaterialDesc);

      pcData[0] = i;
      ctx->m_cmd->setPushConst(m_defaultEmitGBufferPass, 0,
                               sizeof(uint32_t) * 3, &pcData[0]);
      RhiResourceBarrier barrierCountBuffer;
      barrierCountBuffer.m_type = RhiBarrierType::UAVAccess;
      barrierCountBuffer.m_uav.m_buffer = perframeData.m_matClassCountBuffer;
      barrierCountBuffer.m_uav.m_type = RhiResourceType::Buffer;

      ctx->m_cmd->resourceBarrier({barrierCountBuffer});
      ctx->m_cmd->dispatchIndirect(
          perframeData.m_matClassIndirectDispatchBuffer,
          i * sizeof(uint32_t) * 4);
      ctx->m_cmd->resourceBarrier(perframeData.m_gbuffer.m_gbufferBarrier);
    }
  });
  auto rhi = m_app->getRhiLayer();
  auto computeQueue = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  cmd->globalMemoryBarrier();
  cmd->beginScope("Syaro: Emit  GBuffer");
  m_defaultEmitGBufferPass->run(cmd, 0);
  cmd->endScope();
}

IFRIT_APIDECL void SyaroRenderer::renderAtmosphere(PerFrameData &perframeData,
                                                   RenderTargets *renderTargets,
                                                   const GPUCmdBuffer *cmd) {
  // todo: implement
  auto atmoData = m_atmosphereRenderer->getResourceDesc(perframeData);
  auto rhi = m_app->getRhiLayer();
  struct AtmoPushConst {
    uint32_t m_perframe;
    uint32_t m_outTex;
    uint32_t m_depthTex;
    uint32_t pad1;
    decltype(atmoData) m_atmoData;
  } pushConst;
  auto primaryView = getPrimaryView(perframeData);
  pushConst.m_perframe = primaryView.m_viewBufferId->getActiveId();
  pushConst.m_outTex = perframeData.m_taaHistory[perframeData.m_frameId % 2]
                           .m_colorRTId->getActiveId();
  pushConst.m_atmoData = atmoData;
  pushConst.m_depthTex = primaryView.m_visDepthId->getActiveId();

  m_atmospherePass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->imageBarrier(perframeData.m_taaUnresolved.get(),
                             RhiResourceState::Undefined,
                             RhiResourceState::Common, {0, 0, 1, 1});
    ctx->m_cmd->setPushConst(m_atmospherePass, 0, sizeof(AtmoPushConst),
                             &pushConst);
    auto wgX =
        Math::ConstFunc::divRoundUp(primaryView.m_viewData.m_renderWidth,
                                    SyaroConfig::cAtmoRenderThreadGroupSizeX);
    auto wgY =
        Math::ConstFunc::divRoundUp(primaryView.m_viewData.m_renderHeight,
                                    SyaroConfig::cAtmoRenderThreadGroupSizeY);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
    ctx->m_cmd->imageBarrier(perframeData.m_taaUnresolved.get(),
                             RhiResourceState::Common,
                             RhiResourceState::RenderTarget, {0, 0, 1, 1});
  });

  cmd->globalMemoryBarrier();
  cmd->beginScope("Syaro: Atmosphere");
  m_atmospherePass->run(cmd, 0);
  cmd->endScope();
}

IFRIT_APIDECL void
SyaroRenderer::renderAmbientOccl(PerFrameData &perframeData,
                                 RenderTargets *renderTargets,
                                 const GPUCmdBuffer *cmd) {
  auto primaryView = getPrimaryView(perframeData);

  auto normalSamp = perframeData.m_gbuffer.m_normal_smoothness_sampId;
  auto depthSamp = perframeData.m_gbufferDepthIdX;
  auto perframe = primaryView.m_viewBufferId;
  auto ao = perframeData.m_gbuffer.m_specular_occlusionId;

  uint32_t width = primaryView.m_viewData.m_renderWidth;
  uint32_t height = primaryView.m_viewData.m_renderHeight;
  cmd->beginScope("Syaro: Ambient Occlusion");
  cmd->resourceBarrier({perframeData.m_gbuffer.m_normal_smoothnessBarrier,
                        perframeData.m_gbuffer.m_specular_occlusionBarrier});
  m_aoPass->renderHBAO(cmd, width, height, depthSamp.get(), normalSamp.get(),
                       ao.get(), perframe.get());
  cmd->resourceBarrier({perframeData.m_gbuffer.m_specular_occlusionBarrier});
  cmd->endScope();
}

IFRIT_APIDECL void
SyaroRenderer::gatherAllInstances(PerFrameData &perframeData) {
  uint32_t totalInstances = 0;
  uint32_t totalMeshlets = 0;
  for (auto x : perframeData.m_enabledEffects) {
    auto &effect = perframeData.m_shaderEffectData[x];
    totalInstances += effect.m_objectData.size();
  }
  auto rhi = m_app->getRhiLayer();
  if (perframeData.m_allInstanceData.m_lastObjectCount != totalInstances) {
    perframeData.m_allInstanceData.m_lastObjectCount = totalInstances;
    perframeData.m_allInstanceData.m_batchedObjectData =
        rhi->createStorageBufferShared(totalInstances * sizeof(PerObjectData),
                                       true, 0);
    perframeData.m_allInstanceData.m_batchedObjBufRef =
        rhi->createBindlessDescriptorRef();
    auto buf = perframeData.m_allInstanceData.m_batchedObjectData;
    perframeData.m_allInstanceData.m_batchedObjBufRef->addStorageBuffer(buf, 0);
  }
  perframeData.m_allInstanceData.m_objectData.resize(totalInstances);
  // TODO, EMERGENT: the logic is confusing here. Losing instance->mesh
  // relation.
  for (auto i = 0; auto &x : perframeData.m_enabledEffects) {
    auto &effect = perframeData.m_shaderEffectData[x];
    for (auto &obj : effect.m_objectData) {
      perframeData.m_allInstanceData.m_objectData[i] = obj;
      i++;
    }
    uint32_t objDataSize = static_cast<uint32_t>(effect.m_objectData.size());
    uint32_t matSize = static_cast<uint32_t>(
        perframeData.m_shaderEffectData[x].m_materials.size());
    for (int k = 0; k < matSize; k++) {
      auto mesh =
          perframeData.m_shaderEffectData[x].m_meshes[k]->loadMeshUnsafe();
      totalMeshlets += mesh->m_meshlets.size() * objDataSize;
    }
  }
  auto activeBuf =
      perframeData.m_allInstanceData.m_batchedObjectData->getActiveBuffer();
  activeBuf->map();
  activeBuf->writeBuffer(perframeData.m_allInstanceData.m_objectData.data(),
                         perframeData.m_allInstanceData.m_objectData.size() *
                             sizeof(PerObjectData),
                         0);
  activeBuf->flush();
  activeBuf->unmap();
  for (uint32_t k = 0; k < perframeData.m_views.size(); k++) {
    auto &perView = perframeData.m_views[k];
    if (perView.m_allFilteredMeshletsCount == nullptr) {
      perView.m_allFilteredMeshletsCount =
          rhi->createIndirectMeshDrawBufferDevice(
              4, RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT |
                     RhiBufferUsage::RHI_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    if (perView.m_allFilteredMeshletsMaxCount < totalMeshlets) {
      perView.m_allFilteredMeshletsMaxCount = totalMeshlets;
      perView.m_allFilteredMeshlets = rhi->createStorageBufferDevice(
          totalMeshlets * sizeof(uint32_t) * 2,
          RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);

      perView.m_allFilteredMeshletsDesc = rhi->createBindlessDescriptorRef();
      perView.m_allFilteredMeshletsDesc->addStorageBuffer(
          perView.m_allFilteredMeshlets, 0);
      perView.m_allFilteredMeshletsDesc->addStorageBuffer(
          perView.m_allFilteredMeshletsCount, 1);
    }
  }
}

IFRIT_APIDECL void
SyaroRenderer::prepareAggregatedShadowData(PerFrameData &perframeData) {
  auto &shadowData = perframeData.m_shadowData;
  auto rhi = m_app->getRhiLayer();
  if (shadowData.m_shadowViews.size() == 0) {
    shadowData.m_shadowViews.resize(256);
    shadowData.m_allShadowData = rhi->createStorageBufferDevice(
        256 * sizeof(decltype(shadowData.m_shadowViews)::value_type),
        RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    shadowData.m_allShadowDataId =
        rhi->registerStorageBuffer(shadowData.m_allShadowData);
  }
  for (auto d = 0; auto &x : perframeData.m_views) {
    if (x.m_viewType == PerFrameData::ViewType::Shadow) {
      shadowData.m_shadowViews[d].m_texRef = x.m_visDepthId->getActiveId();
      shadowData.m_shadowViews[d].m_viewRef = x.m_viewBufferId->getActiveId();
      d++;
    }
  }
  auto staged = rhi->createStagedSingleBuffer(shadowData.m_allShadowData);
  auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
  tq->runSyncCommand([&](const RhiCommandBuffer *cmd) {
    staged->cmdCopyToDevice(
        cmd, shadowData.m_shadowViews.data(),
        shadowData.m_shadowViews.size() *
            sizeof(decltype(shadowData.m_shadowViews)::value_type),
        0);
  });
}

IFRIT_APIDECL void
SyaroRenderer::taaHistorySetup(PerFrameData &perframeData,
                               RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();
  auto rtArea = renderTargets->getRenderArea();
  if (rtArea.x != 0 || rtArea.y != 0) {
    throw std::runtime_error(
        "Currently, it does not support non-zero x/y offsets");
  }
  auto width = rtArea.width + rtArea.x;
  auto height = rtArea.height + rtArea.y;

  auto needRecreate = (perframeData.m_taaHistory.size() == 0);
  if (!needRecreate) {
    needRecreate = (perframeData.m_taaHistory[0].m_width != width ||
                    perframeData.m_taaHistory[0].m_height != height);
  }
  if (!needRecreate) {
    return;
  }
  perframeData.m_taaHistory.clear();
  perframeData.m_taaHistory.resize(2);
  perframeData.m_taaHistory[0].m_width = width;
  perframeData.m_taaHistory[0].m_height = height;
  perframeData.m_taaSampler = rhi->createTrivialSampler();
  perframeData.m_taaHistorySampler = rhi->createTrivialSampler();
  perframeData.m_taaHistoryDesc = rhi->createBindlessDescriptorRef();
  auto rtFormat = renderTargets->getFormat();

  perframeData.m_taaUnresolved = rhi->createRenderTargetTexture(
      width, height, cTAAFormat,
      RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_SRC_BIT |
          RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);
  perframeData.m_taaHistoryDesc->addCombinedImageSampler(
      perframeData.m_taaUnresolved.get(), perframeData.m_taaSampler.get(), 0);
  for (int i = 0; i < 2; i++) {
    // TODO: choose formats

    perframeData.m_taaHistory[i].m_colorRT = rhi->createRenderTargetTexture(
        width, height, cTAAFormat,
        RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_SRC_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);
    perframeData.m_taaHistory[i].m_colorRTId =
        rhi->registerUAVImage(perframeData.m_taaUnresolved.get(), {0, 0, 1, 1});

    // TODO: clear values
    perframeData.m_taaHistory[i].m_colorRTRef = rhi->createRenderTarget(
        perframeData.m_taaUnresolved.get(), {{0, 0, 0, 0}},
        RhiRenderTargetLoadOp::Load, 0, 0);
    RhiAttachmentBlendInfo blendInfo;
    blendInfo.m_blendEnable = true;
    blendInfo.m_srcColorBlendFactor =
        RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
    blendInfo.m_dstColorBlendFactor =
        RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendInfo.m_colorBlendOp = RhiBlendOp::RHI_BLEND_OP_ADD;
    blendInfo.m_alphaBlendOp = RhiBlendOp::RHI_BLEND_OP_ADD;
    blendInfo.m_srcAlphaBlendFactor =
        RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
    blendInfo.m_dstAlphaBlendFactor =
        RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    perframeData.m_taaHistory[i].m_colorRTRef->setBlendInfo(blendInfo);

    perframeData.m_taaHistory[i].m_rts = rhi->createRenderTargets();
    perframeData.m_taaHistory[i].m_rts->setColorAttachments(
        {perframeData.m_taaHistory[i].m_colorRTRef.get()});
    perframeData.m_taaHistory[i].m_rts->setRenderArea(
        renderTargets->getRenderArea());

    perframeData.m_taaHistoryDesc->addCombinedImageSampler(
        perframeData.m_taaHistory[i].m_colorRT.get(),
        perframeData.m_taaHistorySampler.get(), i + 1);
  }
}

IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission>
SyaroRenderer::render(
    PerFrameData &perframeData, SyaroRenderer::RenderTargets *renderTargets,
    const std::vector<SyaroRenderer::GPUCommandSubmission *> &cmdToWait) {

  // According to
  // lunarg(https://www.lunasdk.org/manual/rhi/command_queues_and_command_buffers/)
  // graphics queue can submit dispatch and transfer commands, but compute queue
  // can only submit compute/transfer commands. Following posts suggests reduce
  // command buffer submission to improve performance
  // https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/
  // https://gpuopen.com/learn/rdna-performance-guide/#command-buffers

  visibilityBufferSetup(perframeData, renderTargets);
  auto &primaryView = getPrimaryView(perframeData);
  buildPipelines(perframeData, GraphicsShaderPassType::Opaque,
                 primaryView.m_visRTs.get());
  prepareDeviceResources(perframeData, renderTargets);
  gatherAllInstances(perframeData);
  recreateInstanceCullingBuffers(
      perframeData, perframeData.m_allInstanceData.m_objectData.size());
  hizBufferSetup(perframeData, renderTargets);
  depthTargetsSetup(perframeData, renderTargets);
  materialClassifyBufferSetup(perframeData, renderTargets);
  recreateGBuffers(perframeData, renderTargets);
  sphizBufferSetup(perframeData, renderTargets);
  taaHistorySetup(perframeData, renderTargets);
  prepareAggregatedShadowData(perframeData);

  // Then draw
  auto rhi = m_app->getRhiLayer();
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  m_textureShowPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->attachBindlessReferenceGraphics(
        m_textureShowPass, 1, perframeData.m_visShowCombinedRef);
    ctx->m_cmd->attachVertexBufferView(
        *rhi->getFullScreenQuadVertexBufferView());
    ctx->m_cmd->attachVertexBuffers(
        0, {rhi->getFullScreenQuadVertexBuffer().get()});
    ctx->m_cmd->drawInstanced(3, 1, 0, 0);
  });

  auto dq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);

  std::vector<RhiTaskSubmission *> cmdToWaitBkp = cmdToWait;
  std::unique_ptr<RhiTaskSubmission> pbrAtmoTask;
  if (perframeData.m_atmosphereData == nullptr) {
    // Need to create an atmosphere output texture
    perframeData.m_atmoOutput = rhi->createRenderTargetTexture(
        renderTargets->getRenderArea().width + renderTargets->getRenderArea().x,
        renderTargets->getRenderArea().height +
            renderTargets->getRenderArea().y,
        RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);

    perframeData.m_atmoOutputId =
        rhi->registerUAVImage(perframeData.m_atmoOutput.get(), {0, 0, 1, 1});

    // Precompute only once
    pbrAtmoTask =
        this->m_atmosphereRenderer->renderInternal(perframeData, cmdToWait);
    cmdToWaitBkp = {pbrAtmoTask.get()};
  }

  auto mainTask = dq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        printf("GPUTimer: %f ms/frame\n", m_timer->getElapsedMs());
        m_timer->start(cmd);
        renderTwoPassOcclCulling(CullingPass::First, perframeData,
                                 renderTargets, cmd,
                                 PerFrameData::ViewType::Primary, ~0u);
        cmd->globalMemoryBarrier();
        renderTwoPassOcclCulling(CullingPass::Second, perframeData,
                                 renderTargets, cmd,
                                 PerFrameData::ViewType::Primary, ~0u);
        cmd->globalMemoryBarrier();
        renderEmitDepthTargets(perframeData, renderTargets, cmd);
        cmd->globalMemoryBarrier();
        renderMaterialClassify(perframeData, renderTargets, cmd);
        cmd->globalMemoryBarrier();
        renderDefaultEmitGBuffer(perframeData, renderTargets, cmd);
        cmd->globalMemoryBarrier();
        renderAmbientOccl(perframeData, renderTargets, cmd);
        cmd->globalMemoryBarrier();

        for (uint32_t i = 0; i < perframeData.m_views.size(); i++) {
          auto &perView = perframeData.m_views[i];
          renderTwoPassOcclCulling(CullingPass::First, perframeData,
                                   renderTargets, cmd,
                                   PerFrameData::ViewType::Shadow, i);
          cmd->globalMemoryBarrier();
          renderTwoPassOcclCulling(CullingPass::Second, perframeData,
                                   renderTargets, cmd,
                                   PerFrameData::ViewType::Shadow, i);
          cmd->globalMemoryBarrier();
        }
        m_timer->stop(cmd);
      },
      cmdToWaitBkp, {});

  auto deferredTask = dq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        renderAtmosphere(perframeData, renderTargets, cmd);
        cmd->globalMemoryBarrier();
        renderDeferredShading(perframeData, renderTargets, cmd);
        cmd->globalMemoryBarrier();
        renderTAAResolve(perframeData, renderTargets, cmd);
        cmd->globalMemoryBarrier();
        renderToneMapping(perframeData, renderTargets, cmd);
      },
      {mainTask.get()}, {});

  return deferredTask;
}

IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission>
SyaroRenderer::render(Scene *scene, Camera *camera,
                      RenderTargets *renderTargets,
                      const RendererConfig &config,
                      const std::vector<GPUCommandSubmission *> &cmdToWait) {
  if (m_perScenePerframe.count(scene) == 0) {
    m_perScenePerframe[scene] = PerFrameData();
  }
  m_renderConfig = config;
  auto &perframeData = m_perScenePerframe[scene];
  auto frameId = perframeData.m_frameId;

  auto haltonX =
      RendererConsts::cHalton2[frameId % RendererConsts::cHalton2.size()];
  auto haltonY =
      RendererConsts::cHalton3[frameId % RendererConsts::cHalton3.size()];
  auto renderArea = renderTargets->getRenderArea();
  auto width = renderArea.width + renderArea.x;
  auto height = renderArea.height + renderArea.y;
  SceneCollectConfig sceneConfig;
  if (config.m_antiAliasingType == AntiAliasingType::TAA) {
    sceneConfig.projectionTranslateX = (haltonX * 2.0f - 1.0f) / width;
    sceneConfig.projectionTranslateY = (haltonY * 2.0f - 1.0f) / height;
  } else {
    sceneConfig.projectionTranslateX = 0.0f;
    sceneConfig.projectionTranslateY = 0.0f;
  }
  collectPerframeData(perframeData, scene, camera,
                      GraphicsShaderPassType::Opaque, sceneConfig);

  perframeData.m_taaJitterX = sceneConfig.projectionTranslateX * 0.5;
  perframeData.m_taaJitterY = sceneConfig.projectionTranslateY * 0.5;
  auto ret = render(perframeData, renderTargets, cmdToWait);
  perframeData.m_frameId++;
  return ret;
}

} // namespace Ifrit::Core