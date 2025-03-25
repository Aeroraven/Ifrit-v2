
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

#include "ifrit/common/base/IfritBase.h"

#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/renderer/RendererUtil.h"
#include "ifrit/core/renderer/SyaroRenderer.h"
#include "ifrit/core/renderer/util/RenderingUtils.h"
#include <algorithm>
#include <bit>

#include "ifrit.shader/Syaro/Syaro.SharedConst.h"

using namespace Ifrit::GraphicsBackend::Rhi;
using Ifrit::Common::Utility::size_cast;
using Ifrit::Math::ConstFunc::divRoundUp;

// Frequently used image usages
IF_CONSTEXPR auto kbImUsage_UAV_SRV = RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT;
IF_CONSTEXPR auto kbImUsage_UAV_SRV_CopyDest =
    RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT | RHI_IMAGE_USAGE_TRANSFER_DST_BIT;
IF_CONSTEXPR auto kbImUsage_UAV_SRV_RT_CopySrc = RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT |
                                                 RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                 RHI_IMAGE_USAGE_TRANSFER_SRC_BIT;
IF_CONSTEXPR auto kbImUsage_UAV_SRV_RT =
    RHI_IMAGE_USAGE_STORAGE_BIT | RHI_IMAGE_USAGE_SAMPLED_BIT | RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
IF_CONSTEXPR auto kbImUsage_SRV_DEPTH = RHI_IMAGE_USAGE_SAMPLED_BIT | RHI_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
IF_CONSTEXPR auto kbImUsage_UAV = RHI_IMAGE_USAGE_STORAGE_BIT;

// Frequently used buffer usages
IF_CONSTEXPR auto kbBufUsage_Indirect = RhiBufferUsage_Indirect | RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO;
IF_CONSTEXPR auto kbBufUsage_SSBO_CopyDest = RhiBufferUsage_SSBO | RhiBufferUsage_CopyDst;
IF_CONSTEXPR auto kbBufUsage_SSBO = RhiBufferUsage_SSBO;

// Frequently used image fmts
IF_CONSTEXPR auto kbImFmt_RGBA32F = RHI_FORMAT_R32G32B32A32_SFLOAT;
IF_CONSTEXPR auto kbImFmt_RGBA16F = RHI_FORMAT_R16G16B16A16_SFLOAT;
IF_CONSTEXPR auto kbImFmt_RG32F = RHI_FORMAT_R32G32_SFLOAT;
IF_CONSTEXPR auto kbImFmt_R32F = RHI_FORMAT_R32_SFLOAT;

namespace Ifrit::Core {

struct GPUHiZDesc {
  u32 m_width;
  u32 m_height;
};

std::vector<RhiResourceBarrier> registerUAVBarriers(const std::vector<RhiBuffer *> &buffers,
                                                    const std::vector<RhiTexture *> &textures) {
  std::vector<RhiResourceBarrier> barriers;
  for (auto &buffer : buffers) {
    RhiUAVBarrier barrier;
    barrier.m_type = RhiResourceType::Buffer;
    barrier.m_buffer = buffer;
    RhiResourceBarrier resBarrier;
    resBarrier.m_type = RhiBarrierType::UAVAccess;
    resBarrier.m_uav = barrier;
    barriers.push_back(resBarrier);
  }
  for (auto &texture : textures) {
    RhiUAVBarrier barrier;
    barrier.m_type = RhiResourceType::Texture;
    barrier.m_texture = texture;
    RhiResourceBarrier resBarrier;
    resBarrier.m_type = RhiBarrierType::UAVAccess;
    resBarrier.m_uav = barrier;
    barriers.push_back(resBarrier);
  }
  return barriers;
}

void runImageBarrier(const RhiCommandList *cmd, RhiTexture *texture, RhiResourceState dst,
                     RhiImageSubResource subResource) {
  std::vector<RhiResourceBarrier> barriers;
  RhiTransitionBarrier barrier;
  barrier.m_type = RhiResourceType::Texture;
  barrier.m_texture = texture;
  barrier.m_srcState = RhiResourceState::AutoTraced;
  barrier.m_dstState = dst;
  barrier.m_subResource = subResource;

  RhiResourceBarrier resBarrier;
  resBarrier.m_type = RhiBarrierType::Transition;
  resBarrier.m_transition = barrier;

  barriers.push_back(resBarrier);
  cmd->resourceBarrier(barriers);
}

void runUAVBufferBarrier(const RhiCommandList *cmd, RhiBuffer *buffer) {
  std::vector<RhiResourceBarrier> barriers;
  RhiUAVBarrier barrier;
  barrier.m_type = RhiResourceType::Buffer;
  barrier.m_buffer = buffer;
  RhiResourceBarrier resBarrier;
  resBarrier.m_type = RhiBarrierType::UAVAccess;
  resBarrier.m_uav = barrier;
  barriers.push_back(resBarrier);
  cmd->resourceBarrier(barriers);
}

RhiScissor getSupersampleDownsampledArea(const RhiRenderTargets *finalRenderTargets, const RendererConfig &cfg) {
  RhiScissor scissor;
  scissor.x = 0;
  scissor.y = 0;
  scissor.width = finalRenderTargets->getRenderArea().width / cfg.m_superSamplingRate;
  scissor.height = finalRenderTargets->getRenderArea().height / cfg.m_superSamplingRate;
  return scissor;
}

// end of util functions

IFRIT_APIDECL PerFrameData::PerViewData &SyaroRenderer::getPrimaryView(PerFrameData &perframeData) {
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
  auto deferTimer = rhi->createDeviceTimer();
  m_timerDefer = deferTimer;
}

IFRIT_APIDECL SyaroRenderer::GPUShader *
SyaroRenderer::createShaderFromFile(const std::string &shaderPath, const std::string &entry,
                                    GraphicsBackend::Rhi::RhiShaderStage stage) {
  auto rhi = m_app->getRhiLayer();
  std::string shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  auto path = shaderBasePath + "/Syaro/" + shaderPath;
  auto shaderCode = Ifrit::Common::Utility::readTextFile(path);
  std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
  return rhi->createShader(shaderPath, shaderCodeVec, entry, stage, RhiShaderSourceType::GLSLCode);
}

IFRIT_APIDECL void SyaroRenderer::setupPostprocessPassAndTextures() {
  // passes
  m_acesToneMapping = std::make_unique<PostprocessPassCollection::PostFxAcesToneMapping>(m_app);
  m_globalFogPass = std::make_unique<PostprocessPassCollection::PostFxGlobalFog>(m_app);
  m_gaussianHori = std::make_unique<PostprocessPassCollection::PostFxGaussianHori>(m_app);
  m_gaussianVert = std::make_unique<PostprocessPassCollection::PostFxGaussianVert>(m_app);
  m_fftConv2d = std::make_unique<PostprocessPassCollection::PostFxFFTConv2d>(m_app);

  m_jointBilateralFilter = std::make_unique<PostprocessPassCollection::PostFxJointBilaterialFilter>(m_app);

  // tex and samplers
  m_postprocTexSampler = m_app->getRhiLayer()->createTrivialSampler();

  // fsr2
  m_fsr2proc = m_app->getRhiLayer()->createFsr2Processor();
}

IFRIT_APIDECL void SyaroRenderer::createPostprocessTextures(u32 width, u32 height) {
  auto rhi = m_app->getRhiLayer();
  auto rtFmt = kbImFmt_RGBA32F;
  if (m_postprocTex.find({width, height}) != m_postprocTex.end()) {
    return;
  }
  for (u32 i = 0; i < 2; i++) {
    auto tex = rhi->createTexture2D("Syaro_PostprocTex", width, height, rtFmt, kbImUsage_UAV_SRV_RT);
    auto colorRT = rhi->createRenderTarget(tex.get(), {{0.0f, 0.0f, 0.0f, 1.0f}}, RhiRenderTargetLoadOp::Load, 0, 0);
    auto rts = rhi->createRenderTargets();

    m_postprocTex[{width, height}][i] = tex;
    m_postprocTexId[{width, height}][i] = rhi->registerCombinedImageSampler(tex.get(), m_postprocTexSampler.get());
    m_postprocTexIdComp[{width, height}][i] = rhi->registerUAVImage(tex.get(), {0, 0, 1, 1});
    m_postprocColorRT[{width, height}][i] = colorRT;

    RhiAttachmentBlendInfo blendInfo;
    blendInfo.m_blendEnable = true;
    blendInfo.m_srcColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
    blendInfo.m_dstColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendInfo.m_colorBlendOp = RhiBlendOp::RHI_BLEND_OP_ADD;
    blendInfo.m_alphaBlendOp = RhiBlendOp::RHI_BLEND_OP_ADD;
    blendInfo.m_srcAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
    blendInfo.m_dstAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    rts->setColorAttachments({colorRT.get()});
    rts->setRenderArea({0, 0, width, height});
    colorRT->setBlendInfo(blendInfo);
    m_postprocRTs[{width, height}][i] = rts;
  }
}

IFRIT_APIDECL void SyaroRenderer::setupAndRunFrameGraph(PerFrameData &perframeData, RenderTargets *renderTargets,
                                                        const GPUCmdBuffer *cmd) {
  // some pipelines
  if (m_deferredShadowPass == nullptr) {
    auto rhi = m_app->getRhiLayer();
    auto vsShader = createShaderFromFile("Syaro.DeferredShadow.vert.glsl", "main", RhiShaderStage::Vertex);
    auto fsShader = createShaderFromFile("Syaro.DeferredShadow.frag.glsl", "main", RhiShaderStage::Fragment);
    auto shadowRtCfg = renderTargets->getFormat();
    shadowRtCfg.m_colorFormats = {RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT};
    shadowRtCfg.m_depthFormat = RhiImageFormat::RHI_FORMAT_UNDEFINED;

    m_deferredShadowPass = rhi->createGraphicsPass();
    m_deferredShadowPass->setVertexShader(vsShader);
    m_deferredShadowPass->setPixelShader(fsShader);
    m_deferredShadowPass->setNumBindlessDescriptorSets(3);
    m_deferredShadowPass->setPushConstSize(3 * u32Size);
    m_deferredShadowPass->setRenderTargetFormat(shadowRtCfg);
  }

  // declare frame graph
  FrameGraph fg;

  std::vector<ResourceNodeId> resShadowMapTexs;
  std::vector<u32> shadowMapTexIds;
  for (auto id = 0; auto &view : perframeData.m_views) {
    if (view.m_viewType == PerFrameData::ViewType::Shadow) {
      auto resId = fg.addResource("ShadowMapTex" + std::to_string(id));
      resShadowMapTexs.push_back(resId);
      shadowMapTexIds.push_back(id);
    }
    id++;
  }
  auto resAtmosphereOutput = fg.addResource("AtmosphereOutput");

  auto resGbufferAlbedoMaterial = fg.addResource("GbufferAlbedoMaterial");
  auto resGbufferNormalSmoothness = fg.addResource("GbufferNormalSmoothness");
  auto resGbufferSpecularAO = fg.addResource("GbufferSpecularAO");
  auto resMotionDepth = fg.addResource("MotionDepth");

  auto resDeferredShadowOutput = fg.addResource("DeferredShadowOutput");
  auto resBlurredShadowIntermediateOutput = fg.addResource("BlurredShadowIntermediateOutput");
  auto resBlurredShadowOutput = fg.addResource("BlurredShadowOutput");

  auto resPrimaryViewDepth = fg.addResource("PrimaryViewDepth");
  auto resDeferredShadingOutput = fg.addResource("DeferredShadingOutput");
  auto resGlobalFogOutput = fg.addResource("GlobalFogOutput");
  auto resTAAFrameOutput = fg.addResource("TAAFrameOutput");
  auto resTAAHistoryOutput = fg.addResource("TAAHistoryOutput");
  auto resFinalOutput = fg.addResource("FinalOutput");

  auto resBloomOutput = fg.addResource("BloomOutput");
  auto resFsr2Output = fg.addResource("Fsr2Output");

  std::vector<ResourceNodeId> resGbufferShadowReq = {resGbufferAlbedoMaterial, resGbufferNormalSmoothness,
                                                     resGbufferSpecularAO, resMotionDepth, resPrimaryViewDepth};
  for (auto &x : resShadowMapTexs) {
    resGbufferShadowReq.push_back(x);
  }

  std::vector<ResourceNodeId> resGbuffer = resGbufferShadowReq;
  resGbuffer.push_back(resBlurredShadowOutput);

  // add passes
  auto passAtmosphere =
      fg.addPass("Atmosphere", FrameGraphPassType::Graphics, {resPrimaryViewDepth}, {resAtmosphereOutput}, {});

  auto passDeferredShadow = fg.addPass("DeferredShadow", FrameGraphPassType::Graphics, resGbufferShadowReq,
                                       {resDeferredShadowOutput}, {resAtmosphereOutput});

  auto passBlurShadowHori = fg.addPass("BlurShadowHori", FrameGraphPassType::Graphics, {resDeferredShadowOutput},
                                       {resBlurredShadowIntermediateOutput}, {});

  auto passBlurShadowVert = fg.addPass("BlurShadowVert", FrameGraphPassType::Graphics,
                                       {resBlurredShadowIntermediateOutput}, {resBlurredShadowOutput}, {});

  auto passDeferredShading = fg.addPass("DeferredShading", FrameGraphPassType::Graphics, resGbuffer,
                                        {resDeferredShadingOutput}, {resAtmosphereOutput});
  PassNodeId passGlobalFog;
  PassNodeId passTAAResolve;
  PassNodeId passConvBloom;
  PassNodeId passFsr2Dispatch = 0;
  if (m_config->m_antiAliasingType == AntiAliasingType::FSR2) {
    passGlobalFog = fg.addPass("GlobalFog", FrameGraphPassType::Graphics,
                               {resPrimaryViewDepth, resDeferredShadingOutput}, {resGlobalFogOutput}, {});
    passConvBloom = fg.addPass("ConvBloom", FrameGraphPassType::Graphics, {resGlobalFogOutput}, {resBloomOutput}, {});
    passFsr2Dispatch = fg.addPass("FSR2Dispatch", FrameGraphPassType::Compute, {resBloomOutput}, {resFsr2Output}, {});
  } else if (m_config->m_antiAliasingType == AntiAliasingType::TAA) {
    passGlobalFog = fg.addPass("GlobalFog", FrameGraphPassType::Graphics,
                               {resPrimaryViewDepth, resDeferredShadingOutput}, {resGlobalFogOutput}, {});
    passTAAResolve = fg.addPass("TAAResolve", FrameGraphPassType::Graphics, {resGlobalFogOutput},
                                {resTAAFrameOutput, resTAAHistoryOutput}, {});
    passConvBloom = fg.addPass("ConvBloom", FrameGraphPassType::Graphics, {resTAAFrameOutput}, {resBloomOutput}, {});

  } else {
    passGlobalFog = fg.addPass("GlobalFog", FrameGraphPassType::Graphics,
                               {resPrimaryViewDepth, resDeferredShadingOutput}, {resGlobalFogOutput}, {});
    passConvBloom = fg.addPass("ConvBloom", FrameGraphPassType::Graphics, {resGlobalFogOutput}, {resBloomOutput}, {});
  }
  PassNodeId passToneMapping;
  if (m_config->m_antiAliasingType == AntiAliasingType::FSR2) {
    passToneMapping = fg.addPass("ToneMapping", FrameGraphPassType::Graphics, {resFsr2Output}, {resFinalOutput}, {});
  } else {
    passToneMapping = fg.addPass("ToneMapping", FrameGraphPassType::Graphics, {resBloomOutput}, {resFinalOutput}, {});
  }

  // binding resources to pass
  auto &primaryView = getPrimaryView(perframeData);
  auto rhi = m_app->getRhiLayer();
  auto mainRtWidth = primaryView.m_renderWidth;
  auto mainRtHeight = primaryView.m_renderHeight;
  createPostprocessTextures(mainRtWidth, mainRtHeight);

  fg.setImportedResource(resAtmosphereOutput, m_postprocTex[{mainRtWidth, mainRtHeight}][0].get(), {0, 0, 1, 1});
  fg.setImportedResource(resGbufferAlbedoMaterial, perframeData.m_gbuffer.m_albedo_materialFlags.get(), {0, 0, 1, 1});
  fg.setImportedResource(resGbufferNormalSmoothness, perframeData.m_gbuffer.m_normal_smoothness.get(), {0, 0, 1, 1});
  fg.setImportedResource(resGbufferSpecularAO, perframeData.m_gbuffer.m_specular_occlusion.get(), {0, 0, 1, 1});
  fg.setImportedResource(resMotionDepth, perframeData.m_velocityMaterial.get(), {0, 0, 1, 1});
  for (auto i = 0; auto &x : resShadowMapTexs) {
    auto viewId = shadowMapTexIds[i];
    fg.setImportedResource(x, perframeData.m_views[viewId].m_visibilityDepth_Combined.get(), {0, 0, 1, 1});
    i++;
  }
  fg.setImportedResource(resDeferredShadowOutput, perframeData.m_deferShadowMask.get(), {0, 0, 1, 1});

  fg.setImportedResource(resBlurredShadowIntermediateOutput, m_postprocTex[{mainRtWidth, mainRtHeight}][1].get(),
                         {0, 0, 1, 1});

  fg.setImportedResource(resBlurredShadowOutput, perframeData.m_deferShadowMask.get(), {0, 0, 1, 1});

  fg.setImportedResource(resPrimaryViewDepth, primaryView.m_visibilityDepth_Combined.get(), {0, 0, 1, 1});
  fg.setImportedResource(resDeferredShadingOutput, m_postprocTex[{mainRtWidth, mainRtHeight}][0].get(), {0, 0, 1, 1});
  fg.setImportedResource(resGlobalFogOutput, perframeData.m_taaUnresolved.get(), {0, 0, 1, 1});
  fg.setImportedResource(resTAAFrameOutput, m_postprocTex[{mainRtWidth, mainRtHeight}][0].get(), {0, 0, 1, 1});
  fg.setImportedResource(resTAAHistoryOutput, perframeData.m_taaHistory[perframeData.m_frameId % 2].m_colorRT.get(),
                         {0, 0, 1, 1});
  fg.setImportedResource(resBloomOutput, m_postprocTex[{mainRtWidth, mainRtHeight}][1].get(), {0, 0, 1, 1});
  fg.setImportedResource(resFinalOutput, renderTargets->getColorAttachment(0)->getRenderTarget(), {0, 0, 1, 1});

  if (m_config->m_antiAliasingType == AntiAliasingType::FSR2) {
    fg.setImportedResource(resFsr2Output, perframeData.m_fsr2Data.m_fsr2Output.get(), {0, 0, 1, 1});
  }

  // setup execution function
  fg.setExecutionFunction(passAtmosphere, [&]() {
    auto postprocId = m_postprocTexIdComp[{mainRtWidth, mainRtHeight}][0];
    auto postprocTex = m_postprocTex[{mainRtWidth, mainRtHeight}][0];
    auto atmoData = m_atmosphereRenderer->getResourceDesc(perframeData);
    auto rhi = m_app->getRhiLayer();
    struct AtmoPushConst {
      ifloat4 sundir;
      u32 m_perframe;
      u32 m_outTex;
      u32 m_depthTex;
      u32 pad1;
      decltype(atmoData) m_atmoData;
    } pushConst;
    pushConst.sundir = perframeData.m_sunDir;
    pushConst.m_perframe = primaryView.m_viewBufferId->getActiveId();
    pushConst.m_outTex = postprocId->getActiveId();
    pushConst.m_atmoData = atmoData;
    pushConst.m_depthTex = primaryView.m_visibilityDepthIdSRV_Combined->getActiveId();
    m_atmospherePass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
      ctx->m_cmd->setPushConst(m_atmospherePass, 0, sizeof(AtmoPushConst), &pushConst);
      auto wgX = Math::ConstFunc::divRoundUp(primaryView.m_renderWidth, SyaroConfig::cAtmoRenderThreadGroupSizeX);
      auto wgY = Math::ConstFunc::divRoundUp(primaryView.m_renderHeight, SyaroConfig::cAtmoRenderThreadGroupSizeY);
      ctx->m_cmd->dispatch(wgX, wgY, 1);
    });
    cmd->beginScope("Syaro: Atmosphere");
    m_atmospherePass->run(cmd, 0);
    cmd->endScope();
  });
  fg.setExecutionFunction(passDeferredShadow, [&]() {
    struct DeferPushConst {
      u32 shadowMapDataRef;
      u32 numShadowMaps;
      u32 depthTexRef;
    } pc;
    pc.numShadowMaps = perframeData.m_shadowData2.m_enabledShadowMaps;
    pc.shadowMapDataRef = perframeData.m_shadowData2.m_allShadowDataId->getActiveId();
    pc.depthTexRef = primaryView.m_visibilityDepthIdSRV_Combined->getActiveId();
    cmd->beginScope("Syaro: Deferred Shadowing");

    auto targetRT = perframeData.m_deferShadowMaskRTs.get();
    RenderingUtil::enqueueFullScreenPass(
        cmd, rhi, m_deferredShadowPass, targetRT,
        {perframeData.m_gbufferDescFrag, perframeData.m_gbufferDepthDesc, primaryView.m_viewBindlessRef}, &pc, 3);
    cmd->endScope();
  });

  fg.setExecutionFunction(passBlurShadowHori, [&]() {
    auto postprocRTs = m_postprocRTs[{mainRtWidth, mainRtHeight}];
    auto postprocRT1 = postprocRTs[1];
    auto deferShadowId = perframeData.m_deferShadowMaskId.get();
    m_gaussianHori->renderPostFx(cmd, postprocRT1.get(), deferShadowId, 3);
  });

  fg.setExecutionFunction(passBlurShadowVert, [&]() {
    auto postprocId = m_postprocTexId[{mainRtWidth, mainRtHeight}][1];
    m_gaussianVert->renderPostFx(cmd, perframeData.m_deferShadowMaskRTs.get(), postprocId.get(), 3);
  });

  fg.setExecutionFunction(passDeferredShading, [&]() {
    setupDeferredShadingPass(renderTargets);

    auto postprocRTs = m_postprocRTs[{mainRtWidth, mainRtHeight}];
    auto postprocRT0 = postprocRTs[0];
    auto curRT = perframeData.m_taaHistory[perframeData.m_frameId % 2].m_rts;
    PipelineAttachmentConfigs paCfg;
    auto rtCfg = curRT->getFormat();
    paCfg.m_colorFormats = rtCfg.m_colorFormats;
    paCfg.m_depthFormat = RhiImageFormat::RHI_FORMAT_UNDEFINED;
    auto pass = m_deferredShadingPass[paCfg];
    struct DeferPushConst {
      ifloat4 sundir;
      u32 shadowMapDataRef;
      u32 numShadowMaps;
      u32 depthTexRef;
      u32 shadowTexRef;
    } pc;
    pc.sundir = perframeData.m_sunDir;
    pc.numShadowMaps = perframeData.m_shadowData2.m_enabledShadowMaps;
    pc.shadowMapDataRef = perframeData.m_shadowData2.m_allShadowDataId->getActiveId();
    pc.depthTexRef = primaryView.m_visibilityDepthIdSRV_Combined->getActiveId();
    pc.shadowTexRef = perframeData.m_deferShadowMaskId->getActiveId();
    cmd->beginScope("Syaro: Deferred Shading");
    RenderingUtil::enqueueFullScreenPass(
        cmd, rhi, pass, postprocRT0.get(),
        {perframeData.m_gbufferDescFrag, perframeData.m_gbufferDepthDesc, primaryView.m_viewBindlessRef}, &pc, 8);
    cmd->endScope();
  });
  fg.setExecutionFunction(passGlobalFog, [&]() {
    auto fogRT = perframeData.m_taaHistory[perframeData.m_frameId % 2].m_rts;
    auto inputId = m_postprocTexId[{mainRtWidth, mainRtHeight}][0].get();
    auto inputDepthId = primaryView.m_visibilityDepthIdSRV_Combined.get();
    auto primaryViewId = primaryView.m_viewBufferId.get();
    m_globalFogPass->renderPostFx(cmd, fogRT.get(), inputId, inputDepthId, primaryViewId);
  });

  if (m_config->m_antiAliasingType == AntiAliasingType::TAA) {
    fg.setExecutionFunction(passTAAResolve, [&]() {
      auto taaRT = rhi->createRenderTargets();
      auto taaCurTarget = rhi->createRenderTarget(perframeData.m_taaHistory[perframeData.m_frameId % 2].m_colorRT.get(),
                                                  {}, RhiRenderTargetLoadOp::Clear, 0, 0);
      auto taaRenderTarget = m_postprocColorRT[{mainRtWidth, mainRtHeight}][0].get();

      taaRT->setColorAttachments({taaCurTarget.get(), taaRenderTarget});
      taaRT->setDepthStencilAttachment(nullptr);
      taaRT->setRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));
      setupTAAPass(renderTargets);
      auto rtCfg = taaRT->getFormat();
      PipelineAttachmentConfigs paCfg;
      paCfg.m_colorFormats = rtCfg.m_colorFormats;
      paCfg.m_depthFormat = RhiImageFormat::RHI_FORMAT_UNDEFINED;
      auto pass = m_taaPass[paCfg];
      u32 jitterX = std::bit_cast<u32, float>(perframeData.m_taaJitterX);
      u32 jitterY = std::bit_cast<u32, float>(perframeData.m_taaJitterY);
      u32 pconst[5] = {
          perframeData.m_frameId, mainRtWidth, mainRtHeight, jitterX, jitterY,
      };
      cmd->beginScope("Syaro: TAA Resolve");
      RenderingUtil::enqueueFullScreenPass(cmd, rhi, pass, taaRT.get(),
                                           {perframeData.m_taaHistoryDesc, perframeData.m_gbufferDepthDesc}, pconst, 3);
      cmd->endScope();
    });
    fg.setExecutionFunction(passConvBloom, [&]() {
      cmd->beginScope("Syaro: Convolution Bloom");
      auto width = mainRtWidth;
      auto height = mainRtHeight;
      auto postprocTex0Id = m_postprocTexId[{width, height}][0].get();
      auto postprocTex1Id = m_postprocTexIdComp[{width, height}][1].get();
      m_fftConv2d->renderPostFx(cmd, postprocTex0Id, postprocTex1Id, nullptr, width, height, 51, 51, 4);
      cmd->endScope();
    });
  } else {
    fg.setExecutionFunction(passConvBloom, [&]() {
      cmd->beginScope("Syaro: Convolution Bloom");
      auto width = mainRtWidth;
      auto height = mainRtHeight;
      auto postprocTex0Id = perframeData.m_taaHistory[perframeData.m_frameId % 2].m_colorRTIdSRV.get();
      auto postprocTex1Id = m_postprocTexIdComp[{width, height}][1].get();
      m_fftConv2d->renderPostFx(cmd, postprocTex0Id, postprocTex1Id, nullptr, width, height, 51, 51, 4);
      cmd->endScope();
    });
  }

  // TODO, place fsr2 here
  if (m_config->m_antiAliasingType == AntiAliasingType::FSR2) {
    fg.setExecutionFunction(passFsr2Dispatch, [&]() {
      cmd->beginScope("Syaro: FSR2 Dispatch");
      auto color = m_postprocTex[{mainRtWidth, mainRtHeight}][0].get();
      auto depth = primaryView.m_visibilityDepth_Combined.get();
      auto motion = perframeData.m_motionVector.get();

      auto mainView = getPrimaryView(perframeData);
      GraphicsBackend::Rhi::FSR2::RhiFSR2DispatchArgs args;
      args.camFar = mainView.m_viewData.m_cameraFar;
      args.camNear = mainView.m_viewData.m_cameraNear;
      args.camFovY = mainView.m_viewData.m_cameraFovY;
      args.color = color;
      args.depth = depth;
      if (perframeData.m_frameId < 2) {
        args.deltaTime = 16.6f;
      } else {
        // TODO: precision loss
        args.deltaTime = perframeData.m_frameTimestamp[perframeData.m_frameId % 2] -
                         perframeData.m_frameTimestamp[(perframeData.m_frameId - 1) % 2];
        args.deltaTime = std::max(args.deltaTime, 16.6f);
      }
      args.exposure = nullptr;
      args.jitterX = perframeData.m_taaJitterX;
      args.jitterY = perframeData.m_taaJitterY;
      args.motion = motion;
      args.reactiveMask = nullptr;
      args.transparencyMask = nullptr;
      args.output = perframeData.m_fsr2Data.m_fsr2Output.get();

      args.reset = primaryView.m_camMoved;

      if (args.output == nullptr) {
        throw std::runtime_error("FSR2 output is null");
      }

      cmd->beginScope("Syaro: FSR2 Dispatch, Impl");
      m_fsr2proc->dispatch(cmd, args);
      cmd->endScope();
      cmd->endScope();

      // make output transist to render target
    });
  }

  if (m_config->m_antiAliasingType == AntiAliasingType::FSR2) {
    auto renderArea = renderTargets->getRenderArea();
    auto width = renderArea.width;
    auto height = renderArea.height;

    fg.setExecutionFunction(passToneMapping, [&]() {
      m_acesToneMapping->renderPostFx(cmd, renderTargets, perframeData.m_fsr2Data.m_fsr2OutputSRVId.get());
    });
  } else {
    fg.setExecutionFunction(passToneMapping, [&]() {
      m_acesToneMapping->renderPostFx(cmd, renderTargets, m_postprocTexId[{mainRtWidth, mainRtHeight}][0].get());
    });
  }

  // transition input resources
  if (m_config->m_antiAliasingType == AntiAliasingType::FSR2) {
    runImageBarrier(cmd, perframeData.m_fsr2Data.m_fsr2Output.get(), RhiResourceState::Common, {0, 0, 1, 1});

    auto motion = perframeData.m_motionVector.get();
    auto primaryView = getPrimaryView(perframeData);
    auto depth = primaryView.m_visibilityDepth_Combined.get();

    runImageBarrier(cmd, motion, RhiResourceState::ShaderRead, {0, 0, 1, 1});
    runImageBarrier(cmd, depth, RhiResourceState::ShaderRead, {0, 0, 1, 1});
  }
  runImageBarrier(cmd, m_postprocTex[{mainRtWidth, mainRtHeight}][0].get(), RhiResourceState::ColorRT, {0, 0, 1, 1});
  runImageBarrier(cmd, m_postprocTex[{mainRtWidth, mainRtHeight}][1].get(), RhiResourceState::ColorRT, {0, 0, 1, 1});

  // run!
  FrameGraphCompiler compiler;
  auto compiledGraph = compiler.compile(fg);
  FrameGraphExecutor executor;

  cmd->beginScope("Syaro: Deferred Shading");
  executor.executeInSingleCmd(cmd, compiledGraph);
  cmd->endScope();
}

IFRIT_APIDECL void SyaroRenderer::setupPbrAtmosphereRenderer() {
  m_atmosphereRenderer = std::make_shared<PbrAtmosphereRenderer>(m_app);
  auto rhi = m_app->getRhiLayer();
  m_atmospherePass = RenderingUtil::createComputePass(rhi, "Syaro/Syaro.PbrAtmoRender.comp.glsl", 0, 19);
}

IFRIT_APIDECL void SyaroRenderer::setupDeferredShadingPass(RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();

  // This seems to be a bit of redundant code
  // The rhi backend now can reference the pipeline with similar
  // CI

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
    auto vsShader = createShaderFromFile("Syaro.DeferredShading.vert.glsl", "main", RhiShaderStage::Vertex);
    auto fsShader = createShaderFromFile("Syaro.DeferredShading.frag.glsl", "main", RhiShaderStage::Fragment);
    pass->setVertexShader(vsShader);
    pass->setPixelShader(fsShader);
    pass->setNumBindlessDescriptorSets(3);
    pass->setPushConstSize(8 * u32Size);
    pass->setRenderTargetFormat(rtCfg);
    m_deferredShadingPass[paCfg] = pass;
  }
}

IFRIT_APIDECL void SyaroRenderer::setupDebugPasses(PerFrameData &perframeData, RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();
  auto rtCfg = renderTargets->getFormat();

  PipelineAttachmentConfigs paCfg;
  paCfg.m_colorFormats = rtCfg.m_colorFormats;
  paCfg.m_depthFormat = rtCfg.m_depthFormat;

  if (m_triangleViewPass.find(paCfg) == m_triangleViewPass.end()) {
    auto pass = rhi->createGraphicsPass();
    auto vsShader = createShaderFromFile("Syaro.TriangleView.vert.glsl", "main", RhiShaderStage::Vertex);
    auto fsShader = createShaderFromFile("Syaro.TriangleView.frag.glsl", "main", RhiShaderStage::Fragment);
    pass->setVertexShader(vsShader);
    pass->setPixelShader(fsShader);
    pass->setNumBindlessDescriptorSets(0);
    pass->setPushConstSize(u32Size);
    pass->setRenderTargetFormat(rtCfg);

    m_triangleViewPass[paCfg] = pass;
  }
}

IFRIT_APIDECL void SyaroRenderer::setupTAAPass(RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();

  PipelineAttachmentConfigs paCfg;
  auto rtCfg = renderTargets->getFormat();
  paCfg.m_colorFormats = {cTAAFormat, RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT};
  paCfg.m_depthFormat = RhiImageFormat::RHI_FORMAT_UNDEFINED;
  rtCfg.m_colorFormats = paCfg.m_colorFormats;

  DrawPass *pass = nullptr;
  if (m_taaPass.find(paCfg) != m_taaPass.end()) {
    pass = m_taaPass[paCfg];
  } else {
    pass = rhi->createGraphicsPass();
    auto vsShader = createShaderFromFile("Syaro.TAA.vert.glsl", "main", RhiShaderStage::Vertex);
    auto fsShader = createShaderFromFile("Syaro.TAA.frag.glsl", "main", RhiShaderStage::Fragment);
    pass->setVertexShader(vsShader);
    pass->setPixelShader(fsShader);
    pass->setNumBindlessDescriptorSets(2);
    pass->setPushConstSize(u32Size * 5);
    rtCfg.m_depthFormat = RhiImageFormat::RHI_FORMAT_UNDEFINED;
    paCfg.m_depthFormat = RhiImageFormat::RHI_FORMAT_UNDEFINED;
    pass->setRenderTargetFormat(rtCfg);
    m_taaPass[paCfg] = pass;
  }
}

IFRIT_APIDECL void SyaroRenderer::setupVisibilityPass() {
  auto rhi = m_app->getRhiLayer();

  // Hardware Rasterize
  if constexpr (true) {
    auto tsShader = createShaderFromFile("Syaro.VisBuffer.task.glsl", "main", RhiShaderStage::Task);
    auto msShader = createShaderFromFile("Syaro.VisBuffer.mesh.glsl", "main", RhiShaderStage::Mesh);
    auto msShaderDepth = createShaderFromFile("Syaro.VisBufferDepth.mesh.glsl", "main", RhiShaderStage::Mesh);
    auto fsShader = createShaderFromFile("Syaro.VisBuffer.frag.glsl", "main", RhiShaderStage::Fragment);

    m_visibilityPassHW = rhi->createGraphicsPass();
#if !SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL
    m_visibilityPassHW->setTaskShader(tsShader);
#endif
    m_visibilityPassHW->setMeshShader(msShader);
    m_visibilityPassHW->setPixelShader(fsShader);
    m_visibilityPassHW->setNumBindlessDescriptorSets(3);
    m_visibilityPassHW->setPushConstSize(u32Size);

    RhiRenderTargetsFormat rtFmt;
    rtFmt.m_colorFormats = {RhiImageFormat::RHI_FORMAT_R32_UINT};
    rtFmt.m_depthFormat = RhiImageFormat::RHI_FORMAT_D32_SFLOAT;
    m_visibilityPassHW->setRenderTargetFormat(rtFmt);

    m_depthOnlyVisibilityPassHW = rhi->createGraphicsPass();
#if !SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL
    m_depthOnlyVisibilityPassHW->setTaskShader(tsShader);
#endif
    m_depthOnlyVisibilityPassHW->setMeshShader(msShaderDepth);
    m_depthOnlyVisibilityPassHW->setNumBindlessDescriptorSets(3);
    m_depthOnlyVisibilityPassHW->setPushConstSize(u32Size);

    RhiRenderTargetsFormat depthOnlyRtFmt;
    depthOnlyRtFmt.m_colorFormats = {};
    depthOnlyRtFmt.m_depthFormat = RhiImageFormat::RHI_FORMAT_D32_SFLOAT;
    m_depthOnlyVisibilityPassHW->setRenderTargetFormat(depthOnlyRtFmt);
  }

#if SYARO_ENABLE_SW_RASTERIZER
  // Software Rasterize
  if (true) {
    auto csShader = createShaderFromFile("Syaro.SoftRasterize.comp.glsl", "main", RhiShaderStage::Compute);
    m_visibilityPassSW = rhi->createComputePass();
    m_visibilityPassSW->setComputeShader(csShader);
    m_visibilityPassSW->setNumBindlessDescriptorSets(3);
    m_visibilityPassSW->setPushConstSize(u32Size * 8);
  }

  // Combine software and hardware rasterize results
  if (true) {
    auto csShader = createShaderFromFile("Syaro.CombineVisBuffer.comp.glsl", "main", RhiShaderStage::Compute);
    m_visibilityCombinePass = rhi->createComputePass();
    m_visibilityCombinePass->setComputeShader(csShader);
    m_visibilityCombinePass->setNumBindlessDescriptorSets(0);
    m_visibilityCombinePass->setPushConstSize(u32Size * 9);
  }
#endif
}

IFRIT_APIDECL void SyaroRenderer::setupInstanceCullingPass() {
  auto rhi = m_app->getRhiLayer();
  m_instanceCullingPass = RenderingUtil::createComputePass(rhi, "Syaro/Syaro.InstanceCulling.comp.glsl", 4, 2);
}

IFRIT_APIDECL void SyaroRenderer::setupPersistentCullingPass() {
  auto rhi = m_app->getRhiLayer();
  m_persistentCullingPass = RenderingUtil::createComputePass(rhi, "Syaro/Syaro.PersistentCulling.comp.glsl", 5, 3);

  m_indirectDrawBuffer = rhi->createBufferDevice("Syaro_IndirectDraw", u32Size * 1, kbBufUsage_Indirect);
  m_indirectDrawBufferId = rhi->registerStorageBuffer(m_indirectDrawBuffer.get());
  m_persistCullDesc = rhi->createBindlessDescriptorRef();
  m_persistCullDesc->addStorageBuffer(m_indirectDrawBuffer.get(), 0);
}

IFRIT_APIDECL void SyaroRenderer::setupSinglePassHiZPass() {
  m_singlePassHiZProc = std::make_shared<SinglePassHiZPass>(m_app);
}
IFRIT_APIDECL void SyaroRenderer::setupEmitDepthTargetsPass() {
  auto rhi = m_app->getRhiLayer();
  m_emitDepthTargetsPass = RenderingUtil::createComputePass(rhi, "Syaro/Syaro.EmitDepthTarget.comp.glsl", 4, 2);
}

IFRIT_APIDECL void SyaroRenderer::setupMaterialClassifyPass() {
  auto rhi = m_app->getRhiLayer();
  m_matclassCountPass = RenderingUtil::createComputePass(rhi, "Syaro/Syaro.ClassifyMaterial.Count.comp.glsl", 1, 3);
  m_matclassReservePass = RenderingUtil::createComputePass(rhi, "Syaro/Syaro.ClassifyMaterial.Reserve.comp.glsl", 1, 3);
  m_matclassScatterPass = RenderingUtil::createComputePass(rhi, "Syaro/Syaro.ClassifyMaterial.Scatter.comp.glsl", 1, 3);
}

IFRIT_APIDECL void SyaroRenderer::materialClassifyBufferSetup(PerFrameData &perframeData,
                                                              RenderTargets *renderTargets) {
  auto numMaterials = perframeData.m_enabledEffects.size();
  auto rhi = m_app->getRhiLayer();

  u32 actualRtWidth = 0, actualRtHeight = 0;
  getSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

  auto width = actualRtWidth;
  auto height = actualRtHeight;
  auto totalSize = width * height;
  bool needRecreate = false;
  bool needRecreateMat = false;
  bool needRecreatePixel = false;
  if (perframeData.m_matClassSupportedNumMaterials < numMaterials || perframeData.m_matClassCountBuffer == nullptr) {
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
    perframeData.m_matClassSupportedNumMaterials = Ifrit::Common::Utility::size_cast<u32>(numMaterials);
    auto createSize = cMatClassCounterBufferSizeBase +
                      cMatClassCounterBufferSizeMult * Ifrit::Common::Utility::size_cast<u32>(numMaterials);

    perframeData.m_matClassCountBuffer =
        rhi->createBufferDevice("Syaro_MatClassCount", createSize, kbBufUsage_SSBO_CopyDest);
    perframeData.m_matClassIndirectDispatchBuffer =
        rhi->createBufferDevice("Syaro_MatClassCountIndirectDisp", u32Size * 4 * numMaterials, kbBufUsage_Indirect);
  }
  if (needRecreatePixel) {
    perframeData.m_matClassSupportedNumPixels = totalSize;
    perframeData.m_matClassFinalBuffer =
        rhi->createBufferDevice("Syaro_MatClassFinal", u32Size * totalSize, kbBufUsage_SSBO);
    perframeData.m_matClassPixelOffsetBuffer =
        rhi->createBufferDevice("Syaro_MatClassPixelOffset", u32Size * totalSize, kbBufUsage_SSBO_CopyDest);
  }

  if (needRecreate) {
    perframeData.m_matClassDesc = rhi->createBindlessDescriptorRef();
    perframeData.m_matClassDesc->addUAVImage(perframeData.m_velocityMaterial.get(), {0, 0, 1, 1}, 0);
    perframeData.m_matClassDesc->addStorageBuffer(perframeData.m_matClassCountBuffer.get(), 1);
    perframeData.m_matClassDesc->addStorageBuffer(perframeData.m_matClassFinalBuffer.get(), 2);
    perframeData.m_matClassDesc->addStorageBuffer(perframeData.m_matClassPixelOffsetBuffer.get(), 3);
    perframeData.m_matClassDesc->addStorageBuffer(perframeData.m_matClassIndirectDispatchBuffer.get(), 4);

    perframeData.m_matClassBarrier.clear();
    RhiResourceBarrier barrierCountBuffer;
    barrierCountBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierCountBuffer.m_uav.m_buffer = perframeData.m_matClassCountBuffer.get();
    barrierCountBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierFinalBuffer;
    barrierFinalBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierFinalBuffer.m_uav.m_buffer = perframeData.m_matClassFinalBuffer.get();
    barrierFinalBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierPixelOffsetBuffer;
    barrierPixelOffsetBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierPixelOffsetBuffer.m_uav.m_buffer = perframeData.m_matClassPixelOffsetBuffer.get();
    barrierPixelOffsetBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierIndirectDispatchBuffer;
    barrierIndirectDispatchBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierIndirectDispatchBuffer.m_uav.m_buffer = perframeData.m_matClassIndirectDispatchBuffer.get();
    barrierIndirectDispatchBuffer.m_uav.m_type = RhiResourceType::Buffer;

    perframeData.m_matClassBarrier.push_back(barrierCountBuffer);
    perframeData.m_matClassBarrier.push_back(barrierFinalBuffer);
    perframeData.m_matClassBarrier.push_back(barrierPixelOffsetBuffer);
    perframeData.m_matClassBarrier.push_back(barrierIndirectDispatchBuffer);
  }
}

IFRIT_APIDECL void SyaroRenderer::depthTargetsSetup(PerFrameData &perframeData, RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();

  u32 actualRtWidth = 0, actualRtHeight = 0;
  getSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

  if (perframeData.m_velocityMaterial != nullptr)
    return;
  perframeData.m_velocityMaterial =
      rhi->createTexture2D("Syaro_VelocityMat", actualRtWidth, actualRtHeight, kbImFmt_RGBA32F, kbImUsage_UAV_SRV);
  perframeData.m_motionVector =
      rhi->createTexture2D("Syaro_Motion", actualRtWidth, actualRtHeight, kbImFmt_RG32F, kbImUsage_UAV_SRV_CopyDest);

  perframeData.m_velocityMaterialDesc = rhi->createBindlessDescriptorRef();
  perframeData.m_velocityMaterialDesc->addUAVImage(perframeData.m_velocityMaterial.get(), {0, 0, 1, 1}, 0);

  auto &primaryView = getPrimaryView(perframeData);
  perframeData.m_velocityMaterialDesc->addCombinedImageSampler(primaryView.m_visibilityBuffer_Combined.get(),
                                                               m_immRes.m_linearSampler.get(), 1);
  perframeData.m_velocityMaterialDesc->addUAVImage(perframeData.m_motionVector.get(), {0, 0, 1, 1}, 2);

  // For gbuffer, depth is required to reconstruct position
  perframeData.m_gbufferDepthDesc = rhi->createBindlessDescriptorRef();
  perframeData.m_gbufferDepthDesc->addCombinedImageSampler(perframeData.m_velocityMaterial.get(),
                                                           m_immRes.m_linearSampler.get(), 0);
}

IFRIT_APIDECL void SyaroRenderer::recreateInstanceCullingBuffers(PerFrameData &perframe, u32 newMaxInstances) {
  for (u32 i = 0; i < perframe.m_views.size(); i++) {
    auto &view = perframe.m_views[i];
    if (view.m_maxSupportedInstances == 0 || view.m_maxSupportedInstances < newMaxInstances) {
      auto rhi = m_app->getRhiLayer();
      view.m_maxSupportedInstances = newMaxInstances;
      view.m_instCullDiscardObj =
          rhi->createBufferDevice("Syaro_InstCullDiscard", u32Size * newMaxInstances, kbBufUsage_SSBO);
      view.m_instCullPassedObj =
          rhi->createBufferDevice("Syaro_InstCullPassed", u32Size * newMaxInstances, kbBufUsage_SSBO_CopyDest);
      view.m_persistCullIndirectDispatch =
          rhi->createBufferDevice("Syaro_InstCullDispatch", u32Size * 12, kbBufUsage_Indirect);

      view.m_instCullDesc = rhi->createBindlessDescriptorRef();
      view.m_instCullDesc->addStorageBuffer(view.m_instCullDiscardObj.get(), 0);
      view.m_instCullDesc->addStorageBuffer(view.m_instCullPassedObj.get(), 1);
      view.m_instCullDesc->addStorageBuffer(view.m_persistCullIndirectDispatch.get(), 2);

      // create barriers
      view.m_persistCullBarrier.clear();
      view.m_persistCullBarrier = registerUAVBarriers(
          {view.m_instCullDiscardObj.get(), view.m_instCullPassedObj.get(), view.m_persistCullIndirectDispatch.get()},
          {});

      view.m_visibilityBarrier.clear();
      view.m_visibilityBarrier =
          registerUAVBarriers({view.m_allFilteredMeshletsHW.get(), view.m_allFilteredMeshletsAllCount.get(),
                               view.m_allFilteredMeshletsSW.get()},
                              {});
    }
  }
}

IFRIT_APIDECL void SyaroRenderer::renderEmitDepthTargets(PerFrameData &perframeData,
                                                         SyaroRenderer::RenderTargets *renderTargets,
                                                         const GPUCmdBuffer *cmd) {
  auto rhi = m_app->getRhiLayer();
  auto &primaryView = getPrimaryView(perframeData);
  m_emitDepthTargetsPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    auto primaryView = getPrimaryView(perframeData);
#if !SYARO_ENABLE_SW_RASTERIZER
    // This barrier is intentionally for layout transition. Sync barrier is
    // issued via globalMemoryBarrier
    ctx->m_cmd->imageBarrier(primaryView.m_visibilityDepth_Combined.get(), RhiResourceState::DepthStencilRenderTarget,
                             RhiResourceState::Common, {0, 0, 1, 1});
#endif
    runImageBarrier(ctx->m_cmd, perframeData.m_velocityMaterial.get(),

                    RhiResourceState::UnorderedAccess, {0, 0, 1, 1});
    runImageBarrier(ctx->m_cmd, perframeData.m_motionVector.get(), RhiResourceState::Common, {0, 0, 1, 1});
    ctx->m_cmd->clearUAVImageFloat(perframeData.m_motionVector.get(), {0, 0, 1, 1}, {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->attachBindlessReferenceCompute(m_emitDepthTargetsPass, 1, primaryView.m_viewBindlessRef);
    ctx->m_cmd->attachBindlessReferenceCompute(m_emitDepthTargetsPass, 2,
                                               perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
    ctx->m_cmd->attachBindlessReferenceCompute(m_emitDepthTargetsPass, 3, primaryView.m_allFilteredMeshletsDesc);
    ctx->m_cmd->attachBindlessReferenceCompute(m_emitDepthTargetsPass, 4, perframeData.m_velocityMaterialDesc);
    u32 pcData[2] = {primaryView.m_renderWidth, primaryView.m_renderHeight};
    ctx->m_cmd->setPushConst(m_emitDepthTargetsPass, 0, u32Size * 2, &pcData[0]);
    u32 wgX = (pcData[0] + cEmitDepthGroupSizeX - 1) / cEmitDepthGroupSizeX;
    u32 wgY = (pcData[1] + cEmitDepthGroupSizeY - 1) / cEmitDepthGroupSizeY;
    ctx->m_cmd->dispatch(wgX, wgY, 1);

    runImageBarrier(ctx->m_cmd, perframeData.m_velocityMaterial.get(),

                    RhiResourceState::UnorderedAccess, {0, 0, 1, 1});

#if !SYARO_ENABLE_SW_RASTERIZER
    runImageBarrier(ctx->m_cmd, primaryView.m_visibilityDepth_Combined.get(), RhiResourceState::DepthStencilRT,
                    {0, 0, 1, 1});
#endif
  });

  cmd->beginScope("Syaro: Emit Depth Targets");
  m_emitDepthTargetsPass->run(cmd, 0);
  cmd->endScope();
}
IFRIT_APIDECL void SyaroRenderer::renderTwoPassOcclCulling(CullingPass cullPass, PerFrameData &perframeData,
                                                           RenderTargets *renderTargets, const GPUCmdBuffer *cmd,
                                                           PerFrameData::ViewType filteredViewType, u32 idx) {
  auto rhi = m_app->getRhiLayer();
  int pcData[2] = {0, 1};

  std::unique_ptr<SyaroRenderer::GPUCommandSubmission> lastTask = nullptr;
  u32 k = idx;
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
  int pcDataInstCull[4] = {0, Ifrit::Common::Utility::size_cast<int>(numObjs), 1,
                           Ifrit::Common::Utility::size_cast<int>(numObjs)};
  m_instanceCullingPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    if (cullPass == CullingPass::First) {
      ctx->m_cmd->bufferClear(perView.m_persistCullIndirectDispatch.get(), 0);
    }
    runUAVBufferBarrier(ctx->m_cmd, perView.m_persistCullIndirectDispatch.get());
    ctx->m_cmd->attachBindlessReferenceCompute(m_instanceCullingPass, 1, perView.m_viewBindlessRef);
    ctx->m_cmd->attachBindlessReferenceCompute(m_instanceCullingPass, 2,
                                               perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
    ctx->m_cmd->attachBindlessReferenceCompute(m_instanceCullingPass, 3, perView.m_instCullDesc);
    ctx->m_cmd->attachBindlessReferenceCompute(m_instanceCullingPass, 4, perView.m_spHiZData.m_hizDesc);

    if (cullPass == CullingPass::First) {
      ctx->m_cmd->setPushConst(m_instanceCullingPass, 0, u32Size * 2, &pcDataInstCull[0]);
      auto tgx = divRoundUp(size_cast<u32>(numObjs), SyaroConfig::cInstanceCullingThreadGroupSizeX);
      ctx->m_cmd->dispatch(tgx, 1, 1);
    } else if (cullPass == CullingPass::Second) {
      ctx->m_cmd->setPushConst(m_instanceCullingPass, 0, u32Size * 2, &pcDataInstCull[2]);
      ctx->m_cmd->dispatchIndirect(perView.m_persistCullIndirectDispatch.get(), 3 * u32Size);
    }
  });

  m_persistentCullingPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    struct PersistCullPushConst {
      u32 passNo;
      u32 swOffset;
      u32 rejectSwRaster;
    } pcPersistCull;

    ctx->m_cmd->resourceBarrier(perView.m_persistCullBarrier);
    if (cullPass == CullingPass::First) {
      ctx->m_cmd->bufferClear(perView.m_allFilteredMeshletsAllCount.get(), 0);
      ctx->m_cmd->bufferClear(m_indirectDrawBuffer.get(), 0);
    }
    runUAVBufferBarrier(ctx->m_cmd, perView.m_allFilteredMeshletsAllCount.get());
    runUAVBufferBarrier(ctx->m_cmd, m_indirectDrawBuffer.get());
    // bind view buffer
    ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 1, perView.m_viewBindlessRef);
    ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 2,
                                               perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
    ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 3, m_persistCullDesc);
    ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 4, perView.m_allFilteredMeshletsDesc);
    ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 5, perView.m_instCullDesc);
    if (perView.m_viewType == PerFrameData::ViewType::Primary) {
      pcPersistCull.rejectSwRaster = 0;
    } else {
      pcPersistCull.rejectSwRaster = 1;
    }
    if (cullPass == CullingPass::First) {
      pcPersistCull.passNo = 0;
      pcPersistCull.swOffset = perView.m_allFilteredMeshlets_SWOffset;

      ctx->m_cmd->setPushConst(m_persistentCullingPass, 0, sizeof(PersistCullPushConst), &pcPersistCull);
      ctx->m_cmd->dispatchIndirect(perView.m_persistCullIndirectDispatch.get(), 0);
    } else if (cullPass == CullingPass::Second) {
      pcPersistCull.passNo = 1;
      pcPersistCull.swOffset = perView.m_allFilteredMeshlets_SWOffset;

      ctx->m_cmd->setPushConst(m_persistentCullingPass, 0, sizeof(PersistCullPushConst), &pcPersistCull);
      ctx->m_cmd->dispatchIndirect(perView.m_persistCullIndirectDispatch.get(), 6 * u32Size);
    }
  });

  auto &visPassHW =
      (filteredViewType == PerFrameData::ViewType::Primary) ? m_visibilityPassHW : m_depthOnlyVisibilityPassHW;
  visPassHW->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    // bind view buffer
    ctx->m_cmd->attachBindlessReferenceGraphics(visPassHW, 1, perView.m_viewBindlessRef);
    ctx->m_cmd->attachBindlessReferenceGraphics(visPassHW, 2, perframeData.m_allInstanceData.m_batchedObjBufRef);
    ctx->m_cmd->setCullMode(RhiCullMode::Back);
    ctx->m_cmd->attachBindlessReferenceGraphics(visPassHW, 3, perView.m_allFilteredMeshletsDesc);
    if (cullPass == CullingPass::First) {
      ctx->m_cmd->setPushConst(visPassHW, 0, u32Size, &pcData[0]);
      ctx->m_cmd->drawMeshTasksIndirect(perView.m_allFilteredMeshletsAllCount.get(), u32Size * 3, 1, 0);
    } else {
      ctx->m_cmd->setPushConst(visPassHW, 0, u32Size, &pcData[1]);
      ctx->m_cmd->drawMeshTasksIndirect(perView.m_allFilteredMeshletsAllCount.get(), u32Size * 0, 1, 0);
    }
  });

  auto &visPassSW = m_visibilityPassSW;

  visPassSW->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    runUAVBufferBarrier(cmd, perView.m_allFilteredMeshletsAllCount.get());

    // if this is the first pass, we need to clear the visibility buffer
    // seems cleaing visibility buffer is not necessary. Just clear the depth
    if (cullPass == CullingPass::First) {
      ctx->m_cmd->bufferClear(perView.m_visPassDepth_SW.get(),
                              0xffffffff); // clear to max_uint
      ctx->m_cmd->bufferClear(perView.m_visPassDepthCASLock_SW.get(), 0);
    }
    runUAVBufferBarrier(ctx->m_cmd, perView.m_visPassDepth_SW.get());
    runUAVBufferBarrier(ctx->m_cmd, perView.m_visPassDepthCASLock_SW.get());
    // bind view buffer
    ctx->m_cmd->attachBindlessReferenceCompute(visPassSW, 1, perView.m_viewBindlessRef);
    ctx->m_cmd->attachBindlessReferenceCompute(visPassSW, 2, perframeData.m_allInstanceData.m_batchedObjBufRef);
    ctx->m_cmd->attachBindlessReferenceCompute(visPassSW, 3, perView.m_allFilteredMeshletsDesc);

    struct SWPushConst {
      u32 passNo;
      u32 depthBufferId;
      u32 visBufferId;
      u32 rtHeight;
      u32 rtWidth;
      u32 swOffset;
      u32 casBufferId;
    } pcsw;
    pcsw.passNo = cullPass == CullingPass::First ? 0 : 1;
    pcsw.depthBufferId = perView.m_visDepthId_SW->getActiveId();
    pcsw.visBufferId = perView.m_visBufferIdUAV_SW->getActiveId();
    pcsw.rtHeight = perView.m_renderHeight;
    pcsw.rtWidth = perView.m_renderWidth;
    pcsw.swOffset = perView.m_allFilteredMeshlets_SWOffset;
    pcsw.casBufferId = perView.m_visDepthCASLockId_SW->getActiveId();
    ctx->m_cmd->setPushConst(visPassSW, 0, sizeof(SWPushConst), &pcsw);
    if (cullPass == CullingPass::First) {
      ctx->m_cmd->dispatchIndirect(perView.m_allFilteredMeshletsAllCount.get(), u32Size * 13);
    } else {
      ctx->m_cmd->dispatchIndirect(perView.m_allFilteredMeshletsAllCount.get(), u32Size * 10);
    }
  });

  auto &combinePass = m_visibilityCombinePass;

  combinePass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    if (cullPass == CullingPass::First) {
      runImageBarrier(ctx->m_cmd, perView.m_visibilityBuffer_Combined.get(), RhiResourceState::UnorderedAccess,
                      {0, 0, 1, 1});
      runImageBarrier(ctx->m_cmd, perView.m_visibilityDepth_Combined.get(), RhiResourceState::UnorderedAccess,
                      {0, 0, 1, 1});
    }
    struct CombinePassPushConst {
      u32 hwVisUAVId;
      u32 hwDepthSRVId;
      u32 swVisUAVId;
      u32 swDepthUAVId;
      u32 rtWidth;
      u32 rtHeight;
      u32 outVisUAVId;
      u32 outDepthUAVId;
      u32 outMode;
    } pcCombine;
    pcCombine.rtWidth = perView.m_renderWidth;
    pcCombine.rtHeight = perView.m_renderHeight;
    pcCombine.outVisUAVId = perView.m_visibilityBufferIdUAV_Combined->getActiveId();
    pcCombine.outDepthUAVId = perView.m_visibilityDepthIdUAV_Combined->getActiveId();
    // Testing, not specifying sw ids
    pcCombine.hwVisUAVId = perView.m_visBufferIdUAV_HW->getActiveId();
    pcCombine.hwDepthSRVId = perView.m_visDepthId_HW->getActiveId();
    pcCombine.swVisUAVId = perView.m_visBufferIdUAV_SW->getActiveId();
    pcCombine.swDepthUAVId = perView.m_visDepthId_SW->getActiveId();
    pcCombine.outMode = m_config->m_visualizationType == RendererVisualizationType::SwHwMaps;

    constexpr auto wgSizeX = SyaroConfig::cCombineVisBufferThreadGroupSizeX;
    constexpr auto wgSizeY = SyaroConfig::cCombineVisBufferThreadGroupSizeY;

    auto tgX = Math::ConstFunc::divRoundUp(pcCombine.rtWidth, wgSizeX);
    auto tgY = Math::ConstFunc::divRoundUp(pcCombine.rtHeight, wgSizeY);
    cmd->setPushConst(combinePass, 0, sizeof(pcCombine), &pcCombine);
    ctx->m_cmd->dispatch(tgX, tgY, 1);
  });

  if (cullPass == CullingPass::First) {
    cmd->beginScope("Syaro: Cull Rasterize I");
  } else {
    cmd->beginScope("Syaro: Cull Rasterize II");
  }
  cmd->beginScope("Syaro: Instance Culling Pass");
  m_instanceCullingPass->run(cmd, 0);
  cmd->endScope();
  cmd->globalMemoryBarrier();
  cmd->beginScope("Syaro: Persistent Culling Pass");
  m_persistentCullingPass->run(cmd, 0);
  cmd->endScope();
  cmd->globalMemoryBarrier();

  // SW Rasterize, TODO:Run in parallel with HW Rasterize. Not specify barrier
  // here
  if (perView.m_viewType == PerFrameData::ViewType::Primary) {
    cmd->beginScope("Syaro: SW Rasterize");
    visPassSW->run(cmd, 0);
    cmd->endScope();
  }
  cmd->globalMemoryBarrier();
  // HW Rasterize
  cmd->beginScope("Syaro: HW Rasterize");
  if (cullPass == CullingPass::First) {
    runUAVBufferBarrier(cmd, perView.m_allFilteredMeshletsAllCount.get());
    visPassHW->run(cmd, perView.m_visRTs_HW.get(), 0);
  } else {
    runUAVBufferBarrier(cmd, perView.m_allFilteredMeshletsAllCount.get());
    visPassHW->run(cmd, perView.m_visRTs2_HW.get(), 0);
  }
  cmd->endScope();

  // Combine HW and SW results,
  if (perView.m_viewType == PerFrameData::ViewType::Primary) {
    cmd->globalMemoryBarrier();
    cmd->beginScope("Syaro: SW Rasterize Merge");
    combinePass->run(cmd, 0);
    cmd->endScope();
  }

  // Run hi-z pass

  cmd->globalMemoryBarrier();
  cmd->beginScope("Syaro: HiZ Pass");
  m_singlePassHiZProc->runHiZPass(perView.m_spHiZData, cmd, perView.m_renderWidth, perView.m_renderHeight, false);
  cmd->endScope();
  cmd->endScope();
}

IFRIT_APIDECL void SyaroRenderer::renderTriangleView(PerFrameData &perframeData, RenderTargets *renderTargets,
                                                     const GPUCmdBuffer *cmd) {
  auto rhi = m_app->getRhiLayer();
  auto &primaryView = getPrimaryView(perframeData);
  setupDebugPasses(perframeData, renderTargets);

  auto rtCfg = renderTargets->getFormat();
  PipelineAttachmentConfigs cfg;
  cfg.m_colorFormats = rtCfg.m_colorFormats;
  cfg.m_depthFormat = rtCfg.m_depthFormat;

  auto triangleView = m_triangleViewPass[cfg];
  struct PushConst {
    u32 visBufferSRV;
  } pc;
  pc.visBufferSRV = primaryView.m_visibilityBufferIdSRV_Combined->getActiveId();
  RenderingUtil::enqueueFullScreenPass(cmd, rhi, triangleView, renderTargets, {}, &pc, 1);
}

IFRIT_APIDECL void SyaroRenderer::renderMaterialClassify(PerFrameData &perframeData, RenderTargets *renderTargets,
                                                         const GPUCmdBuffer *cmd) {
  auto rhi = m_app->getRhiLayer();
  auto totalMaterials = size_cast<u32>(perframeData.m_enabledEffects.size());

  u32 actualRtWidth = 0, actualRtHeight = 0;
  getSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

  auto width = actualRtWidth;
  auto height = actualRtHeight;
  u32 pcData[3] = {width, height, totalMaterials};

  constexpr u32 pTileWidth = cMatClassQuadSize * cMatClassGroupSizeCountScatterX;
  constexpr u32 pTileHeight = cMatClassQuadSize * cMatClassGroupSizeCountScatterY;

  // Counting
  auto wgX = (width + pTileWidth - 1) / pTileWidth;
  auto wgY = (height + pTileHeight - 1) / pTileHeight;
  m_matclassCountPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->bufferClear(perframeData.m_matClassCountBuffer.get(), 0);
    runUAVBufferBarrier(ctx->m_cmd, perframeData.m_matClassCountBuffer.get());

    ctx->m_cmd->attachBindlessReferenceCompute(m_matclassCountPass, 1, perframeData.m_matClassDesc);
    ctx->m_cmd->setPushConst(m_matclassCountPass, 0, u32Size * 3, &pcData[0]);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
  });

  // Reserving
  auto wgX2 = (totalMaterials + cMatClassGroupSizeReserveX - 1) / cMatClassGroupSizeReserveX;
  m_matclassReservePass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->attachBindlessReferenceCompute(m_matclassReservePass, 1, perframeData.m_matClassDesc);
    ctx->m_cmd->setPushConst(m_matclassReservePass, 0, u32Size * 3, &pcData[0]);
    ctx->m_cmd->dispatch(wgX2, 1, 1);
  });

  // Scatter
  m_matclassScatterPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->attachBindlessReferenceCompute(m_matclassScatterPass, 1, perframeData.m_matClassDesc);
    ctx->m_cmd->setPushConst(m_matclassScatterPass, 0, u32Size * 3, &pcData[0]);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
  });

  // Start rendering
  cmd->beginScope("Syaro: Material Classification");
  cmd->globalMemoryBarrier();
  m_matclassCountPass->run(cmd, 0);
  cmd->resourceBarrier(perframeData.m_matClassBarrier);
  cmd->globalMemoryBarrier();
  m_matclassReservePass->run(cmd, 0);
  cmd->resourceBarrier(perframeData.m_matClassBarrier);
  cmd->globalMemoryBarrier();
  m_matclassScatterPass->run(cmd, 0);
  cmd->resourceBarrier(perframeData.m_matClassBarrier);
  cmd->globalMemoryBarrier();
  cmd->endScope();
}

IFRIT_APIDECL void SyaroRenderer::setupDefaultEmitGBufferPass() {
  auto rhi = m_app->getRhiLayer();
  m_defaultEmitGBufferPass = RenderingUtil::createComputePass(rhi, "Syaro/Syaro.EmitGBuffer.Default.comp.glsl", 6, 3);
}

IFRIT_APIDECL void SyaroRenderer::sphizBufferSetup(PerFrameData &perframeData, RenderTargets *renderTargets) {
  for (u32 k = 0; k < perframeData.m_views.size(); k++) {
    auto &perView = perframeData.m_views[k];
    auto width = perView.m_renderWidth;
    auto height = perView.m_renderHeight;
    auto rWidth = 1 << (int)std::ceil(std::log2(width));
    auto rHeight = 1 << (int)std::ceil(std::log2(height));
    // Max-HiZ required by instance cull
    if (m_singlePassHiZProc->checkResourceToRebuild(perView.m_spHiZData, width, height)) {
      m_singlePassHiZProc->prepareHiZResources(perView.m_spHiZData, perView.m_visibilityDepth_Combined.get(),
                                               m_immRes.m_linearSampler.get(), rWidth, rHeight);
    }
    // Min-HiZ required by ssgi
    if (m_singlePassHiZProc->checkResourceToRebuild(perView.m_spHiZDataMin, width, height)) {
      m_singlePassHiZProc->prepareHiZResources(perView.m_spHiZDataMin, perView.m_visibilityDepth_Combined.get(),
                                               m_immRes.m_linearSampler.get(), rWidth, rHeight);
    }
  }
}

IFRIT_APIDECL void SyaroRenderer::visibilityBufferSetup(PerFrameData &perframeData, RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();

  for (u32 k = 0; k < perframeData.m_views.size(); k++) {
    auto &perView = perframeData.m_views[k];
    bool createCond = (perView.m_visibilityBuffer_Combined == nullptr);

    u32 actualRtw = 0, actualRth = 0;
    getSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);

    auto visHeight = perView.m_renderHeight;
    auto visWidth = perView.m_renderWidth;
    if (visHeight == 0 || visWidth == 0) {
      if (perView.m_viewType == PerFrameData::ViewType::Primary) {
        // use render target size
        visHeight = actualRth;
        visWidth = actualRtw;
      } else if (perView.m_viewType == PerFrameData::ViewType::Shadow) {
        Logging::error("Shadow view has no size");
        std::abort();
      }
    }

    if (!createCond) {
      return;
    }
    // It seems nanite's paper uses R32G32 for mesh visibility, but
    // I wonder the depth is implicitly calculated from the depth buffer
    // so here I use R32 for visibility buffer
    // Update: Now it's required for SW rasterizer. But I use a separate
    // buffer

    // For HW rasterizer
    if constexpr (true) {
      auto visBufferHW = rhi->createTexture2D("Syaro_VisBufferHW", visWidth, visHeight,
                                              PerFrameData::c_visibilityFormat, kbImUsage_UAV_SRV_RT);
      auto visDepthHW = rhi->createDepthTexture("Syaro_VisDepthHW", visWidth, visHeight);
      auto visDepthSampler = rhi->createTrivialSampler();
      perView.m_visibilityBuffer_HW = visBufferHW;
      perView.m_visBufferIdUAV_HW = rhi->registerUAVImage(perView.m_visibilityBuffer_HW.get(), {0, 0, 1, 1});

      // first pass rts
      perView.m_visPassDepth_HW = visDepthHW;
      perView.m_visDepthId_HW =
          rhi->registerCombinedImageSampler(perView.m_visPassDepth_HW.get(), m_immRes.m_linearSampler.get());
      perView.m_visDepthRT_HW =
          rhi->createRenderTargetDepthStencil(visDepthHW.get(), {{}, 1.0f}, RhiRenderTargetLoadOp::Clear);

      perView.m_visRTs_HW = rhi->createRenderTargets();
      if (perView.m_viewType == PerFrameData::ViewType::Primary) {
        perView.m_visColorRT_HW =
            rhi->createRenderTarget(visBufferHW.get(), {{0, 0, 0, 0}}, RhiRenderTargetLoadOp::Clear, 0, 0);
        perView.m_visRTs_HW->setColorAttachments({perView.m_visColorRT_HW.get()});
      } else {
        // shadow passes do not need color attachment
        perView.m_visRTs_HW->setColorAttachments({});
      }
      perView.m_visRTs_HW->setDepthStencilAttachment(perView.m_visDepthRT_HW.get());
      perView.m_visRTs_HW->setRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));
      if (perView.m_viewType == PerFrameData::ViewType::Shadow) {
        auto rtHeight = (perView.m_renderHeight);
        auto rtWidth = (perView.m_renderWidth);
        perView.m_visRTs_HW->setRenderArea({0, 0, rtWidth, rtHeight});
      }

      // second pass rts
      perView.m_visDepthRT2_HW =
          rhi->createRenderTargetDepthStencil(visDepthHW.get(), {{}, 1.0f}, RhiRenderTargetLoadOp::Load);
      perView.m_visRTs2_HW = rhi->createRenderTargets();
      if (perView.m_viewType == PerFrameData::ViewType::Primary) {
        perView.m_visColorRT2_HW =
            rhi->createRenderTarget(visBufferHW.get(), {{0, 0, 0, 0}}, RhiRenderTargetLoadOp::Load, 0, 0);
        perView.m_visRTs2_HW->setColorAttachments({perView.m_visColorRT2_HW.get()});
      } else {
        // shadow passes do not need color attachment
        perView.m_visRTs2_HW->setColorAttachments({});
      }
      perView.m_visRTs2_HW->setDepthStencilAttachment(perView.m_visDepthRT2_HW.get());
      perView.m_visRTs2_HW->setRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));
      if (perView.m_viewType == PerFrameData::ViewType::Shadow) {
        auto rtHeight = (perView.m_renderHeight);
        auto rtWidth = (perView.m_renderWidth);
        perView.m_visRTs2_HW->setRenderArea({0, 0, rtWidth, rtHeight});
      }
    }

    // For SW rasterizer
    if (SYARO_ENABLE_SW_RASTERIZER && perView.m_viewType == PerFrameData::ViewType::Primary) {
      perView.m_visibilityBuffer_SW = rhi->createTexture2D("Syaro_VisBufferSW", visWidth, visHeight,
                                                           PerFrameData::c_visibilityFormat, kbImUsage_UAV_SRV);
      perView.m_visPassDepth_SW =
          rhi->createBufferDevice("Syaro_VisDepthSW", u64Size * visWidth * visHeight, kbBufUsage_SSBO_CopyDest);

      perView.m_visPassDepthCASLock_SW =
          rhi->createBufferDevice("Syaro_VisCasSW", f32Size * visWidth * visHeight, kbBufUsage_SSBO_CopyDest);

      perView.m_visDepthId_SW = rhi->registerStorageBuffer(perView.m_visPassDepth_SW.get());
      perView.m_visDepthCASLockId_SW = rhi->registerStorageBuffer(perView.m_visPassDepthCASLock_SW.get());
      perView.m_visBufferIdUAV_SW = rhi->registerUAVImage(perView.m_visibilityBuffer_SW.get(), {0, 0, 1, 1});
    }

    // For combined buffer
    if (SYARO_ENABLE_SW_RASTERIZER && perView.m_viewType == PerFrameData::ViewType::Primary) {
      perView.m_visibilityBuffer_Combined = rhi->createTexture2D("Syaro_VisBufferComb", visWidth, visHeight,
                                                                 PerFrameData::c_visibilityFormat, kbImUsage_UAV_SRV);
      perView.m_visibilityDepth_Combined =
          rhi->createTexture2D("Syaro_VisDepthComb", visWidth, visHeight, kbImFmt_R32F, kbImUsage_UAV_SRV);
      perView.m_visibilityBufferIdUAV_Combined =
          rhi->registerUAVImage(perView.m_visibilityBuffer_Combined.get(), {0, 0, 1, 1});
      perView.m_visibilityDepthIdUAV_Combined =
          rhi->registerUAVImage(perView.m_visibilityDepth_Combined.get(), {0, 0, 1, 1});
      perView.m_visibilityBufferIdSRV_Combined =
          rhi->registerCombinedImageSampler(perView.m_visibilityBuffer_Combined.get(), m_immRes.m_nearestSampler.get());
      perView.m_visibilityDepthIdSRV_Combined =
          rhi->registerCombinedImageSampler(perView.m_visibilityDepth_Combined.get(), m_immRes.m_linearSampler.get());
    } else {
      perView.m_visibilityDepth_Combined = perView.m_visPassDepth_HW;
      perView.m_visibilityBuffer_Combined = perView.m_visibilityBuffer_HW;
      perView.m_visibilityDepthIdSRV_Combined = perView.m_visDepthId_HW;
      perView.m_visibilityBufferIdSRV_Combined =
          rhi->registerCombinedImageSampler(perView.m_visibilityBuffer_Combined.get(), m_immRes.m_linearSampler.get());

      // UAV is not needed for HW rasterizer
      perView.m_visibilityBufferIdUAV_Combined = nullptr;
      perView.m_visibilityDepthIdUAV_Combined = nullptr;
    }
  }
}

IFRIT_APIDECL void SyaroRenderer::renderDefaultEmitGBuffer(PerFrameData &perframeData, RenderTargets *renderTargets,
                                                           const GPUCmdBuffer *cmd) {
  auto numMaterials = perframeData.m_enabledEffects.size();
  m_defaultEmitGBufferPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    // first transition all gbuffer textures to UAV/Common
    runImageBarrier(ctx->m_cmd, perframeData.m_gbuffer.m_albedo_materialFlags.get(), RhiResourceState::Common,
                    {0, 0, 1, 1});
    runImageBarrier(ctx->m_cmd, perframeData.m_gbuffer.m_normal_smoothness.get(), RhiResourceState::Common,
                    {0, 0, 1, 1});
    runImageBarrier(ctx->m_cmd, perframeData.m_gbuffer.m_emissive.get(), RhiResourceState::Common, {0, 0, 1, 1});
    runImageBarrier(ctx->m_cmd, perframeData.m_gbuffer.m_specular_occlusion.get(), RhiResourceState::Common,
                    {0, 0, 1, 1});
    runImageBarrier(ctx->m_cmd, perframeData.m_gbuffer.m_shadowMask.get(), RhiResourceState::Common, {0, 0, 1, 1});

    // Clear gbuffer textures
    ctx->m_cmd->clearUAVImageFloat(perframeData.m_gbuffer.m_albedo_materialFlags.get(), {0, 0, 1, 1},
                                   {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->clearUAVImageFloat(perframeData.m_gbuffer.m_normal_smoothness.get(), {0, 0, 1, 1},
                                   {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->clearUAVImageFloat(perframeData.m_gbuffer.m_emissive.get(), {0, 0, 1, 1}, {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->clearUAVImageFloat(perframeData.m_gbuffer.m_specular_occlusion.get(), {0, 0, 1, 1},
                                   {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->clearUAVImageFloat(perframeData.m_gbuffer.m_shadowMask.get(), {0, 0, 1, 1}, {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->resourceBarrier(perframeData.m_gbuffer.m_gbufferBarrier);

    // For each material, make
    // one dispatch
    u32 actualRtw = 0, actualRth = 0;
    getSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);
    u32 pcData[3] = {0, actualRtw, actualRth};
    auto &primaryView = getPrimaryView(perframeData);
    for (int i = 0; i < numMaterials; i++) {
      ctx->m_cmd->attachBindlessReferenceCompute(m_defaultEmitGBufferPass, 1, perframeData.m_matClassDesc);
      ctx->m_cmd->attachBindlessReferenceCompute(m_defaultEmitGBufferPass, 2, perframeData.m_gbuffer.m_gbufferDesc);
      ctx->m_cmd->attachBindlessReferenceCompute(m_defaultEmitGBufferPass, 3, primaryView.m_viewBindlessRef);
      ctx->m_cmd->attachBindlessReferenceCompute(m_defaultEmitGBufferPass, 4,
                                                 perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
      ctx->m_cmd->attachBindlessReferenceCompute(m_defaultEmitGBufferPass, 5, primaryView.m_allFilteredMeshletsDesc);
      ctx->m_cmd->attachBindlessReferenceCompute(m_defaultEmitGBufferPass, 6, perframeData.m_velocityMaterialDesc);

      pcData[0] = i;
      ctx->m_cmd->setPushConst(m_defaultEmitGBufferPass, 0, u32Size * 3, &pcData[0]);
      RhiResourceBarrier barrierCountBuffer;
      barrierCountBuffer.m_type = RhiBarrierType::UAVAccess;
      barrierCountBuffer.m_uav.m_buffer = perframeData.m_matClassCountBuffer.get();
      barrierCountBuffer.m_uav.m_type = RhiResourceType::Buffer;

      ctx->m_cmd->resourceBarrier({barrierCountBuffer});
      ctx->m_cmd->dispatchIndirect(perframeData.m_matClassIndirectDispatchBuffer.get(), i * u32Size * 4);
      ctx->m_cmd->resourceBarrier(perframeData.m_gbuffer.m_gbufferBarrier);
    }
  });
  auto rhi = m_app->getRhiLayer();
  cmd->beginScope("Syaro: Emit  GBuffer");
  m_defaultEmitGBufferPass->run(cmd, 0);
  cmd->endScope();
}

IFRIT_APIDECL void SyaroRenderer::renderAmbientOccl(PerFrameData &perframeData, RenderTargets *renderTargets,
                                                    const GPUCmdBuffer *cmd) {
  auto primaryView = getPrimaryView(perframeData);

  auto albedoSamp = perframeData.m_gbuffer.m_albedo_materialFlags_sampId;
  auto normalSamp = perframeData.m_gbuffer.m_normal_smoothness_sampId;
  auto depthSamp = primaryView.m_visibilityDepthIdSRV_Combined;
  auto perframe = primaryView.m_viewBufferId;
  auto ao = perframeData.m_gbuffer.m_specular_occlusionId;
  auto aoIntermediate = perframeData.m_gbuffer.m_specular_occlusion_intermediateId;
  auto aoIntermediateSRV = perframeData.m_gbuffer.m_specular_occlusion_intermediate_sampId;

  auto aoRT = perframeData.m_gbuffer.m_specular_occlusion_RTs;
  auto fsr2samp = perframeData.m_fsr2Data.m_fsr2OutputSRVId;

  u32 width = primaryView.m_renderWidth;
  u32 height = primaryView.m_renderHeight;

  auto aoBlurFunc = [&]() {
    // Blurring AO
    auto rhi = m_app->getRhiLayer();
    cmd->globalMemoryBarrier();
    auto colorRT1 = rhi->createRenderTarget(perframeData.m_gbuffer.m_specular_occlusion_intermediate.get(),
                                            {{0, 0, 0, 0}}, RhiRenderTargetLoadOp::Clear, 0, 0);
    auto colorRT2 = rhi->createRenderTarget(perframeData.m_gbuffer.m_specular_occlusion.get(), {{0, 0, 0, 0}},
                                            RhiRenderTargetLoadOp::Clear, 0, 0);
    auto rt1 = rhi->createRenderTargets();
    rt1->setColorAttachments({colorRT1.get()});
    rt1->setRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));
    auto rt2 = rhi->createRenderTargets();
    rt2->setColorAttachments({colorRT2.get()});
    rt2->setRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));

    m_gaussianHori->renderPostFx(cmd, rt1.get(), perframeData.m_gbuffer.m_specular_occlusion_sampId.get(), 5);
    m_gaussianVert->renderPostFx(cmd, rt2.get(), perframeData.m_gbuffer.m_specular_occlusion_intermediate_sampId.get(),
                                 5);
  };

  cmd->beginScope("Syaro: Ambient Occlusion");
  cmd->resourceBarrier(
      {perframeData.m_gbuffer.m_normal_smoothnessBarrier, perframeData.m_gbuffer.m_specular_occlusionBarrier});
  if (m_config->m_indirectLightingType == IndirectLightingType::HBAO) {
    m_aoPass->renderHBAO(cmd, width, height, depthSamp.get(), normalSamp.get(), ao.get(), perframe.get());
    cmd->resourceBarrier({perframeData.m_gbuffer.m_specular_occlusionBarrier});
    aoBlurFunc();
  } else if (m_config->m_indirectLightingType == IndirectLightingType::SSGI) {
    m_aoPass->renderHBAO(cmd, width, height, depthSamp.get(), normalSamp.get(), ao.get(), perframe.get());
    cmd->resourceBarrier({perframeData.m_gbuffer.m_specular_occlusionBarrier});
    aoBlurFunc();
    cmd->globalMemoryBarrier();
    m_singlePassHiZProc->runHiZPass(primaryView.m_spHiZDataMin, cmd, width, height, true);
    cmd->globalMemoryBarrier();
    m_aoPass->renderSSGI(
        cmd, width, height, primaryView.m_viewBufferId.get(), primaryView.m_spHiZDataMin.m_hizRefBufferId.get(),
        primaryView.m_spHiZData.m_hizRefBufferId.get(), normalSamp.get(), aoIntermediate.get(), albedoSamp.get(),
        primaryView.m_spHiZDataMin.m_hizWidth, primaryView.m_spHiZDataMin.m_hizHeight,
        primaryView.m_spHiZDataMin.m_hizIters, m_immRes.m_blueNoiseSRV.get());

    cmd->globalMemoryBarrier();
    m_jointBilateralFilter->renderPostFx(cmd, aoRT.get(), aoIntermediateSRV.get(), normalSamp.get(), depthSamp.get(),
                                         0);
  }
  cmd->globalMemoryBarrier();
  cmd->endScope();
}

IFRIT_APIDECL void SyaroRenderer::gatherAllInstances(PerFrameData &perframeData) {
  u32 totalInstances = 0;
  u32 totalMeshlets = 0;
  for (auto x : perframeData.m_enabledEffects) {
    auto &effect = perframeData.m_shaderEffectData[x];
    totalInstances += size_cast<u32>(effect.m_objectData.size());
  }
  auto rhi = m_app->getRhiLayer();
  if (perframeData.m_allInstanceData.m_lastObjectCount != totalInstances) {
    perframeData.m_allInstanceData.m_lastObjectCount = totalInstances;
    perframeData.m_allInstanceData.m_batchedObjectData =
        rhi->createBufferCoherent(totalInstances * sizeof(PerObjectData), kbBufUsage_SSBO);
    perframeData.m_allInstanceData.m_batchedObjBufRef = rhi->createBindlessDescriptorRef();
    auto buf = perframeData.m_allInstanceData.m_batchedObjectData;
    perframeData.m_allInstanceData.m_batchedObjBufRef->addStorageBuffer(buf.get(), 0);
  }
  perframeData.m_allInstanceData.m_objectData.resize(totalInstances);

  for (auto i = 0; auto &x : perframeData.m_enabledEffects) {
    auto &effect = perframeData.m_shaderEffectData[x];
    for (auto &obj : effect.m_objectData) {
      perframeData.m_allInstanceData.m_objectData[i] = obj;
      i++;
    }
    u32 objDataSize = static_cast<u32>(effect.m_objectData.size());
    u32 matSize = static_cast<u32>(perframeData.m_shaderEffectData[x].m_materials.size());
    for (u32 k = 0; k < matSize; k++) {
      auto mesh = perframeData.m_shaderEffectData[x].m_meshes[k]->loadMeshUnsafe();
      auto lv0Meshlets = mesh->m_numMeshletsEachLod[0];
      auto lv1Meshlets = 0;
      if (mesh->m_numMeshletsEachLod.size() > 1) {
        lv1Meshlets = mesh->m_numMeshletsEachLod[1];
      }
      totalMeshlets += (lv0Meshlets + lv1Meshlets);
    }
  }
  auto activeBuf = perframeData.m_allInstanceData.m_batchedObjectData->getActiveBuffer();
  activeBuf->map();
  activeBuf->writeBuffer(perframeData.m_allInstanceData.m_objectData.data(),
                         size_cast<u32>(perframeData.m_allInstanceData.m_objectData.size() * sizeof(PerObjectData)), 0);
  activeBuf->flush();
  activeBuf->unmap();

  RhiBufferRef allFilteredMeshletsHW = nullptr;
  RhiBufferRef allFilteredMeshletsSW = nullptr;
  for (u32 k = 0; k < perframeData.m_views.size(); k++) {
    auto &perView = perframeData.m_views[k];
    if (perView.m_allFilteredMeshletsAllCount == nullptr) {
      perView.m_allFilteredMeshletsAllCount =
          rhi->createBufferDevice("Syaro_FilteredClusterCnt", u32Size * 20, kbBufUsage_Indirect);
    }
    if (perView.m_allFilteredMeshletsMaxCount < totalMeshlets) {
      perView.m_allFilteredMeshletsMaxCount = totalMeshlets;
      constexpr u32 bufMultiplier = 1 + SYARO_ENABLE_SW_RASTERIZER;
      if (allFilteredMeshletsHW == nullptr) {
        allFilteredMeshletsHW = rhi->createBufferDevice(
            "Syaro_FilteredClusterHW", totalMeshlets * bufMultiplier * sizeof(iint2) + 2, kbBufUsage_SSBO_CopyDest);
      }
      perView.m_allFilteredMeshletsHW = allFilteredMeshletsHW;

#if SYARO_ENABLE_SW_RASTERIZER
      if (allFilteredMeshletsSW == nullptr) {
        allFilteredMeshletsSW = allFilteredMeshletsHW;
      }
      perView.m_allFilteredMeshlets_SWOffset = totalMeshlets + 1;
      perView.m_allFilteredMeshletsSW = allFilteredMeshletsSW;
#endif

      perView.m_allFilteredMeshletsDesc = rhi->createBindlessDescriptorRef();
      perView.m_allFilteredMeshletsDesc->addStorageBuffer(perView.m_allFilteredMeshletsHW.get(), 0);
#if SYARO_ENABLE_SW_RASTERIZER
      perView.m_allFilteredMeshletsDesc->addStorageBuffer(perView.m_allFilteredMeshletsSW.get(), 1);
#else
      perView.m_allFilteredMeshletsDesc->addStorageBuffer(perView.m_allFilteredMeshletsHW, 1);
#endif
      perView.m_allFilteredMeshletsDesc->addStorageBuffer(perView.m_allFilteredMeshletsAllCount.get(), 2);
    }
  }
}

IFRIT_APIDECL void SyaroRenderer::prepareAggregatedShadowData(PerFrameData &perframeData) {
  auto &shadowData = perframeData.m_shadowData2;
  auto rhi = m_app->getRhiLayer();

  if (shadowData.m_allShadowData == nullptr) {
    shadowData.m_allShadowData = rhi->createBufferCoherent(256 * sizeof(decltype(shadowData.m_shadowViews)::value_type),
                                                           kbBufUsage_SSBO_CopyDest);
    shadowData.m_allShadowDataId = rhi->registerStorageBufferShared(shadowData.m_allShadowData.get());
  }
  for (auto d = 0; auto &x : shadowData.m_shadowViews) {
    for (auto i = 0u; i < x.m_csmSplits; i++) {
      auto idx = x.m_viewMapping[i];
      x.m_texRef[i] = perframeData.m_views[idx].m_visibilityDepthIdSRV_Combined->getActiveId();
      x.m_viewRef[i] = perframeData.m_views[idx].m_viewBufferId->getActiveId();
    }
  }
  auto p = shadowData.m_allShadowData->getActiveBuffer();
  p->map();
  p->writeBuffer(
      shadowData.m_shadowViews.data(),
      size_cast<u32>(shadowData.m_shadowViews.size() * sizeof(decltype(shadowData.m_shadowViews)::value_type)), 0);
  p->flush();
  p->unmap();

  auto mainView = getPrimaryView(perframeData);
  auto mainRtWidth = mainView.m_renderWidth;
  auto mainRtHeight = mainView.m_renderHeight;
  if (perframeData.m_deferShadowMask == nullptr) {
    perframeData.m_deferShadowMask =
        rhi->createTexture2D("Syaro_DeferShadow", mainRtWidth, mainRtHeight, kbImFmt_RGBA32F, kbImUsage_UAV_SRV_RT);

    perframeData.m_deferShadowMaskRT = rhi->createRenderTarget(
        perframeData.m_deferShadowMask.get(), {{0.0f, 0.0f, 0.0f, 1.0f}}, RhiRenderTargetLoadOp::Clear, 0, 0);
    perframeData.m_deferShadowMaskRTs = rhi->createRenderTargets();
    perframeData.m_deferShadowMaskRTs->setColorAttachments({perframeData.m_deferShadowMaskRT.get()});
    perframeData.m_deferShadowMaskRTs->setRenderArea({0, 0, u32(mainRtWidth), u32(mainRtHeight)});
    perframeData.m_deferShadowMaskId =
        rhi->registerCombinedImageSampler(perframeData.m_deferShadowMask.get(), m_immRes.m_linearSampler.get());
  }
}

IFRIT_APIDECL void SyaroRenderer::fsr2Setup(PerFrameData &perframeData, RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();
  u32 actualRtw = 0, actualRth = 0;
  u32 outputRtw = 0, outputRth = 0;
  getSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);

  auto outputArea = renderTargets->getRenderArea();
  outputRtw = outputArea.width;
  outputRth = outputArea.height;
  if (perframeData.m_fsr2Data.m_fsr2Output != nullptr)
    return;
  perframeData.m_fsr2Data.m_fsr2Output =
      rhi->createTexture2D("Syaro_FSR2Out", outputRtw, outputRth, kbImFmt_RGBA16F, kbImUsage_UAV_SRV);

  perframeData.m_fsr2Data.m_fsr2OutputSRVId =
      rhi->registerCombinedImageSampler(perframeData.m_fsr2Data.m_fsr2Output.get(), m_immRes.m_linearSampler.get());

  if (m_config->m_antiAliasingType == AntiAliasingType::FSR2) {
    GraphicsBackend::Rhi::FSR2::RhiFSR2InitialzeArgs args;
    args.displayHeight = outputRth;
    args.displayWidth = outputRtw;
    args.maxRenderWidth = actualRtw;
    args.maxRenderHeight = actualRth;
    m_fsr2proc->init(args);
  }
}

IFRIT_APIDECL void SyaroRenderer::taaHistorySetup(PerFrameData &perframeData, RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();

  u32 actualRtw = 0, actualRth = 0;
  getSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);

  auto width = actualRtw;
  auto height = actualRth;

  auto needRecreate = (perframeData.m_taaHistory.size() == 0);
  if (!needRecreate) {
    needRecreate = (perframeData.m_taaHistory[0].m_width != width || perframeData.m_taaHistory[0].m_height != height);
  }
  if (!needRecreate) {
    return;
  }
  perframeData.m_taaHistory.clear();
  perframeData.m_taaHistory.resize(2);
  perframeData.m_taaHistory[0].m_width = width;
  perframeData.m_taaHistory[0].m_height = height;
  perframeData.m_taaHistoryDesc = rhi->createBindlessDescriptorRef();
  auto rtFormat = renderTargets->getFormat();

  perframeData.m_taaUnresolved =
      rhi->createTexture2D("Syaro_TAAUnresolved", width, height, cTAAFormat, kbImUsage_UAV_SRV_RT_CopySrc);

  perframeData.m_taaHistoryDesc->addCombinedImageSampler(perframeData.m_taaUnresolved.get(),
                                                         m_immRes.m_linearSampler.get(), 0);
  for (int i = 0; i < 2; i++) {
    // TODO: choose formats
    perframeData.m_taaHistory[i].m_colorRT =
        rhi->createTexture2D("Syaro_TAAHistory", width, height, cTAAFormat, kbImUsage_UAV_SRV_RT_CopySrc);
    perframeData.m_taaHistory[i].m_colorRTId = rhi->registerUAVImage(perframeData.m_taaUnresolved.get(), {0, 0, 1, 1});
    perframeData.m_taaHistory[i].m_colorRTIdSRV =
        rhi->registerCombinedImageSampler(perframeData.m_taaUnresolved.get(), m_immRes.m_linearSampler.get());

    // TODO: clear values
    perframeData.m_taaHistory[i].m_colorRTRef =
        rhi->createRenderTarget(perframeData.m_taaUnresolved.get(), {{0, 0, 0, 0}}, RhiRenderTargetLoadOp::Clear, 0, 0);

    RhiAttachmentBlendInfo blendInfo;
    blendInfo.m_blendEnable = false;
    blendInfo.m_srcColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
    blendInfo.m_dstColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendInfo.m_colorBlendOp = RhiBlendOp::RHI_BLEND_OP_ADD;
    blendInfo.m_alphaBlendOp = RhiBlendOp::RHI_BLEND_OP_ADD;
    blendInfo.m_srcAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_SRC_ALPHA;
    blendInfo.m_dstAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    perframeData.m_taaHistory[i].m_colorRTRef->setBlendInfo(blendInfo);

    perframeData.m_taaHistory[i].m_rts = rhi->createRenderTargets();
    perframeData.m_taaHistory[i].m_rts->setColorAttachments({perframeData.m_taaHistory[i].m_colorRTRef.get()});
    perframeData.m_taaHistory[i].m_rts->setRenderArea(getSupersampleDownsampledArea(renderTargets, *m_config));

    perframeData.m_taaHistoryDesc->addCombinedImageSampler(perframeData.m_taaHistory[i].m_colorRT.get(),
                                                           m_immRes.m_linearSampler.get(), i + 1);
  }
}

IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission>
SyaroRenderer::render(PerFrameData &perframeData, SyaroRenderer::RenderTargets *renderTargets,
                      const std::vector<SyaroRenderer::GPUCommandSubmission *> &cmdToWait) {

  // According to
  // lunarg(https://www.lunasdk.org/manual/rhi/command_queues_and_command_buffers/)
  // graphics queue can accept dispatch and transfer commands,
  // but compute queue can only accept compute/transfer commands.
  // Following posts suggests to reduce command buffer submission to
  // improve performance
  // https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/
  // https://gpuopen.com/learn/rdna-performance-guide/#command-buffers

  auto start = std::chrono::high_resolution_clock::now();

  visibilityBufferSetup(perframeData, renderTargets);
  auto &primaryView = getPrimaryView(perframeData);
  buildPipelines(perframeData, GraphicsShaderPassType::Opaque, primaryView.m_visRTs_HW.get());
  prepareDeviceResources(perframeData, renderTargets);
  gatherAllInstances(perframeData);

  recreateInstanceCullingBuffers(perframeData, size_cast<u32>(perframeData.m_allInstanceData.m_objectData.size()));
  depthTargetsSetup(perframeData, renderTargets);
  materialClassifyBufferSetup(perframeData, renderTargets);
  recreateGBuffers(perframeData, renderTargets);
  sphizBufferSetup(perframeData, renderTargets);
  taaHistorySetup(perframeData, renderTargets);
  fsr2Setup(perframeData, renderTargets);

  auto start1 = std::chrono::high_resolution_clock::now();
  prepareAggregatedShadowData(perframeData);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

  // Then draw
  auto rhi = m_app->getRhiLayer();
  auto dq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);

  std::vector<RhiTaskSubmission *> cmdToWaitBkp = cmdToWait;
  std::unique_ptr<RhiTaskSubmission> pbrAtmoTask;
  if (perframeData.m_atmosphereData == nullptr) {
    // Need to create an atmosphere output texture
    u32 actualRtw = 0, actualRth = 0;
    getSupersampledRenderArea(renderTargets, &actualRtw, &actualRth);
    perframeData.m_atmoOutput =
        rhi->createTexture2D("Syaro_AtmoOutput", actualRtw, actualRth, kbImFmt_RGBA32F, kbImUsage_UAV_SRV);

    perframeData.m_atmoOutputId = rhi->registerUAVImage(perframeData.m_atmoOutput.get(), {0, 0, 1, 1});

    // Precompute only once
    pbrAtmoTask = this->m_atmosphereRenderer->renderInternal(perframeData, cmdToWait);
    cmdToWaitBkp = {pbrAtmoTask.get()};
  }

  auto end0 = std::chrono::high_resolution_clock::now();
  auto elapsed0 = std::chrono::duration_cast<std::chrono::microseconds>(end0 - start);

  start = std::chrono::high_resolution_clock::now();
  auto mainTask = dq->runAsyncCommand(
      [&](const RhiCommandList *cmd) {
        for (u32 i = 0; i < perframeData.m_views.size(); i++) {
          if (perframeData.m_views[i].m_viewType == PerFrameData::ViewType::Shadow &&
              m_config->m_visualizationType == RendererVisualizationType::Default &&
              m_renderRole == SyaroRenderRole::SYARO_FULL) {
            cmd->beginScope("Syaro: Draw Call, Shadow View");
            cmd->globalMemoryBarrier();
            auto &perView = perframeData.m_views[i];
            renderTwoPassOcclCulling(CullingPass::First, perframeData, renderTargets, cmd,
                                     PerFrameData::ViewType::Shadow, i);
            renderTwoPassOcclCulling(CullingPass::Second, perframeData, renderTargets, cmd,
                                     PerFrameData::ViewType::Shadow, i);
            cmd->endScope();
          } else if (perframeData.m_views[i].m_viewType == PerFrameData::ViewType::Primary) {
            cmd->beginScope("Syaro: Draw Call, Main View");
            renderTwoPassOcclCulling(CullingPass::First, perframeData, renderTargets, cmd,
                                     PerFrameData::ViewType::Primary, ~0u);
            renderTwoPassOcclCulling(CullingPass::Second, perframeData, renderTargets, cmd,
                                     PerFrameData::ViewType::Primary, ~0u);
            cmd->globalMemoryBarrier();
            if (m_config->m_visualizationType != RendererVisualizationType::Default) {
              return;
            }
            renderEmitDepthTargets(perframeData, renderTargets, cmd);
            cmd->globalMemoryBarrier();
            renderMaterialClassify(perframeData, renderTargets, cmd);
            cmd->globalMemoryBarrier();
            renderDefaultEmitGBuffer(perframeData, renderTargets, cmd);
            cmd->globalMemoryBarrier();
            if (m_renderRole == SyaroRenderRole::SYARO_FULL) {
              renderAmbientOccl(perframeData, renderTargets, cmd);
            }
            cmd->endScope();
          }
        }
      },
      cmdToWaitBkp, {});

  if (m_renderRole == SyaroRenderRole::SYARO_FULL) {
    if (m_config->m_visualizationType != RendererVisualizationType::Default) {
      auto deferredTask = dq->runAsyncCommand(
          [&](const RhiCommandList *cmd) {
            if (m_config->m_visualizationType == RendererVisualizationType::Triangle ||
                m_config->m_visualizationType == RendererVisualizationType::SwHwMaps) {
              cmd->globalMemoryBarrier();
              renderTriangleView(perframeData, renderTargets, cmd);
              return;
            }
          },
          {mainTask.get()}, {});
      return deferredTask;
    } else {
      auto deferredTask = dq->runAsyncCommand(
          [&](const RhiCommandList *cmd) { setupAndRunFrameGraph(perframeData, renderTargets, cmd); }, {mainTask.get()},
          {});
      return deferredTask;
    }
  } else {
    return mainTask;
  }
}

IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission>
SyaroRenderer::render(Scene *scene, Camera *camera, RenderTargets *renderTargets, const RendererConfig &config,
                      const std::vector<GPUCommandSubmission *> &cmdToWait) {

  auto start = std::chrono::high_resolution_clock::now();
  prepareImmutableResources();

  if (m_perScenePerframe.count(scene) == 0) {
    m_perScenePerframe[scene] = PerFrameData();
  }
  m_renderConfig = config;
  m_config = &config;
  auto &perframeData = m_perScenePerframe[scene];

  auto frameId = perframeData.m_frameId;
  auto frameTimestampRaw = std::chrono::high_resolution_clock::now();
  auto frameTimestampMicro =
      std::chrono::duration_cast<std::chrono::microseconds>(frameTimestampRaw.time_since_epoch());
  auto frameTimestampMicroCount = frameTimestampMicro.count();
  float frameTimestampMili = frameTimestampMicroCount / 1000.0f;
  perframeData.m_frameTimestamp[frameId % 2] = frameTimestampMili;

  auto haltonX = RendererConsts::cHalton2[frameId % RendererConsts::cHalton2.size()];
  auto haltonY = RendererConsts::cHalton3[frameId % RendererConsts::cHalton3.size()];

  u32 actualRw = 0, actualRh = 0;
  getSupersampledRenderArea(renderTargets, &actualRw, &actualRh);
  auto width = actualRw;
  auto height = actualRh;
  auto outputWidth = renderTargets->getRenderArea().width;
  SceneCollectConfig sceneConfig;
  float jx, jy;
  if (config.m_antiAliasingType == AntiAliasingType::TAA) {
    sceneConfig.projectionTranslateX = (haltonX * 2.0f - 1.0f) / width;
    sceneConfig.projectionTranslateY = (haltonY * 2.0f - 1.0f) / height;
  } else if (config.m_antiAliasingType == AntiAliasingType::FSR2) {

    m_fsr2proc->getJitters(&jx, &jy, perframeData.m_frameId, actualRw, outputWidth);
    sceneConfig.projectionTranslateX = 2.0f * jx / width;

    // Note that we use Y Down in proj cam
    sceneConfig.projectionTranslateY = 2.0f * jy / height;
  } else {
    sceneConfig.projectionTranslateX = 0.0f;
    sceneConfig.projectionTranslateY = 0.0f;
  }

  // If debug views, ignore all jitters
  if (config.m_visualizationType != RendererVisualizationType::Default) {
    perframeData.m_taaJitterX = 0;
    perframeData.m_taaJitterY = 0;
    sceneConfig.projectionTranslateX = 0.0f;
    sceneConfig.projectionTranslateY = 0.0f;
  }
  setRendererConfig(&config);
  collectPerframeData(perframeData, scene, camera, GraphicsShaderPassType::Opaque, renderTargets, sceneConfig);

  auto end0 = std::chrono::high_resolution_clock::now();
  auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start);
  // iDebug("CPU time, frame collecting: {} ms", duration0.count());
  if (config.m_antiAliasingType == AntiAliasingType::TAA) {
    perframeData.m_taaJitterX = sceneConfig.projectionTranslateX * 0.5f;
    perframeData.m_taaJitterY = sceneConfig.projectionTranslateY * 0.5f;
  } else if (config.m_antiAliasingType == AntiAliasingType::FSR2) {
    perframeData.m_taaJitterX = jx;
    perframeData.m_taaJitterY = jy;
  } else {
    perframeData.m_taaJitterX = 0;
    perframeData.m_taaJitterY = 0;
  }

  // If debug views, ignore all jitters
  if (config.m_visualizationType != RendererVisualizationType::Default) {
    perframeData.m_taaJitterX = 0;
    perframeData.m_taaJitterY = 0;
    sceneConfig.projectionTranslateX = 0.0f;
    sceneConfig.projectionTranslateY = 0.0f;
  }

  auto ret = render(perframeData, renderTargets, cmdToWait);
  perframeData.m_frameId++;

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return ret;
}

} // namespace Ifrit::Core