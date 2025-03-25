
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

#pragma once
#include "AmbientOcclusionPass.h"
#include "PbrAtmosphereRenderer.h"
#include "RendererBase.h"
#include "framegraph/FrameGraph.h"
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/Hash.h"
#include "postprocessing/PostFxAcesTonemapping.h"
#include "postprocessing/PostFxFFTConv2d.h"
#include "postprocessing/PostFxGaussianHori.h"
#include "postprocessing/PostFxGaussianVert.h"
#include "postprocessing/PostFxGlobalFog.h"
#include "postprocessing/PostFxJointBilaterialFilter.h"
#include "postprocessing/PostFxStockhamDFT2.h"

#include "commonpass/SinglePassHiZ.h"

namespace Ifrit::Core {

enum class SyaroRenderRole { SYARO_FULL, SYARO_DEFERRED_GBUFFER };

class IFRIT_APIDECL SyaroRenderer : public RendererBase {
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCommandSubmission = Ifrit::GraphicsBackend::Rhi::RhiTaskSubmission;
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBufferRef;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiDescHandleLegacy;
  using GPUDescRef = Ifrit::GraphicsBackend::Rhi::RhiBindlessDescriptorRef;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;
  using DrawPass = Ifrit::GraphicsBackend::Rhi::RhiGraphicsPass;
  using GPUShader = Ifrit::GraphicsBackend::Rhi::RhiShader;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTextureRef;
  using GPUColorRT = Ifrit::GraphicsBackend::Rhi::RhiColorAttachment;
  using GPURTs = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCmdBuffer = Ifrit::GraphicsBackend::Rhi::RhiCommandList;
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSamplerRef;

  enum class CullingPass { First, Second };

private:
  // Renderer Role
  SyaroRenderRole m_renderRole = SyaroRenderRole::SYARO_FULL;

  // Base
  ComputePass *m_persistentCullingPass = nullptr;
  GPUBuffer m_indirectDrawBuffer = nullptr;
  GPUDescRef *m_persistCullDesc = nullptr;

  DrawPass *m_visibilityPassHW = nullptr;
  DrawPass *m_depthOnlyVisibilityPassHW = nullptr;
  ComputePass *m_visibilityPassSW = nullptr;
  ComputePass *m_visibilityCombinePass = nullptr;

  // Instance culling
  ComputePass *m_instanceCullingPass = nullptr;

  // Single pass HiZ
  Ref<SinglePassHiZPass> m_singlePassHiZProc = nullptr;

  IF_CONSTEXPR static u32 cSPHiZGroupSizeX = 256;
  IF_CONSTEXPR static u32 cSPHiZTileSize = 64;

  // Emit depth targets
  ComputePass *m_emitDepthTargetsPass = nullptr;
  IF_CONSTEXPR static u32 cEmitDepthGroupSizeX = 16;
  IF_CONSTEXPR static u32 cEmitDepthGroupSizeY = 16;

  // Material classify
  ComputePass *m_matclassCountPass = nullptr;
  ComputePass *m_matclassReservePass = nullptr;
  ComputePass *m_matclassScatterPass = nullptr;
  IF_CONSTEXPR static u32 cMatClassQuadSize = 2;
  IF_CONSTEXPR static u32 cMatClassGroupSizeCountScatterX = 8;
  IF_CONSTEXPR static u32 cMatClassGroupSizeCountScatterY = 8;
  IF_CONSTEXPR static u32 cMatClassGroupSizeReserveX = 128;
  IF_CONSTEXPR static u32 cMatClassCounterBufferSizeBase = 2 * sizeof(u32);
  IF_CONSTEXPR static u32 cMatClassCounterBufferSizeMult = 2 * sizeof(u32);

  // Emit GBuffer, pass here is for default / debugging
  ComputePass *m_defaultEmitGBufferPass = nullptr;

  // Perframe data maintained by the renderer, this is unsafe
  HashMap<Scene *, PerFrameData> m_perScenePerframe;

  // TAA
  ComputePass *m_taaHistoryPass = nullptr;
  IF_CONSTEXPR static Ifrit::GraphicsBackend::Rhi::RhiImageFormat cTAAFormat =
      Ifrit::GraphicsBackend::Rhi::RhiImageFormat::RhiImgFmt_R32G32B32A32_SFLOAT;

  // Finally, deferred pass
  CustomHashMap<PipelineAttachmentConfigs, DrawPass *, PipelineAttachmentConfigsHash> m_deferredShadingPass;
  DrawPass *m_deferredShadowPass = nullptr;
  CustomHashMap<PipelineAttachmentConfigs, DrawPass *, PipelineAttachmentConfigsHash> m_taaPass;

  // FSR2
  Uref<GraphicsBackend::Rhi::FSR2::RhiFsr2Processor> m_fsr2proc;

  // Atmosphere
  ComputePass *m_atmospherePass = nullptr;
  Ref<PbrAtmosphereRenderer> m_atmosphereRenderer;

  // Timer
  Ref<Ifrit::GraphicsBackend::Rhi::RhiDeviceTimer> m_timer;
  Ref<Ifrit::GraphicsBackend::Rhi::RhiDeviceTimer> m_timerDefer;

  // AO
  Ref<AmbientOcclusionPass> m_aoPass;

  // Postprocess, just 2 textures and 1 sampler is required.
  using PairHash = Ifrit::Common::Utility::PairwiseHash<u32, u32>;
  std::unordered_map<std::pair<u32, u32>, std::array<GPUTexture, 2>, PairHash> m_postprocTex;
  std::unordered_map<std::pair<u32, u32>, std::array<Ref<GPUBindId>, 2>, PairHash> m_postprocTexSRV;
  std::unordered_map<std::pair<u32, u32>, std::array<Ref<GPUColorRT>, 2>, PairHash> m_postprocColorRT;
  std::unordered_map<std::pair<u32, u32>, std::array<Ref<GPURTs>, 2>, PairHash> m_postprocRTs;
  GPUSampler m_postprocTexSampler;
  Ref<GPUBindId> m_postprocTexSamplerId;

  // All postprocess passes required
  Uref<PostprocessPassCollection::PostFxAcesToneMapping> m_acesToneMapping;
  Uref<PostprocessPassCollection::PostFxGlobalFog> m_globalFogPass;
  Uref<PostprocessPassCollection::PostFxGaussianHori> m_gaussianHori;
  Uref<PostprocessPassCollection::PostFxGaussianVert> m_gaussianVert;
  Uref<PostprocessPassCollection::PostFxFFTConv2d> m_fftConv2d;
  Uref<PostprocessPassCollection::PostFxJointBilaterialFilter> m_jointBilateralFilter;

  // Intermediate views
  std::unordered_map<PipelineAttachmentConfigs, DrawPass *, PipelineAttachmentConfigsHash> m_triangleViewPass;

  // Render config
  RendererConfig m_renderConfig;

private:
  // Util functions
  GPUShader *createShaderFromFile(const String &shaderPath, const String &entry,
                                  GraphicsBackend::Rhi::RhiShaderStage stage);

  // Setup functions
  void recreateInstanceCullingBuffers(PerFrameData &perframe, u32 newMaxInstances);
  void setupInstanceCullingPass();
  void setupPersistentCullingPass();
  void setupVisibilityPass();
  void setupEmitDepthTargetsPass();
  void setupMaterialClassifyPass();
  void setupDefaultEmitGBufferPass();
  void setupPbrAtmosphereRenderer();
  void setupSinglePassHiZPass();
  void setupFSR2Data();
  void setupPostprocessPassAndTextures();
  void createTimer();

  void setupDeferredShadingPass(RenderTargets *renderTargets);
  void setupTAAPass(RenderTargets *renderTargets);

  void sphizBufferSetup(PerFrameData &perframeData, RenderTargets *renderTargets);
  void visibilityBufferSetup(PerFrameData &perframeData, RenderTargets *renderTargets);
  void depthTargetsSetup(PerFrameData &perframeData, RenderTargets *renderTargets);
  void materialClassifyBufferSetup(PerFrameData &perframeData, RenderTargets *renderTargets);
  void taaHistorySetup(PerFrameData &perframeData, RenderTargets *renderTargets);
  void fsr2Setup(PerFrameData &perframeData, RenderTargets *renderTargets);
  void createPostprocessTextures(u32 width, u32 height);
  void prepareAggregatedShadowData(PerFrameData &perframeData);

  void setupDebugPasses(PerFrameData &perframeData, RenderTargets *renderTargets);

  // Many passes are not material-dependent, so a unified instance buffer
  // might reduce calls
  void gatherAllInstances(PerFrameData &perframeData);

  PerFrameData::PerViewData &getPrimaryView(PerFrameData &perframeData);

private:
  // Decompose the rendering procedure into many parts
  void renderTwoPassOcclCulling(CullingPass cullPass, PerFrameData &perframeData, RenderTargets *renderTargets,
                                const GPUCmdBuffer *cmd, PerFrameData::ViewType filteredViewType, u32 idx);

  void renderTriangleView(PerFrameData &perframeData, RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

  void renderEmitDepthTargets(PerFrameData &perframeData, RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

  void renderMaterialClassify(PerFrameData &perframeData, RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

  // This is for debugging. The proc should be material-specific
  void renderDefaultEmitGBuffer(PerFrameData &perframeData, RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

  void renderAmbientOccl(PerFrameData &perframeData, RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

  // Frame graph
  void setupAndRunFrameGraph(PerFrameData &perframeData, RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

  virtual Uref<GPUCommandSubmission> render(PerFrameData &perframeData, RenderTargets *renderTargets,
                                            const Vec<GPUCommandSubmission *> &cmdToWait);

public:
  SyaroRenderer(IApplication *app) : RendererBase(app) {
    setupPersistentCullingPass();
    setupVisibilityPass();
    setupInstanceCullingPass();
    setupEmitDepthTargetsPass();
    setupMaterialClassifyPass();
    setupDefaultEmitGBufferPass();
    setupSinglePassHiZPass();
    setupPbrAtmosphereRenderer();
    setupPostprocessPassAndTextures();
    createTimer();

    m_aoPass = std::make_shared<AmbientOcclusionPass>(app);
  }
  inline void setRenderRole(SyaroRenderRole role) { m_renderRole = role; }
  inline PerFrameData getPerframeData(Scene *scene) { return m_perScenePerframe[scene]; }
  virtual Uref<GPUCommandSubmission> render(Scene *scene, Camera *camera, RenderTargets *renderTargets,
                                            const RendererConfig &config,
                                            const Vec<GPUCommandSubmission *> &cmdToWait) override;
};
} // namespace Ifrit::Core