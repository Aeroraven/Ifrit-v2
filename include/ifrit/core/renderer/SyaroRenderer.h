
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
#include "ifrit/common/util/Hash.h"
#include "postprocessing/PostFxAcesTonemapping.h"
#include "postprocessing/PostFxFFTConv2d.h"
#include "postprocessing/PostFxGaussianHori.h"
#include "postprocessing/PostFxGaussianVert.h"
#include "postprocessing/PostFxGlobalFog.h"
#include "postprocessing/PostFxStockhamDFT2.h"

namespace Ifrit::Core {
class IFRIT_APIDECL SyaroRenderer : public RendererBase {
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCommandSubmission = Ifrit::GraphicsBackend::Rhi::RhiTaskSubmission;
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBuffer;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using GPUDescRef = Ifrit::GraphicsBackend::Rhi::RhiBindlessDescriptorRef;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;
  using DrawPass = Ifrit::GraphicsBackend::Rhi::RhiGraphicsPass;
  using GPUShader = Ifrit::GraphicsBackend::Rhi::RhiShader;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using GPUColorRT = Ifrit::GraphicsBackend::Rhi::RhiColorAttachment;
  using GPURTs = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCmdBuffer = Ifrit::GraphicsBackend::Rhi::RhiCommandBuffer;
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSampler;

  enum class CullingPass { First, Second };

private:
  ComputePass *m_persistentCullingPass = nullptr;
  GPUBuffer *m_indirectDrawBuffer = nullptr;
  std::shared_ptr<GPUBindId> m_indirectDrawBufferId = nullptr;
  GPUDescRef *m_persistCullDesc = nullptr;

  DrawPass *m_visibilityPassHW = nullptr;
  DrawPass *m_depthOnlyVisibilityPassHW = nullptr;
  ComputePass *m_visibilityPassSW = nullptr;
  ComputePass *m_visibilityCombinePass = nullptr;

  // Instance culling
  ComputePass *m_instanceCullingPass = nullptr;

  // Single pass HiZ
  ComputePass *m_singlePassHiZPass = nullptr;
  constexpr static uint32_t cSPHiZGroupSizeX = 256;
  constexpr static uint32_t cSPHiZTileSize = 64;

  // Emit depth targets
  ComputePass *m_emitDepthTargetsPass = nullptr;
  constexpr static uint32_t cEmitDepthGroupSizeX = 16;
  constexpr static uint32_t cEmitDepthGroupSizeY = 16;

  // Material classify
  ComputePass *m_matclassCountPass = nullptr;
  ComputePass *m_matclassReservePass = nullptr;
  ComputePass *m_matclassScatterPass = nullptr;
  constexpr static uint32_t cMatClassQuadSize = 2;
  constexpr static uint32_t cMatClassGroupSizeCountScatterX = 8;
  constexpr static uint32_t cMatClassGroupSizeCountScatterY = 8;
  constexpr static uint32_t cMatClassGroupSizeReserveX = 128;
  constexpr static uint32_t cMatClassCounterBufferSizeBase =
      2 * sizeof(uint32_t);
  constexpr static uint32_t cMatClassCounterBufferSizeMult =
      2 * sizeof(uint32_t);

  // Emit GBuffer, pass here is for default / debugging
  ComputePass *m_defaultEmitGBufferPass = nullptr;

  // Perframe data maintained by the renderer, this is unsafe
  std::unordered_map<Scene *, PerFrameData> m_perScenePerframe;

  // TAA
  ComputePass *m_taaHistoryPass = nullptr;
  constexpr static Ifrit::GraphicsBackend::Rhi::RhiImageFormat cTAAFormat =
      Ifrit::GraphicsBackend::Rhi::RhiImageFormat::
          RHI_FORMAT_R32G32B32A32_SFLOAT;
  // Finally, deferred pass
  std::unordered_map<PipelineAttachmentConfigs, DrawPass *,
                     PipelineAttachmentConfigsHash>
      m_deferredShadingPass;

  DrawPass *m_deferredShadowPass = nullptr;

  std::unordered_map<PipelineAttachmentConfigs, DrawPass *,
                     PipelineAttachmentConfigsHash>
      m_taaPass;

  // FSR2
  std::unique_ptr<GraphicsBackend::Rhi::FSR2::RhiFsr2Processor> m_fsr2proc;

  // Atmosphere
  ComputePass *m_atmospherePass = nullptr;
  std::shared_ptr<PbrAtmosphereRenderer> m_atmosphereRenderer;

  // Timer
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiDeviceTimer> m_timer;
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiDeviceTimer> m_timerDefer;

  // AO
  std::shared_ptr<AmbientOcclusionPass> m_aoPass;

  // Postprocess, just 2 textures and 1 sampler is required.
  using PairHash = Ifrit::Common::Utility::PairwiseHash<uint32_t, uint32_t>;
  std::unordered_map<std::pair<uint32_t, uint32_t>,
                     std::array<std::shared_ptr<GPUTexture>, 2>, PairHash>
      m_postprocTex;
  std::unordered_map<std::pair<uint32_t, uint32_t>,
                     std::array<std::shared_ptr<GPUBindId>, 2>, PairHash>
      m_postprocTexId;
  std::unordered_map<std::pair<uint32_t, uint32_t>,
                     std::array<std::shared_ptr<GPUBindId>, 2>, PairHash>
      m_postprocTexIdComp;
  std::unordered_map<std::pair<uint32_t, uint32_t>,
                     std::array<std::shared_ptr<GPUColorRT>, 2>, PairHash>
      m_postprocColorRT;
  std::unordered_map<std::pair<uint32_t, uint32_t>,
                     std::array<std::shared_ptr<GPURTs>, 2>, PairHash>
      m_postprocRTs;
  std::shared_ptr<GPUSampler> m_postprocTexSampler;
  std::shared_ptr<GPUBindId> m_postprocTexSamplerId;

  // All postprocess passes required
  std::unique_ptr<PostprocessPassCollection::PostFxAcesToneMapping>
      m_acesToneMapping;
  std::unique_ptr<PostprocessPassCollection::PostFxGlobalFog> m_globalFogPass;
  std::unique_ptr<PostprocessPassCollection::PostFxGaussianHori> m_gaussianHori;
  std::unique_ptr<PostprocessPassCollection::PostFxGaussianVert> m_gaussianVert;
  std::unique_ptr<PostprocessPassCollection::PostFxStockhamDFT2> m_stockhamDFT2;
  std::unique_ptr<PostprocessPassCollection::PostFxFFTConv2d> m_fftConv2d;

  // Render config
  RendererConfig m_renderConfig;

private:
  // Util functions
  GPUShader *createShaderFromFile(const std::string &shaderPath,
                                  const std::string &entry,
                                  GraphicsBackend::Rhi::RhiShaderStage stage);

  // Setup functions
  void recreateInstanceCullingBuffers(PerFrameData &perframe,
                                      uint32_t newMaxInstances);
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

  void hizBufferSetup(PerFrameData &perframeData, RenderTargets *renderTargets);
  void sphizBufferSetup(PerFrameData &perframeData,
                        RenderTargets *renderTargets);
  void visibilityBufferSetup(PerFrameData &perframeData,
                             RenderTargets *renderTargets);
  void depthTargetsSetup(PerFrameData &perframeData,
                         RenderTargets *renderTargets);
  void materialClassifyBufferSetup(PerFrameData &perframeData,
                                   RenderTargets *renderTargets);
  void taaHistorySetup(PerFrameData &perframeData,
                       RenderTargets *renderTargets);
  void createPostprocessTextures(uint32_t width, uint32_t height);
  void prepareAggregatedShadowData(PerFrameData &perframeData);

  // Many passes are not material-dependent, so a unified instance buffer
  // might reduce calls
  void gatherAllInstances(PerFrameData &perframeData);

  PerFrameData::PerViewData &getPrimaryView(PerFrameData &perframeData);

private:
  // Decompose the rendering procedure into many parts
  void renderTwoPassOcclCulling(CullingPass cullPass,
                                PerFrameData &perframeData,
                                RenderTargets *renderTargets,
                                const GPUCmdBuffer *cmd,
                                PerFrameData::ViewType filteredViewType,
                                uint32_t idx);

  void renderEmitDepthTargets(PerFrameData &perframeData,
                              RenderTargets *renderTargets,
                              const GPUCmdBuffer *cmd);

  void renderMaterialClassify(PerFrameData &perframeData,
                              RenderTargets *renderTargets,
                              const GPUCmdBuffer *cmd);

  // This is for debugging. The proc should be material-specific
  void renderDefaultEmitGBuffer(PerFrameData &perframeData,
                                RenderTargets *renderTargets,
                                const GPUCmdBuffer *cmd);

  void renderAmbientOccl(PerFrameData &perframeData,
                         RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

  // Frame graph
  void setupAndRunFrameGraph(PerFrameData &perframeData,
                             RenderTargets *renderTargets,
                             const GPUCmdBuffer *cmd);

  virtual std::unique_ptr<GPUCommandSubmission>
  render(PerFrameData &perframeData, RenderTargets *renderTargets,
         const std::vector<GPUCommandSubmission *> &cmdToWait);

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

  virtual std::unique_ptr<GPUCommandSubmission>
  render(Scene *scene, Camera *camera, RenderTargets *renderTargets,
         const RendererConfig &config,
         const std::vector<GPUCommandSubmission *> &cmdToWait) override;
};
} // namespace Ifrit::Core