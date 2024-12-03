#pragma once
#include "PbrAtmosphereRenderer.h"
#include "RendererBase.h"

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

  enum class CullingPass { First, Second };

private:
  ComputePass *m_persistentCullingPass = nullptr;
  GPUBuffer *m_indirectDrawBuffer = nullptr;
  std::shared_ptr<GPUBindId> m_indirectDrawBufferId = nullptr;
  GPUDescRef *m_persistCullDesc = nullptr;

  DrawPass *m_visibilityPass = nullptr;
  // I think renderdoc can do this, but this is for quick debugging
  DrawPass *m_textureShowPass = nullptr;

  // Instance culling
  ComputePass *m_instanceCullingPass = nullptr;

  // HiZ buffer
  ComputePass *m_hizPass = nullptr;
  constexpr static uint32_t cHiZGroupSizeX = 16;
  constexpr static uint32_t cHiZGroupSizeY = 16;

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
  ComputePass *m_matclassDebugPass = nullptr;
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

  std::unordered_map<PipelineAttachmentConfigs, DrawPass *,
                     PipelineAttachmentConfigsHash>
      m_taaPass;

  // Atmosphere
  ComputePass *m_atmospherePass = nullptr;
  std::shared_ptr<PbrAtmosphereRenderer> m_atmosphereRenderer;

  // Timer
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiDeviceTimer> m_timer;

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
  void setupTextureShowPass();
  void setupHiZPass();
  void setupEmitDepthTargetsPass();
  void setupMaterialClassifyPass();
  void setupDefaultEmitGBufferPass();
  void setupPbrAtmosphereRenderer();

  void setupSinglePassHiZPass();
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

  // Many passes are not material-dependent, so a unified instance buffer
  // might reduce calls
  void gatherAllInstances(PerFrameData &perframeData);

  PerFrameData::PerViewData &getPrimaryView(PerFrameData &perframeData);

private:
  // Decompose the rendering procedure into many parts
  void renderTwoPassOcclCulling(CullingPass cullPass,
                                PerFrameData &perframeData,
                                RenderTargets *renderTargets,
                                const GPUCmdBuffer *cmd);

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

  void renderDeferredShading(PerFrameData &perframeData,
                             RenderTargets *renderTargets,
                             const GPUCmdBuffer *cmd);

  void renderTAAResolve(PerFrameData &perframeData,
                        RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

  void renderAtmosphere(PerFrameData &perframeData,
                        RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

  virtual std::unique_ptr<GPUCommandSubmission>
  render(PerFrameData &perframeData, RenderTargets *renderTargets,
         const std::vector<GPUCommandSubmission *> &cmdToWait);

public:
  SyaroRenderer(IApplication *app) : RendererBase(app) {
    setupPersistentCullingPass();
    setupTextureShowPass();
    setupVisibilityPass();
    setupInstanceCullingPass();
    setupHiZPass();
    setupEmitDepthTargetsPass();
    setupMaterialClassifyPass();
    setupDefaultEmitGBufferPass();
    setupSinglePassHiZPass();
    setupPbrAtmosphereRenderer();
    createTimer();
  }

  virtual std::unique_ptr<GPUCommandSubmission>
  render(Scene *scene, Camera *camera, RenderTargets *renderTargets,
         const RendererConfig &config,
         const std::vector<GPUCommandSubmission *> &cmdToWait) override;
};
} // namespace Ifrit::Core