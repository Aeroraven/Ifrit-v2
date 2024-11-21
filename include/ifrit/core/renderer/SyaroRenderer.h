#pragma once
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

private:
  // Util functions
  GPUShader *createShaderFromFile(const std::string &shaderPath,
                                  const std::string &entry,
                                  GraphicsBackend::Rhi::RhiShaderStage stage);

  // Setup functions
  void recreateInstanceCullingBuffers(PerFrameData& perframe,
                                      uint32_t newMaxInstances);
  void setupInstanceCullingPass();
  void setupPersistentCullingPass();
  void setupVisibilityPass();
  void setupTextureShowPass();
  void setupHiZPass();
  void setupEmitDepthTargetsPass();
  void setupMaterialClassifyPass();
  void setupDefaultEmitGBufferPass();

  void hizBufferSetup(PerFrameData &perframeData, RenderTargets *renderTargets);
  void visibilityBufferSetup(PerFrameData &perframeData,
                             RenderTargets *renderTargets);
  void depthTargetsSetup(PerFrameData &perframeData,
                         RenderTargets *renderTargets);
  void materialClassifyBufferSetup(PerFrameData &perframeData,
                                   RenderTargets *renderTargets);

  // Many passes are not material-dependent, so a unified instance buffer
  // might reduce calls
  void gatherAllInstances(PerFrameData &perframeData);

  PerFrameData::PerViewData &getPrimaryView(PerFrameData &perframeData);

private:
  // Decompose the rendering procedure into many parts
  std::unique_ptr<GPUCommandSubmission> renderTwoPassOcclCulling(
      CullingPass cullPass, PerFrameData &perframeData,
      RenderTargets *renderTargets,
      const std::vector<GPUCommandSubmission *> &cmdToWait);

  std::unique_ptr<GPUCommandSubmission>
  renderEmitDepthTargets(PerFrameData &perframeData,
                         RenderTargets *renderTargets,
                         const std::vector<GPUCommandSubmission *> &cmdToWait);

  std::unique_ptr<GPUCommandSubmission>
  renderMaterialClassify(PerFrameData &perframeData,
                         RenderTargets *renderTargets,
                         const std::vector<GPUCommandSubmission *> &cmdToWait);

  // This is for debugging. The proc should be material-specific
  std::unique_ptr<GPUCommandSubmission> renderDefaultEmitGBuffer(
      PerFrameData &perframeData, RenderTargets *renderTargets,
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
  }
  virtual std::unique_ptr<GPUCommandSubmission>
  render(PerFrameData &perframeData, RenderTargets *renderTargets,
         const std::vector<GPUCommandSubmission *> &cmdToWait) override;

  virtual std::unique_ptr<GPUCommandSubmission>
  render(Scene *scene, Camera *camera, RenderTargets *renderTargets,
         const std::vector<GPUCommandSubmission *> &cmdToWait) override;
};
} // namespace Ifrit::Core