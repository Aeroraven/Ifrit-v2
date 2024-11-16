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
  GPUBuffer *m_instCullDiscardObj = nullptr;
  GPUBuffer *m_instCullPassedObj = nullptr;
  GPUBuffer *m_persistCullIndirectDispatch = nullptr;
  GPUDescRef *m_instCullDesc = nullptr;
  uint32_t m_maxSupportedInstances = 0;

  // HiZ buffer
  ComputePass *m_hizPass = nullptr;
  constexpr static uint32_t cHiZGroupSizeX = 16;
  constexpr static uint32_t cHiZGroupSizeY = 16;

private:
  // Util functions
  GPUShader *createShaderFromFile(const std::string &shaderPath,
                                  const std::string &entry,
                                  GraphicsBackend::Rhi::RhiShaderStage stage);

  // Setup functions
  void recreateInstanceCullingBuffers(uint32_t newMaxInstances);
  void setupInstanceCullingPass();
  void setupPersistentCullingPass();
  void setupVisibilityPass();
  void setupTextureShowPass();
  void setupHiZPass();

  void hizBufferSetup(PerFrameData &perframeData, RenderTargets *renderTargets);
  void visibilityBufferSetup(PerFrameData &perframeData,
                             RenderTargets *renderTargets);

  // Many passes are not material-dependent, so a unified instance buffer might
  // reduce calls
  void gatherAllInstances(PerFrameData &perframeData);

private:
  // Decompose the rendering procedure into many parts
  virtual std::unique_ptr<GPUCommandSubmission> renderTwoPassOcclCulling(
      CullingPass cullPass, PerFrameData &perframeData,
      RenderTargets *renderTargets,
      const std::vector<GPUCommandSubmission *> &cmdToWait);

public:
  SyaroRenderer(IApplication *app) : RendererBase(app) {
    setupPersistentCullingPass();
    setupTextureShowPass();
    setupVisibilityPass();
    setupInstanceCullingPass();
    setupHiZPass();
  }
  virtual std::unique_ptr<GPUCommandSubmission>
  render(PerFrameData &perframeData, RenderTargets *renderTargets,
         const std::vector<GPUCommandSubmission *> &cmdToWait) override;
};
} // namespace Ifrit::Core