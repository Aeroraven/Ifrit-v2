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

private:
  ComputePass *m_cullingPass = nullptr;
  GPUBuffer *m_indirectDrawBuffer = nullptr;
  std::shared_ptr<GPUBindId> m_indirectDrawBufferId = nullptr;
  GPUDescRef *m_cullingDescriptor = nullptr;

  DrawPass *m_visibilityPass = nullptr;
  DrawPass *m_textureShowPass = nullptr; // I think renderdoc can do this, but
                                         // this is for quick debugging

private:
  // Util functions
  GPUShader *createShaderFromFile(const std::string &shaderPath,
                                  const std::string &entry,
                                  GraphicsBackend::Rhi::RhiShaderStage stage);

  // Setup functions
  void setupPersistentCullingPass();
  void setupVisibilityPass();
  void setupTextureShowPass();
  void visbilityBufferSetup(PerFrameData &perframeData,
                            RenderTargets *renderTargets);

  // Many passes are not material-dependent, so a unified instance buffer might
  // reduce calls
  void gatherAllInstances(PerFrameData &perframeData);

public:
  SyaroRenderer(IApplication *app) : RendererBase(app) {
    setupPersistentCullingPass();
    setupTextureShowPass();
    setupVisibilityPass();
  }
  virtual std::unique_ptr<GPUCommandSubmission>
  render(PerFrameData &perframeData, RenderTargets *renderTargets,
         const std::vector<GPUCommandSubmission *> &cmdToWait) override;
};
} // namespace Ifrit::Core