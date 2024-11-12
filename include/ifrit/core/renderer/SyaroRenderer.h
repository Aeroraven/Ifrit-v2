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

private:
  ComputePass *m_cullingPass = nullptr;
  GPUBuffer *m_indirectDrawBuffer = nullptr;
  std::shared_ptr<GPUBindId> m_indirectDrawBufferId = nullptr;
  GPUDescRef *m_cullingDescriptor = nullptr;

  // For debugging
  DrawPass *m_textureShowPass = nullptr; // I think renderdoc can do this, but
                                         // this is for quick debugging

private:
  void setupCullingPass();
  void setupTextureShowPass();
  void visbilityBufferSetup(PerFrameData &perframeData,
                            RenderTargets *renderTargets);

public:
  SyaroRenderer(IApplication *app) : RendererBase(app) {
    setupCullingPass();
    setupTextureShowPass();
  }
  virtual std::unique_ptr<GPUCommandSubmission>
  render(PerFrameData &perframeData, RenderTargets *renderTargets,
         const std::vector<GPUCommandSubmission *> &cmdToWait) override;
};
} // namespace Ifrit::Core