#pragma once
#include "RendererBase.h"
namespace Ifrit::Core {
class IFRIT_APIDECL SimpleRenderer : public RendererBase {
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCommandSubmission = Ifrit::GraphicsBackend::Rhi::RhiTaskSubmission;
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBuffer;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using GPUDescRef = Ifrit::GraphicsBackend::Rhi::RhiBindlessDescriptorRef;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;

private:
  ComputePass *m_cullingPass = nullptr;
  GPUBuffer *m_indirectDrawBuffer = nullptr;
  std::shared_ptr<GPUBindId> m_indirectDrawBufferId = nullptr;
  GPUDescRef *m_cullingDescriptor = nullptr;

public:
  SimpleRenderer(IApplication *app) : RendererBase(app) {}
  void setupCullingPass();
  virtual std::unique_ptr<GPUCommandSubmission>
  render(PerFrameData &perframeData, RenderTargets *renderTargets,
         const std::vector<GPUCommandSubmission *> &cmdToWait) override;
};
} // namespace Ifrit::Core