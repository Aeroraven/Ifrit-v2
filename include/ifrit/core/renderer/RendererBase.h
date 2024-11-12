#pragma once
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/scene/FrameCollector.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {

// TODO: move render graph to here
class IFRIT_APIDECL RendererBase {
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCommandSubmission = Ifrit::GraphicsBackend::Rhi::RhiTaskSubmission;

protected:
  IApplication *m_app;

public:
  RendererBase(IApplication *app) : m_app(app) {}
  virtual void buildPipelines(PerFrameData &perframeData,
                              GraphicsShaderPassType passType,
                              RenderTargets *renderTargets);
  virtual void prepareDeviceResources(PerFrameData &perframeData);
  virtual std::unique_ptr<GPUCommandSubmission>
  render(PerFrameData &perframeData, RenderTargets *renderTargets,
         const std::vector<GPUCommandSubmission *> &cmdToWait) = 0;

  virtual void endFrame(const std::vector<GPUCommandSubmission *> &cmdToWait);
  virtual std::unique_ptr<GPUCommandSubmission> beginFrame();
};
} // namespace Ifrit::Core