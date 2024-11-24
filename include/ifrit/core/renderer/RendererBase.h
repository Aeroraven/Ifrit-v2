#pragma once
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/base/Scene.h"
#include "ifrit/core/scene/FrameCollector.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {

struct SceneCollectConfig {
  float projectionTranslateX = 0.0f;
  float projectionTranslateY = 0.0f;
};

enum class AntiAliasingType { None, TAA };

struct RendererConfig {
  AntiAliasingType m_antiAliasingType = AntiAliasingType::None;
};

// TODO: move render graph to here
class IFRIT_APIDECL RendererBase {
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCommandSubmission = Ifrit::GraphicsBackend::Rhi::RhiTaskSubmission;

protected:
  IApplication *m_app;

protected:
  RendererBase(IApplication *app) : m_app(app) {}
  virtual void buildPipelines(PerFrameData &perframeData,
                              GraphicsShaderPassType passType,
                              RenderTargets *renderTargets);
  virtual void prepareDeviceResources(PerFrameData &perframeData,
                                      RenderTargets *renderTargets);
  virtual void updateLastFrameTransforms(PerFrameData &perframeData);
  virtual void recreateGBuffers(PerFrameData &perframeData,
                                RenderTargets *renderTargets);

  virtual void collectPerframeData(PerFrameData &perframeData, Scene *scene,
                                   Camera *camera,
                                   GraphicsShaderPassType passType,
                                   const SceneCollectConfig &config);

public:
  virtual std::unique_ptr<GPUCommandSubmission>
  render(Scene *scene, Camera *camera, RenderTargets *renderTargets,
         const RendererConfig &config,
         const std::vector<GPUCommandSubmission *> &cmdToWait) = 0;

  virtual void endFrame(const std::vector<GPUCommandSubmission *> &cmdToWait);
  virtual std::unique_ptr<GPUCommandSubmission> beginFrame();
};
} // namespace Ifrit::Core