
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
  uint32_t m_defaultShadowMapSize = 2048;
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