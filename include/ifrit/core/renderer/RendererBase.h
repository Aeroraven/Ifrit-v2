
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
#include "ifrit/core/renderer/util/CascadeShadowMapPreproc.h"

#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/base/Scene.h"

#include "ifrit/core/scene/FrameCollector.h"
#include "ifrit/rhi/common/RhiLayer.h"

#include <mutex>

namespace Ifrit::Core {

struct SceneCollectConfig {
  float projectionTranslateX = 0.0f;
  float projectionTranslateY = 0.0f;
};

struct ImmutableRendererResources {
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSampler;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  std::mutex m_mutex;
  bool m_initialized = false;
  std::shared_ptr<GPUSampler> m_linearSampler = nullptr;
  std::shared_ptr<GPUSampler> m_nearestSampler = nullptr;

  std::shared_ptr<GPUTexture> m_blueNoise = nullptr;
  std::shared_ptr<GPUBindId> m_blueNoiseSRV = nullptr;
};

enum class AntiAliasingType { None, TAA, FSR2 };
enum class RendererVisualizationType { Default, Triangle, SwHwMaps };
enum class IndirectLightingType { HBAO, SSGI };

struct RendererConfig {
  struct ShadowConfig {
    constexpr static uint32_t k_maxShadowMaps = 256;
    float m_maxDistance = 5.0f;
    uint32_t m_csmCount = 4;
    std::array<float, 4> m_csmSplits = {0.067f, 0.133f, 0.267f, 0.533f};
    std::array<float, 4> m_csmBorders = {0.08f, 0.05f, 0.0f, 0.0f};
  };

  AntiAliasingType m_antiAliasingType = AntiAliasingType::None;
  IndirectLightingType m_indirectLightingType = IndirectLightingType::HBAO;
  RendererVisualizationType m_visualizationType =
      RendererVisualizationType::Default;
  ShadowConfig m_shadowConfig;
  float m_superSamplingRate = 1.0f;
};

// TODO: move render graph to here
class IFRIT_APIDECL RendererBase {
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCommandSubmission = Ifrit::GraphicsBackend::Rhi::RhiTaskSubmission;

protected:
  IApplication *m_app;
  const RendererConfig *m_config = nullptr;
  ImmutableRendererResources m_immRes;

protected:
  RendererBase(IApplication *app) : m_app(app) {}

  inline void getSupersampledRenderArea(const RenderTargets *finalRenderTargets,
                                        uint32_t *renderWidth,
                                        uint32_t *renderHeight) {
    *renderWidth =
        static_cast<uint32_t>(finalRenderTargets->getRenderArea().width /
                              m_config->m_superSamplingRate);
    *renderHeight =
        static_cast<uint32_t>(finalRenderTargets->getRenderArea().height /
                              m_config->m_superSamplingRate);
  }

  virtual void prepareImmutableResources();

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
                                   RenderTargets *renderTargets,
                                   const SceneCollectConfig &config);

  inline void setRendererConfig(const RendererConfig *config) {
    m_config = config;
  }

public:
  virtual std::unique_ptr<GPUCommandSubmission>
  render(Scene *scene, Camera *camera, RenderTargets *renderTargets,
         const RendererConfig &config,
         const std::vector<GPUCommandSubmission *> &cmdToWait) = 0;

  virtual void endFrame(const std::vector<GPUCommandSubmission *> &cmdToWait);
  virtual std::unique_ptr<GPUCommandSubmission> beginFrame();
};
} // namespace Ifrit::Core