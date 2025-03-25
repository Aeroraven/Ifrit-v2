
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

#include "ifrit/common/base/IfritBase.h"

#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/renderer/RendererUtil.h"
#include "ifrit/core/renderer/SyaroRenderer.h"
#include "ifrit/core/renderer/util/RenderingUtils.h"
#include <algorithm>
#include <bit>

using namespace Ifrit::GraphicsBackend::Rhi;
using Ifrit::Common::Utility::size_cast;
using Ifrit::Math::ConstFunc::divRoundUp;

namespace Ifrit::Core {

struct AyanamiRendererResources;
class IFRIT_APIDECL AyanamiRenderer : public RendererBase {
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCommandSubmission = Ifrit::GraphicsBackend::Rhi::RhiTaskSubmission;
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBuffer;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiDescHandleLegacy;
  using GPUDescRef = Ifrit::GraphicsBackend::Rhi::RhiBindlessDescriptorRef;
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;
  using DrawPass = Ifrit::GraphicsBackend::Rhi::RhiGraphicsPass;
  using GPUShader = Ifrit::GraphicsBackend::Rhi::RhiShader;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using GPUColorRT = Ifrit::GraphicsBackend::Rhi::RhiColorAttachment;
  using GPURTs = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUCmdBuffer = Ifrit::GraphicsBackend::Rhi::RhiCommandList;
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSampler;

  // Perframe data maintained by the renderer, this is unsafe
  // This will be dropped in the future
  std::unordered_map<Scene *, PerFrameData> m_perScenePerframe;

private:
  std::unique_ptr<SyaroRenderer> m_gbufferRenderer;
  AyanamiRendererResources *m_resources = nullptr;

private:
  void initRenderer();
  void prepareResources(RenderTargets *renderTargets, const RendererConfig &config);
  void setupAndRunFrameGraph(PerFrameData &perframe, RenderTargets *renderTargets, const GPUCmdBuffer *cmd);

public:
  AyanamiRenderer(IApplication *app) : RendererBase(app), m_gbufferRenderer(std::make_unique<SyaroRenderer>(app)) {
    m_gbufferRenderer->setRenderRole(SyaroRenderRole::SYARO_DEFERRED_GBUFFER);
    initRenderer();
  }
  virtual ~AyanamiRenderer();
  virtual std::unique_ptr<GPUCommandSubmission> render(Scene *scene, Camera *camera, RenderTargets *renderTargets,
                                                       const RendererConfig &config,
                                                       const std::vector<GPUCommandSubmission *> &cmdToWait) override;
};

} // namespace Ifrit::Core