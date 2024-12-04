
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
struct PostprocessPassConfig {
  std::string fragPath;
  uint32_t numPushConstants;
  uint32_t numDescriptorSets;
};

class IFRIT_APIDECL PostprocessPass {
protected:
  using DrawPass = Ifrit::GraphicsBackend::Rhi::RhiGraphicsPass;
  using RenderTargets = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUShader = Ifrit::GraphicsBackend::Rhi::RhiShader;
  using GPUCmdBuffer = Ifrit::GraphicsBackend::Rhi::RhiCommandBuffer;
  using GPUBindlessRef = Ifrit::GraphicsBackend::Rhi::RhiBindlessDescriptorRef;

protected:
  PostprocessPassConfig m_cfg;
  IApplication *m_app;
  std::unordered_map<PipelineAttachmentConfigs, DrawPass *,
                     PipelineAttachmentConfigsHash>
      m_renderPipelines;

  GPUShader *createShaderFromFile(const std::string &shaderPath,
                                  const std::string &entry,
                                  GraphicsBackend::Rhi::RhiShaderStage stage);
  DrawPass *setupRenderPipeline(RenderTargets *renderTargets);

protected:
  void renderInternal(PerFrameData *perframeData, RenderTargets *renderTargets,
                      const GPUCmdBuffer *cmd, const void *pushConstants,
                      const std::vector<GPUBindlessRef *> &bindDescs,
                      const std::string &scopeName);

public:
  PostprocessPass(IApplication *app, const PostprocessPassConfig &cfg);
  virtual ~PostprocessPass() = default;
};
} // namespace Ifrit::Core
