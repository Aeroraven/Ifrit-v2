
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
class IFRIT_APIDECL AmbientOcclusionPass {
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;
  using CommandBuffer = Ifrit::GraphicsBackend::Rhi::RhiCommandBuffer;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using GPUShader = Ifrit::GraphicsBackend::Rhi::RhiShader;

private:
  IApplication *m_app;
  ComputePass *m_hbaoPass = nullptr;
  ComputePass *m_ssgiPass = nullptr;

  void setupHBAOPass();
  void setupSSGIPass();
  GPUShader *createShaderFromFile(const std::string &shaderPath,
                                  const std::string &entry,
                                  GraphicsBackend::Rhi::RhiShaderStage stage);

public:
  AmbientOcclusionPass(IApplication *app) : m_app(app) {}
  void renderHBAO(const CommandBuffer *cmd, uint32_t width, uint32_t height,
                  GPUBindId *depthSamp, GPUBindId *normalSamp, GPUBindId *aoTex,
                  GPUBindId *perframeData);

  void renderSSGI(const CommandBuffer *cmd, uint32_t width, uint32_t height,
                  GPUBindId *perframeData, GPUBindId *depthHizUAV,
                  GPUBindId *normalSRV, GPUBindId *aoUAV, GPUBindId *albedoSRV,
                  uint32_t hizTexW, uint32_t hizTexH, uint32_t numLods);
};
} // namespace Ifrit::Core