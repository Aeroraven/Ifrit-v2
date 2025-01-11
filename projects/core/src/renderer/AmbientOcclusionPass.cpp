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
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "ifrit/core/renderer/AmbientOcclusionPass.h"
#include "ifrit.shader/AmbientOcclusion/AmbientOcclusion.Shared.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/core/renderer/RendererUtil.h"

using namespace Ifrit::GraphicsBackend::Rhi;

namespace Ifrit::Core {
IFRIT_APIDECL AmbientOcclusionPass::GPUShader *
AmbientOcclusionPass::createShaderFromFile(
    const std::string &shaderPath, const std::string &entry,
    GraphicsBackend::Rhi::RhiShaderStage stage) {
  auto rhi = m_app->getRhiLayer();
  std::string shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  auto path = shaderBasePath + "/AmbientOcclusion/" + shaderPath;
  auto shaderCode = Ifrit::Common::Utility::readTextFile(path);
  std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
  return rhi->createShader(shaderPath, shaderCodeVec, entry, stage,
                           RhiShaderSourceType::GLSLCode);
}

IFRIT_APIDECL void AmbientOcclusionPass::setupHBAOPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile(
      "HBAO.comp.glsl", "main", GraphicsBackend::Rhi::RhiShaderStage::Compute);
  m_hbaoPass = rhi->createComputePass();
  m_hbaoPass->setComputeShader(shader);
  m_hbaoPass->setNumBindlessDescriptorSets(0);
  m_hbaoPass->setPushConstSize(sizeof(uint32_t) * 6);
}

IFRIT_APIDECL void AmbientOcclusionPass::setupSSGIPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile(
      "SSGI.comp.glsl", "main", GraphicsBackend::Rhi::RhiShaderStage::Compute);
  m_ssgiPass = rhi->createComputePass();
  m_ssgiPass->setComputeShader(shader);
  m_ssgiPass->setNumBindlessDescriptorSets(0);
  m_ssgiPass->setPushConstSize(sizeof(uint32_t) * 10);
}

IFRIT_APIDECL void
AmbientOcclusionPass::renderHBAO(const CommandBuffer *cmd, uint32_t width,
                                 uint32_t height, GPUBindId *depthSamp,
                                 GPUBindId *normalSamp, GPUBindId *aoTex,
                                 GPUBindId *perframeData) {
  if (m_hbaoPass == nullptr) {
    setupHBAOPass();
  }
  struct HBAOPushConst {
    uint32_t perframe;
    uint32_t normalTex;
    uint32_t depthTex;
    uint32_t aoTex;
    float radius;
    float maxRadius;
  } pc;
  pc.perframe = perframeData->getActiveId();
  pc.normalTex = normalSamp->getActiveId();
  pc.depthTex = depthSamp->getActiveId();
  pc.aoTex = aoTex->getActiveId();
  pc.radius = 0.5f;
  pc.maxRadius = 1.0f;

  m_hbaoPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    using namespace Ifrit::Core::Shaders::AmbientOcclusionConfig;
    ctx->m_cmd->setPushConst(m_hbaoPass, 0, sizeof(HBAOPushConst), &pc);
    uint32_t wgX =
        Ifrit::Math::ConstFunc::divRoundUp(width, cHBAOThreadGroupSizeX);
    uint32_t wgY =
        Ifrit::Math::ConstFunc::divRoundUp(height, cHBAOThreadGroupSizeY);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
  });

  m_hbaoPass->run(cmd, 0);
}

IFRIT_APIDECL void AmbientOcclusionPass::renderSSGI(
    const CommandBuffer *cmd, uint32_t width, uint32_t height,
    GPUBindId *perframeData, GPUBindId *depthHizUAV, GPUBindId *normalSRV,
    GPUBindId *aoUAV, GPUBindId *albedoSRV, uint32_t hizTexW, uint32_t hizTexH,
    uint32_t numLods) {
  struct SSGIPushConst {
    uint32_t perframe;
    uint32_t normalTex;
    uint32_t depthTex;
    uint32_t aoTex;
    uint32_t albedoTex;
    uint32_t hizTexW;
    uint32_t hizTexH;
    uint32_t rtW;
    uint32_t rtH;
    uint32_t numLods;
  } pc;

  pc.perframe = perframeData->getActiveId();
  pc.normalTex = normalSRV->getActiveId();
  pc.depthTex = depthHizUAV->getActiveId();
  pc.aoTex = aoUAV->getActiveId();
  pc.albedoTex = albedoSRV->getActiveId();
  pc.hizTexW = hizTexW;
  pc.hizTexH = hizTexH;
  pc.rtW = width;
  pc.rtH = height;
  pc.numLods = numLods;

  if (m_ssgiPass == nullptr) {
    setupSSGIPass();
  }

  m_ssgiPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    using namespace Ifrit::Core::Shaders::AmbientOcclusionConfig;
    ctx->m_cmd->setPushConst(m_ssgiPass, 0, sizeof(SSGIPushConst), &pc);

    uint32_t wgX =
        Ifrit::Math::ConstFunc::divRoundUp(width, cSSGIThreadGroupSizeX);
    uint32_t wgY =
        Ifrit::Math::ConstFunc::divRoundUp(height, cSSGIThreadGroupSizeY);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
  });
}

} // namespace Ifrit::Core
