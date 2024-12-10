
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
#include "ifrit/core/renderer/util/RenderingUtils.h"

namespace Ifrit::Core::RenderingUtil {

IFRIT_APIDECL GraphicsBackend::Rhi::RhiShader *
loadShaderFromFile(GraphicsBackend::Rhi::RhiBackend *rhi,
                   const char *shaderPath, const char *entryPoint,
                   GraphicsBackend::Rhi::RhiShaderStage stage) {
  std::string shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  auto path = shaderBasePath + "/" + shaderPath;
  auto shaderCode = Ifrit::Common::Utility::readTextFile(path);
  std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
  return rhi->createShader(shaderPath, shaderCodeVec, entryPoint, stage,
                           GraphicsBackend::Rhi::RhiShaderSourceType::GLSLCode);
}

IFRIT_APIDECL GraphicsBackend::Rhi::RhiComputePass *
createComputePass(GraphicsBackend::Rhi::RhiBackend *rhi, const char *shaderPath,
                  uint32_t numBindlessDescs, uint32_t numPushConsts) {
  auto shader = loadShaderFromFile(
      rhi, shaderPath, "main", GraphicsBackend::Rhi::RhiShaderStage::Compute);
  auto pass = rhi->createComputePass();
  pass->setComputeShader(shader);
  pass->setNumBindlessDescriptorSets(numBindlessDescs);
  pass->setPushConstSize(numPushConsts * sizeof(uint32_t));
  return pass;
}

IFRIT_APIDECL void enqueueFullScreenPass(
    const GraphicsBackend::Rhi::RhiCommandBuffer *cmd,
    GraphicsBackend::Rhi::RhiBackend *rhi,
    GraphicsBackend::Rhi::RhiGraphicsPass *pass,
    GraphicsBackend::Rhi::RhiRenderTargets *rt,
    const std::vector<GraphicsBackend::Rhi::RhiBindlessDescriptorRef *>
        &vBindlessDescs,
    const void *pPushConst, uint32_t numPushConsts) {

  pass->setRecordFunction(
      [&](const GraphicsBackend::Rhi::RhiRenderPassContext *ctx) {
        for (auto i = 1; auto &desc : vBindlessDescs) {
          ctx->m_cmd->attachBindlessReferenceGraphics(pass, i++, desc);
        }
        ctx->m_cmd->setPushConst(pass, 0, numPushConsts * sizeof(uint32_t),
                                 pPushConst);

        // TODO: this should be done in vertex shader. Buffer is not needed
        ctx->m_cmd->attachVertexBufferView(
            *rhi->getFullScreenQuadVertexBufferView());
        ctx->m_cmd->attachVertexBuffers(
            0, {rhi->getFullScreenQuadVertexBuffer().get()});
        ctx->m_cmd->drawInstanced(3, 1, 0, 0);
      });

  pass->run(cmd, rt, 0);
}
} // namespace Ifrit::Core::RenderingUtil