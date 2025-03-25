
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

IFRIT_APIDECL GraphicsBackend::Rhi::RhiShader *loadShaderFromFile(GraphicsBackend::Rhi::RhiBackend *rhi,
                                                                  const char *shaderPath, const char *entryPoint,
                                                                  GraphicsBackend::Rhi::RhiShaderStage stage) {
  String shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  auto path = shaderBasePath + "/" + shaderPath;
  auto shaderCode = Ifrit::Common::Utility::readTextFile(path);
  Vec<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
  return rhi->createShader(shaderPath, shaderCodeVec, entryPoint, stage,
                           GraphicsBackend::Rhi::RhiShaderSourceType::GLSLCode);
}

IFRIT_APIDECL GraphicsBackend::Rhi::RhiComputePass *createComputePass(GraphicsBackend::Rhi::RhiBackend *rhi,
                                                                      const char *shaderPath, u32 numBindlessDescs,
                                                                      u32 numPushConsts) {
  auto shader = loadShaderFromFile(rhi, shaderPath, "main", GraphicsBackend::Rhi::RhiShaderStage::Compute);
  auto pass = rhi->createComputePass();
  pass->setComputeShader(shader);
  pass->setNumBindlessDescriptorSets(numBindlessDescs);
  pass->setPushConstSize(numPushConsts * sizeof(u32));
  return pass;
}

IFRIT_APIDECL GraphicsBackend::Rhi::RhiGraphicsPass *
createGraphicsPass(GraphicsBackend::Rhi::RhiBackend *rhi, const char *vsPath, const char *fsPath, u32 numBindlessDescs,
                   u32 numPushConsts, const GraphicsBackend::Rhi::RhiRenderTargetsFormat &vFmts) {
  auto vs = loadShaderFromFile(rhi, vsPath, "main", GraphicsBackend::Rhi::RhiShaderStage::Vertex);
  auto fs = loadShaderFromFile(rhi, fsPath, "main", GraphicsBackend::Rhi::RhiShaderStage::Fragment);
  auto pass = rhi->createGraphicsPass();
  pass->setVertexShader(vs);
  pass->setPixelShader(fs);
  pass->setNumBindlessDescriptorSets(numBindlessDescs);
  pass->setPushConstSize(numPushConsts * sizeof(u32));
  pass->setRenderTargetFormat(vFmts);
  return pass;
}

IFRIT_APIDECL void enqueueFullScreenPass(const GraphicsBackend::Rhi::RhiCommandList *cmd,
                                         GraphicsBackend::Rhi::RhiBackend *rhi,
                                         GraphicsBackend::Rhi::RhiGraphicsPass *pass,
                                         GraphicsBackend::Rhi::RhiRenderTargets *rt,
                                         const Vec<GraphicsBackend::Rhi::RhiBindlessDescriptorRef *> &vBindlessDescs,
                                         const void *pPushConst, u32 numPushConsts) {

  pass->setRecordFunction([&](const GraphicsBackend::Rhi::RhiRenderPassContext *ctx) {
    for (auto i = 1; auto &desc : vBindlessDescs) {
      ctx->m_cmd->attachBindlessReferenceGraphics(pass, i++, desc);
    }
    if (numPushConsts > 0)
      ctx->m_cmd->setPushConst(pass, 0, numPushConsts * sizeof(u32), pPushConst);

    // TODO: this should be done in vertex shader. Buffer is not needed
    ctx->m_cmd->attachVertexBufferView(*rhi->getFullScreenQuadVertexBufferView());
    ctx->m_cmd->attachVertexBuffers(0, {rhi->getFullScreenQuadVertexBuffer().get()});
    ctx->m_cmd->drawInstanced(3, 1, 0, 0);
  });

  pass->run(cmd, rt, 0);
}
IFRIT_APIDECL void warpRenderTargets(GraphicsBackend::Rhi::RhiBackend *rhi, GraphicsBackend::Rhi::RhiTexture *vTex,
                                     Ref<GraphicsBackend::Rhi::RhiColorAttachment> &vCA,
                                     Ref<GraphicsBackend::Rhi::RhiRenderTargets> &vRT) {
  vCA =
      rhi->createRenderTarget(vTex, {0.0f, 0.0f, 0.0f, 0.0f}, GraphicsBackend::Rhi::RhiRenderTargetLoadOp::Clear, 0, 0);
  vRT = rhi->createRenderTargets();
  vRT->setColorAttachments({vCA.get()});
  vRT->setRenderArea({0, 0, vTex->getWidth(), vTex->getHeight()});
}
} // namespace Ifrit::Core::RenderingUtil