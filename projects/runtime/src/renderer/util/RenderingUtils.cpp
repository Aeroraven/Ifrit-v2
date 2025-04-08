
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
#include "ifrit/runtime/renderer/util/RenderingUtils.h"
#include "ifrit/runtime/material/ShaderRegistry.h"

namespace Ifrit::Runtime::RenderingUtil
{

    IFRIT_APIDECL Graphics::Rhi::RhiShader* LoadShaderFromFile(Graphics::Rhi::RhiBackend* rhi, const char* shaderPath,
        const char* entryPoint, Graphics::Rhi::RhiShaderStage stage)
    {
        String    shaderBasePath = IFRIT_RUNTIME_SHARED_SHADER_PATH;
        auto      path           = shaderBasePath + "/" + shaderPath;
        auto      shaderCode     = ReadTextFile(path);
        Vec<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
        return rhi->CreateShader(
            shaderPath, shaderCodeVec, entryPoint, stage, Graphics::Rhi::RhiShaderSourceType::GLSLCode);
    }

    IFRIT_APIDECL Graphics::Rhi::RhiComputePass* CreateComputePassInternal(
        IApplication* app, const char* shaderName, u32 numBindlessDescs, u32 numPushConsts)
    {
        auto rhi       = app->GetRhi();
        auto shaderlib = app->GetShaderRegistry();

        auto shader = shaderlib->GetShader(shaderName, 0);
        auto pass   = rhi->CreateComputePass();
        pass->SetComputeShader(shader);
        pass->SetNumBindlessDescriptorSets(numBindlessDescs);
        pass->SetPushConstSize(numPushConsts * sizeof(u32));
        return pass;
    }

    IFRIT_APIDECL Graphics::Rhi::RhiGraphicsPass* CreateGraphicsPassInternal(IApplication* app, const char* nameVS,
        const char* nameFS, u32 numBindlessDescs, u32 numPushConsts, const Graphics::Rhi::RhiRenderTargetsFormat& vFmts)
    {
        auto registry = app->GetShaderRegistry();
        auto vs       = registry->GetShader(nameVS, 0);
        auto fs       = registry->GetShader(nameFS, 0);
        auto rhi      = app->GetRhi();

        auto pass = rhi->CreateGraphicsPass();
        pass->SetVertexShader(vs);
        pass->SetPixelShader(fs);
        pass->SetNumBindlessDescriptorSets(numBindlessDescs);
        pass->SetPushConstSize(numPushConsts * sizeof(u32));
        pass->SetRenderTargetFormat(vFmts);
        return pass;
    }

    IFRIT_APIDECL void EnqueueFullScreenPass(const Graphics::Rhi::RhiCommandList* cmd, Graphics::Rhi::RhiBackend* rhi,
        Graphics::Rhi::RhiGraphicsPass* pass, Graphics::Rhi::RhiRenderTargets* rt,
        const Vec<Graphics::Rhi::RhiBindlessDescriptorRef*>& vBindlessDescs, const void* pPushConst, u32 numPushConsts)
    {

        pass->SetRecordFunction([&](const Graphics::Rhi::RhiRenderPassContext* ctx) {
            for (auto i = 1; auto& desc : vBindlessDescs)
            {
                ctx->m_cmd->AttachUniformRef(i++, desc);
            }
            if (numPushConsts > 0)
                ctx->m_cmd->SetPushConst(pPushConst, 0, numPushConsts * sizeof(u32));

            // TODO: this should be done in vertex shader. Buffer is not needed
            ctx->m_cmd->AttachVertexBufferView(*rhi->GetFullScreenQuadVertexBufferView());
            ctx->m_cmd->AttachVertexBuffers(0, { rhi->GetFullScreenQuadVertexBuffer().get() });
            ctx->m_cmd->DrawInstanced(3, 1, 0, 0);
        });

        pass->Run(cmd, rt, 0);
    }
    IFRIT_APIDECL void warpRenderTargets(Graphics::Rhi::RhiBackend* rhi, Graphics::Rhi::RhiTexture* vTex,
        Ref<Graphics::Rhi::RhiColorAttachment>& vCA, Ref<Graphics::Rhi::RhiRenderTargets>& vRT)
    {
        vCA = rhi->CreateRenderTarget(
            vTex, { 0.0f, 0.0f, 0.0f, 0.0f }, Graphics::Rhi::RhiRenderTargetLoadOp::Clear, 0, 0);
        vRT = rhi->CreateRenderTargets();
        vRT->SetColorAttachments({ vCA.get() });
        vRT->SetRenderArea({ 0, 0, vTex->GetWidth(), vTex->GetHeight() });
    }
} // namespace Ifrit::Runtime::RenderingUtil