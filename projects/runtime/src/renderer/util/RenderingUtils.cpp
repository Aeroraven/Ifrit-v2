
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
#include "ifrit/rhi/common/RhiStructHelper.h"

namespace Ifrit::Runtime::RenderingUtil
{

    IFRIT_APIDECL RHI::RhiComputePass* CreateComputePassInternal(
        IApplication* app, const ShaderVariantDesc& desc, u32 numBindlessDescs, u32 numPushConsts)
    {
        auto rhi       = app->GetRhi();
        auto shaderlib = app->GetShaderRegistry();

        auto shader = shaderlib->GetShader(desc);
        auto pass   = rhi->CreateComputePass();
        pass->SetComputeShader(shader);
        pass->SetNumBindlessDescriptorSets(numBindlessDescs);
        pass->SetPushConstSize(numPushConsts * sizeof(u32));
        return pass;
    }

    IFRIT_APIDECL RHI::RhiGraphicsPass* CreateGraphicsPassInternal(IApplication* app, const ShaderVariantDesc& vsDesc,
        const ShaderVariantDesc& fsDesc, u32 numBindlessDescs, u32 numPushConsts,
        const RHI::RhiRenderTargetsFormat& vFmts)
    {
        auto registry = app->GetShaderRegistry();
        auto vs       = registry->GetShader(vsDesc);
        auto fs       = registry->GetShader(fsDesc);
        auto rhi      = app->GetRhi();

        auto pass = rhi->CreateGraphicsPass();
        pass->SetVertexShader(vs);
        pass->SetPixelShader(fs);
        pass->SetNumBindlessDescriptorSets(numBindlessDescs);
        pass->SetPushConstSize(numPushConsts * sizeof(u32));
        pass->SetRenderTargetFormat(vFmts);
        return pass;
    }

    IFRIT_APIDECL void EnqueueFullScreenPass(const RHI::RhiCommandList* cmd, RHI::RhiBackend* rhi,
        RHI::RhiGraphicsPass* pass, RHI::RhiRenderTargets* rt,
        const Vec<RHI::RhiBindlessDescriptorRef*>& vBindlessDescs, const void* pPushConst, u32 numPushConsts)
    {

        pass->SetRecordFunction([&](const RHI::RhiRenderPassContext* ctx) {
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
    IFRIT_APIDECL void WarpRenderTargets(
        RHI::RhiBackend* rhi, RHI::RhiTexture* vTex, Ref<RHI::RhiColorAttachment>& vCA, Ref<RHI::RhiRenderTargets>& vRT)
    {
        vCA = rhi->CreateRenderTarget(
            vTex, RHI::CreateRhiClearColorValue(Vector4f(0.0f)), RHI::RhiRenderTargetLoadOp::Clear, 0, 0);
        vRT = rhi->CreateRenderTargets();
        vRT->SetColorAttachments({ vCA.get() });
        vRT->SetRenderArea({ 0, 0, vTex->GetWidth(), vTex->GetHeight() });
    }
} // namespace Ifrit::Runtime::RenderingUtil