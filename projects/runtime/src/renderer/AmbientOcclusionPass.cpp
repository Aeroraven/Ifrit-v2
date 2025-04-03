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
#include "ifrit/runtime/renderer/AmbientOcclusionPass.h"
#include "ifrit.shader/AmbientOcclusion/AmbientOcclusion.Shared.h"
#include "ifrit/core/math/constfunc/ConstFunc.h"
#include "ifrit/core/file/FileOps.h"
#include "ifrit/runtime/renderer/RendererUtil.h"

#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

using namespace Ifrit::Graphics::Rhi;

namespace Ifrit::Runtime
{
    IFRIT_APIDECL AmbientOcclusionPass::GPUShader* AmbientOcclusionPass::GetInternalShader(const char* name)
    {
        auto registry = m_app->GetShaderRegistry();
        return registry->GetShader(name, 0);
    }

    IFRIT_APIDECL void AmbientOcclusionPass::SetupHBAOPass()
    {
        auto rhi    = m_app->GetRhi();
        auto shader = GetInternalShader(Internal::kIntShaderTable.GI.HBAOCS);
        m_hbaoPass  = rhi->CreateComputePass();
        m_hbaoPass->SetComputeShader(shader);
        m_hbaoPass->SetNumBindlessDescriptorSets(0);
        m_hbaoPass->SetPushConstSize(sizeof(u32) * 6);
    }

    IFRIT_APIDECL void AmbientOcclusionPass::SetupSSGIPass()
    {
        auto rhi    = m_app->GetRhi();
        auto shader = GetInternalShader(Internal::kIntShaderTable.GI.SSGICS);
        m_ssgiPass  = rhi->CreateComputePass();
        m_ssgiPass->SetComputeShader(shader);
        m_ssgiPass->SetNumBindlessDescriptorSets(0);
        m_ssgiPass->SetPushConstSize(sizeof(u32) * 12);
    }

    IFRIT_APIDECL void AmbientOcclusionPass::RenderHBAO(const CommandBuffer* cmd, u32 width, u32 height,
        GPUBindId* depthSamp, GPUBindId* normalSamp, u32 aoTex, GPUBindId* perframeData)
    {
        if (m_hbaoPass == nullptr)
        {
            SetupHBAOPass();
        }
        struct HBAOPushConst
        {
            u32   perframe;
            u32   normalTex;
            u32   depthTex;
            u32   aoTex;
            float radius;
            float maxRadius;
        } pc;
        pc.perframe  = perframeData->GetActiveId();
        pc.normalTex = normalSamp->GetActiveId();
        pc.depthTex  = depthSamp->GetActiveId();
        pc.aoTex     = aoTex;
        pc.radius    = 0.5f;
        pc.maxRadius = 1.0f;

        m_hbaoPass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            using namespace Ifrit::Runtime::Shaders::AmbientOcclusionConfig;
            ctx->m_cmd->SetPushConst(m_hbaoPass, 0, sizeof(HBAOPushConst), &pc);
            u32 wgX = Ifrit::Math::DivRoundUp(width, cHBAOThreadGroupSizeX);
            u32 wgY = Ifrit::Math::DivRoundUp(height, cHBAOThreadGroupSizeY);
            ctx->m_cmd->Dispatch(wgX, wgY, 1);
        });

        m_hbaoPass->Run(cmd, 0);
    }

    IFRIT_APIDECL void AmbientOcclusionPass::RenderSSGI(const CommandBuffer* cmd, u32 width, u32 height,
        GPUBindId* perframeData, u32 depthHizMinUAV, u32 depthHizMaxUAV, GPUBindId* normalSRV, u32 aoUAV,
        GPUBindId* albedoSRV, u32 hizTexW, u32 hizTexH, u32 numLods, GPUBindId* blueNoiseSRV)
    {
        struct SSGIPushConst
        {
            u32 perframe;
            u32 normalTex;
            u32 depthTexMin;
            u32 depthTexMax;
            u32 aoTex;
            u32 albedoTex;
            u32 hizTexW;
            u32 hizTexH;
            u32 rtW;
            u32 rtH;
            u32 numLods;
            u32 blueNoiseSRV;
        } pc;

        pc.perframe     = perframeData->GetActiveId();
        pc.normalTex    = normalSRV->GetActiveId();
        pc.depthTexMin  = depthHizMinUAV;
        pc.depthTexMax  = depthHizMaxUAV;
        pc.aoTex        = aoUAV;
        pc.albedoTex    = albedoSRV->GetActiveId();
        pc.hizTexW      = hizTexW;
        pc.hizTexH      = hizTexH;
        pc.rtW          = width;
        pc.rtH          = height;
        pc.numLods      = numLods;
        pc.blueNoiseSRV = blueNoiseSRV->GetActiveId();

        if (m_ssgiPass == nullptr)
        {
            SetupSSGIPass();
        }

        m_ssgiPass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            using namespace Ifrit::Runtime::Shaders::AmbientOcclusionConfig;
            ctx->m_cmd->SetPushConst(m_ssgiPass, 0, sizeof(SSGIPushConst), &pc);

            u32 wgX = Ifrit::Math::DivRoundUp(width, cSSGIThreadGroupSizeX);
            u32 wgY = Ifrit::Math::DivRoundUp(height, cSSGIThreadGroupSizeY);
            ctx->m_cmd->Dispatch(wgX, wgY, 1);
        });
        m_ssgiPass->Run(cmd, 0);
    }

} // namespace Ifrit::Runtime
