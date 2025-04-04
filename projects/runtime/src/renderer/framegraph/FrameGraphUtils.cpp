
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

#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"

namespace Ifrit::Runtime::FrameGraphUtils
{
    using namespace Ifrit::Graphics::Rhi;

    Vec<u8> PtrToVector(const void* ptr, u32 size)
    {
        Vec<u8> vec(size);
        memcpy(vec.data(), ptr, size);
        return vec;
    }

    IFRIT_APIDECL GraphicsPassNode& AddFullScreenQuadPass(FrameGraphBuilder& builder, const String& name,
        const String& vs, const String& fs, Graphics::Rhi::RhiRenderTargets* rts, const void* ptr, u32 pushConsts)
    {
        auto&   pass           = builder.AddGraphicsPass(name, vs, fs, pushConsts, rts);
        auto    rhi            = builder.GetRhi();
        auto    underlyingPass = pass.GetPass();
        Vec<u8> pushConstsData = PtrToVector(ptr, pushConsts * sizeof(u32));
        // DO NOT USE REFERENCES HERE.
        pass.SetExecutionFunction([pushConsts, pushConstsData, rhi, underlyingPass](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            if (pushConsts > 0)
                cmd->SetPushConst(underlyingPass, 0, pushConsts * sizeof(u32), pushConstsData.data());
            cmd->AttachVertexBufferView(*rhi->GetFullScreenQuadVertexBufferView());
            cmd->AttachVertexBuffers(0, { rhi->GetFullScreenQuadVertexBuffer().get() });
            cmd->DrawInstanced(3, 1, 0, 0);
        });
        return pass;
    }

    IFRIT_APIDECL GraphicsPassNode& AddMeshDrawPass(FrameGraphBuilder& builder, const String& name, const String& ms,
        const String& fs, Graphics::Rhi::RhiRenderTargets* rts, Vector3i workGroups, const void* ptr, u32 pushConsts)
    {
        auto&   pass           = builder.AddMeshGraphicsPass(name, ms, fs, pushConsts, rts);
        auto    rhi            = builder.GetRhi();
        auto    underlyingPass = pass.GetPass();
        Vec<u8> pushConstsData = PtrToVector(ptr, pushConsts * sizeof(u32));
        pass.SetExecutionFunction(
            [pushConsts, pushConstsData, rhi, underlyingPass, workGroups](const FrameGraphPassContext& ctx) {
                auto cmd = ctx.m_CmdList;
                if (pushConsts > 0)
                    cmd->SetPushConst(underlyingPass, 0, pushConsts * sizeof(u32), pushConstsData.data());
                cmd->DrawMeshTasks(workGroups.x, workGroups.y, workGroups.z);
            });
        return pass;
    }

    IFRIT_APIDECL ComputePassNode& AddComputePass(FrameGraphBuilder& builder, const String& name, const String& shader,
        Vector3i workGroups, const void* ptr, u32 pushConsts)
    {
        auto&   pass           = builder.AddComputePass(name, shader, pushConsts);
        auto    rhi            = builder.GetRhi();
        auto    cp             = pass.GetPass();
        Vec<u8> pushConstsData = PtrToVector(ptr, pushConsts * sizeof(u32));
        pass.SetExecutionFunction([workGroups, pushConsts, pushConstsData, rhi, cp](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            if (pushConsts > 0)
                cmd->SetPushConst(cp, 0, pushConsts * sizeof(u32), pushConstsData.data());
            cmd->Dispatch(workGroups.x, workGroups.y, workGroups.z);
        });
        return pass;
    }

    IFRIT_APIDECL PassNode& AddClearUAVPass(
        FrameGraphBuilder& builder, const String& name, ResourceNode& buffer, u32 clearValue)
    {
        auto& pass = builder.AddPass(name, FrameGraphPassType::Transfer).AddWriteResource(buffer);
        if (buffer.GetType() != FrameGraphResourceType::ResourceBuffer)
        {
            iError("FrameGraphUtils: Clear UAV pass only supports buffer resources.");
            std::abort();
        }
        auto buf = buffer.GetBuffer();
        pass.SetExecutionFunction([buf, clearValue](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            cmd->BufferClear(buf, clearValue);
        });
        return pass;
    }

} // namespace Ifrit::Runtime::FrameGraphUtils