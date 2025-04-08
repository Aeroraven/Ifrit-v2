
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
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

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
        const String& vs, const String& fs, u32 pushConsts, FnPassFunction onCall)
    {
        auto& pass           = builder.AddGraphicsPass(name, vs, fs, pushConsts);
        auto  rhi            = builder.GetRhi();
        auto  underlyingPass = pass.GetPass();
        // DO NOT USE REFERENCES HERE.
        pass.SetExecutionFunction([onCall, rhi](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            onCall(ctx);
            cmd->AttachVertexBufferView(*rhi->GetFullScreenQuadVertexBufferView());
            cmd->AttachVertexBuffers(0, { rhi->GetFullScreenQuadVertexBuffer().get() });
            cmd->DrawInstanced(3, 1, 0, 0);
        });
        return pass;
    }

    IFRIT_RUNTIME_API GraphicsPassNode& AddPostProcessPass(
        FrameGraphBuilder& builder, const String& name, const String& fs, u32 pushConsts, FnPassFunction onCall)
    {
        auto& pass = builder.AddGraphicsPass(name, Internal::kIntShaderTable.Common.FullScreenVS, fs, pushConsts);
        auto  rhi  = builder.GetRhi();
        auto  underlyingPass = pass.GetPass();
        pass.SetExecutionFunction([rhi, onCall](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            onCall(ctx);
            cmd->AttachVertexBufferView(*rhi->GetFullScreenQuadVertexBufferView());
            cmd->AttachVertexBuffers(0, { rhi->GetFullScreenQuadVertexBuffer().get() });
            cmd->DrawInstanced(3, 1, 0, 0);
        });
        return pass;
    }

    IFRIT_APIDECL GraphicsPassNode& AddMeshDrawPass(FrameGraphBuilder& builder, const String& name, const String& ms,
        const String& fs, Vector3i workGroups, u32 pushConsts, const GraphicsPassArgs& args, FnPassFunction onCall)
    {
        auto& pass           = builder.AddMeshGraphicsPass(name, ms, fs, pushConsts);
        auto  rhi            = builder.GetRhi();
        auto  underlyingPass = pass.GetPass();
        pass.SetExecutionFunction([onCall, workGroups, args](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            onCall(ctx);
            if (args.m_CullMode != Graphics::Rhi::RhiCullMode::None)
                cmd->SetCullMode(args.m_CullMode);
            cmd->DrawMeshTasks(workGroups.x, workGroups.y, workGroups.z);
        });
        return pass;
    }

    IFRIT_APIDECL ComputePassNode& AddComputePass(FrameGraphBuilder& builder, const String& name, const String& shader,
        Vector3i workGroups, u32 pushConsts, FnPassFunction onCall)
    {
        auto& pass = builder.AddComputePass(name, shader, pushConsts);
        auto  rhi  = builder.GetRhi();
        auto  cp   = pass.GetPass();
        pass.SetExecutionFunction([workGroups, pushConsts, onCall, cp](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            onCall(ctx);
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