
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
#include "ifrit/rhi/common/RhiStructHelper.h"

namespace Ifrit::Runtime::FrameGraphUtils
{
    using namespace Ifrit::RHI;

    Vec<u8> PtrToVector(const void* ptr, u32 size)
    {
        Vec<u8> vec(size);
        memcpy(vec.data(), ptr, size);
        return vec;
    }

    IFRIT_RUNTIME_API FrameGraphScopeGuard::FrameGraphScopeGuard(FrameGraphBuilder& builder, const String& name)
        : m_Builder(&builder)
    {
        m_Scope = &m_Builder->AddScopeBegin(name);
    }

    IFRIT_RUNTIME_API FrameGraphScopeGuard::~FrameGraphScopeGuard()
    {
        if (m_Builder && m_Scope)
        {
            m_Builder->AddScopeEnd(*m_Scope);
            m_Scope = nullptr;
        }
    }

    IFRIT_RUNTIME_API FrameGraphStatScopeGuard::FrameGraphStatScopeGuard(FrameGraphBuilder& builder, const String& name)
        : m_Builder(&builder)
    {
        m_Scope = &m_Builder->AddStatScopeBegin(name);
    }

    IFRIT_RUNTIME_API FrameGraphStatScopeGuard::~FrameGraphStatScopeGuard()
    {
        if (m_Builder && m_Scope)
        {
            m_Builder->AddStatScopeEnd(*m_Scope);
            m_Scope = nullptr;
        }
    }

    IFRIT_RUNTIME_API Owner<FrameGraphScopeGuard> AddFrameGraphEventScope(
        FrameGraphBuilder& builder, const String& name)
    {
        auto scopeGuard = MakeOwner<FrameGraphScopeGuard>(builder, name);
        return scopeGuard;
    }

    IFRIT_RUNTIME_API Owner<FrameGraphStatScopeGuard> AddFrameGraphStatScope(
        FrameGraphBuilder& builder, const String& name)
    {
        auto scopeGuard = MakeOwner<FrameGraphStatScopeGuard>(builder, name);
        return scopeGuard;
    }

    IFRIT_APIDECL GraphicsPassNode& AddFullScreenQuadPass(FrameGraphBuilder& builder, const String& name,
        const ShaderVariantDesc& vs, const ShaderVariantDesc& fs, u32 pushConsts, FnPassFunction onCall)
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

    IFRIT_RUNTIME_API GraphicsPassNode& AddPostProcessPass(FrameGraphBuilder& builder, const String& name,
        const ShaderVariantDesc& fs, u32 pushConsts, FnPassFunction onCall)
    {
        auto& pass = builder.AddGraphicsPass(
            name, ShaderVariantDesc(Internal::kIntShaderTable.Common.FullScreenVS, {}), fs, pushConsts);
        auto rhi            = builder.GetRhi();
        auto underlyingPass = pass.GetPass();
        pass.SetExecutionFunction([rhi, onCall](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            onCall(ctx);
            cmd->AttachVertexBufferView(*rhi->GetFullScreenQuadVertexBufferView());
            cmd->AttachVertexBuffers(0, { rhi->GetFullScreenQuadVertexBuffer().get() });
            cmd->DrawInstanced(3, 1, 0, 0);
        });
        return pass;
    }

    IFRIT_APIDECL GraphicsPassNode& AddMeshDrawPass(FrameGraphBuilder& builder, const String& name,
        const ShaderVariantDesc& ms, const ShaderVariantDesc& fs, Vector3i workGroups, u32 pushConsts,
        const GraphicsPassArgs& args, FnPassFunction onCall)
    {
        auto& pass           = builder.AddMeshGraphicsPass(name, ms, fs, pushConsts);
        auto  rhi            = builder.GetRhi();
        auto  underlyingPass = pass.GetPass();
        pass.SetExecutionFunction([onCall, workGroups, args](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            onCall(ctx);
            if (args.m_CullMode != RHI::RhiCullMode::None)
                cmd->SetCullMode(args.m_CullMode);
            cmd->DrawMeshTasks(workGroups.x, workGroups.y, workGroups.z);
        });
        return pass;
    }

    IFRIT_APIDECL GraphicsPassNode& AddIndirectDrawPass(FrameGraphBuilder& builder, const String& name,
        const ShaderVariantDesc& vs, const ShaderVariantDesc& fs, ResourceNode& indirectArgs, ResourceNode& indexBuffer,
        u32 offset, u32 pushConsts, const GraphicsPassArgs& args, FnPassFunction onCall)
    {
        auto& pass           = builder.AddGraphicsPass(name, vs, fs, pushConsts);
        auto  rhi            = builder.GetRhi();
        auto  underlyingPass = pass.GetPass();
        pass.AddReadResource(indirectArgs);
        pass.AddReadResource(indexBuffer);
        pass.SetExecutionFunction([onCall, &indirectArgs, indexBuffer, offset, args](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;
            onCall(ctx);
            if (args.m_CullMode != RHI::RhiCullMode::None)
                cmd->SetCullMode(args.m_CullMode);
            cmd->AttachIndexBuffer(indexBuffer.GetBuffer());
            cmd->DrawIndexedIndirect(indirectArgs.GetBuffer(), 0);
        });
        return pass;
    }

    IFRIT_APIDECL ComputePassNode& AddComputePass(FrameGraphBuilder& builder, const String& name,
        const ShaderVariantDesc& shader, Vector3i workGroups, u32 pushConsts, FnPassFunction onCall)
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

    IFRIT_APIDECL ComputePassNode& AddIndirectComputePass(FrameGraphBuilder& builder, const String& name,
        const ShaderVariantDesc& shader, ResourceNode& workGroupsIndirect, u32 offset, u32 pushConsts,
        FnPassFunction onCall)
    {
        auto& pass = builder.AddComputePass(name, shader, pushConsts);
        auto  rhi  = builder.GetRhi();
        auto  cp   = pass.GetPass();
        pass.AddReadResource(workGroupsIndirect);
        pass.SetExecutionFunction(
            [&workGroupsIndirect, pushConsts, onCall, cp, offset](const FrameGraphPassContext& ctx) {
                auto cmd = ctx.m_CmdList;
                onCall(ctx);
                cmd->DispatchIndirect(workGroupsIndirect.GetBuffer(), offset);
            });
        return pass;
    }

    IFRIT_APIDECL PassNode& AddClearUAVPass(
        FrameGraphBuilder& builder, const String& name, ResourceNode& buffer, u32 clearValue)
    {
        auto& pass = builder.AddPass(name, FrameGraphPassType::Transfer).AddWriteResource(buffer);
        if (buffer.GetType() != FrameGraphResourceType::ResourceBuffer)
        {
            IF_LOG_CRITICAL("FrameGraph", "Clear UAV pass only supports buffer resources.");
        }
        else
        {
            pass.SetExecutionFunction([&buffer, clearValue](const FrameGraphPassContext& ctx) {
                auto buf = buffer.GetBuffer();
                auto cmd = ctx.m_CmdList;
                cmd->BufferClear(buf, clearValue);
                cmd->GlobalMemoryBarrier();
            });
        }

        return pass;
    }

    IFRIT_APIDECL PassNode& AddClearUAVTexturePass(
        FrameGraphBuilder& builder, const String& name, ResourceNode& texture, RHI::RhiClearColorValue clearValue)
    {
        auto& pass = builder.AddPass(name, FrameGraphPassType::Transfer).AddWriteResource(texture);
        if (texture.GetType() != FrameGraphResourceType::ResourceTexture)
        {
            IF_LOG_CRITICAL("FrameGraph", "Clear UAV pass only supports texture resources.");
            std::abort();
        }
        else
        {
            pass.SetExecutionFunction([&texture, clearValue](const FrameGraphPassContext& ctx) {
                auto cmd = ctx.m_CmdList;
                cmd->ClearUAVTexture(texture.GetTexture(), { 0, 0, 1, 1 }, clearValue);
            });
        }

        return pass;
    }

} // namespace Ifrit::Runtime::FrameGraphUtils