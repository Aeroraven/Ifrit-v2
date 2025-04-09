
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime::FrameGraphUtils
{
    struct GraphicsPassArgs
    {
        Graphics::Rhi::RhiCullMode m_CullMode = Graphics::Rhi::RhiCullMode::Back;
    };

    template <typename T> u32 GetPushConstSize() { return sizeof(T) / sizeof(u32); }

    using FnPassFunction = Fn<void(const FrameGraphPassContext&)>;

    IFRIT_RUNTIME_API GraphicsPassNode& AddFullScreenQuadPass(FrameGraphBuilder& builder, const String& name,
        const String& vs, const String& fs, u32 pushConsts, FnPassFunction onCall);

    IFRIT_RUNTIME_API GraphicsPassNode& AddPostProcessPass(
        FrameGraphBuilder& builder, const String& name, const String& fs, u32 pushConsts, FnPassFunction onCall);

    IFRIT_RUNTIME_API GraphicsPassNode& AddMeshDrawPass(FrameGraphBuilder& builder, const String& name,
        const String& ms, const String& fs, Vector3i workGroups, u32 pushConsts, const GraphicsPassArgs& args,
        FnPassFunction onCall);

    IFRIT_RUNTIME_API ComputePassNode&  AddComputePass(FrameGraphBuilder& builder, const String& name,
         const String& shader, Vector3i workGroups, u32 pushConsts, FnPassFunction onCall);

    IFRIT_RUNTIME_API PassNode&         AddClearUAVPass(
                FrameGraphBuilder& builder, const String& name, ResourceNode& buffer, u32 clearValue);

    IFRIT_RUNTIME_API PassNode& AddClearUAVTexturePass(
        FrameGraphBuilder& builder, const String& name, ResourceNode& texture, u64 clearValue);
    // Templated Version

    template <typename PassData> using FnPassFunctionWithData = Fn<void(PassData, const FrameGraphPassContext&)>;

    template <typename PassData, typename RootSignature = PassData>
    GraphicsPassNode& AddFullScreenQuadPass(FrameGraphBuilder& builder, const String& name, const String& vs,
        const String& fs, PassData passData, FnPassFunctionWithData<PassData> onCall)
    {
        auto& node = AddFullScreenQuadPass(builder, name, vs, fs, GetPushConstSize<RootSignature>(),
            [passData, onCall](const FrameGraphPassContext& ctx) { onCall(passData, ctx); });
        return node;
    }

    template <typename PassData, typename RootSignature = PassData>
    GraphicsPassNode& AddPostProcessPass(FrameGraphBuilder& builder, const String& name, const String& fs,
        PassData passData, FnPassFunctionWithData<PassData> onCall)
    {
        auto& node = AddPostProcessPass(builder, name, fs, GetPushConstSize<RootSignature>(),
            [passData, onCall](const FrameGraphPassContext& ctx) { onCall(passData, ctx); });
        return node;
    }

    template <typename PassData, typename RootSignature = PassData>
    GraphicsPassNode& AddMeshDrawPass(FrameGraphBuilder& builder, const String& name, const String& ms,
        const String& fs, Vector3i workGroups, const GraphicsPassArgs& args, PassData passData,
        FnPassFunctionWithData<PassData> onCall)
    {
        auto& node = AddMeshDrawPass(builder, name, ms, fs, workGroups, GetPushConstSize<RootSignature>(), args,
            [onCall, passData](const FrameGraphPassContext& ctx) { onCall(passData, ctx); });
        return node;
    }

    template <typename PassData, typename RootSignature = PassData>
    ComputePassNode& AddComputePass(FrameGraphBuilder& builder, const String& name, const String& shader,
        Vector3i workGroups, PassData passData, FnPassFunctionWithData<PassData> onCall)
    {
        auto& node = AddComputePass(builder, name, shader, workGroups, GetPushConstSize<RootSignature>(),
            [onCall, passData](const FrameGraphPassContext& ctx) { onCall(passData, ctx); });
        return node;
    }

    // Other Utilities
    template <typename T> void SetRootSignature(const T& data, const FrameGraphPassContext& ctx)
    {
        ctx.m_CmdList->SetPushConst(&data, 0, sizeof(T));
    }

} // namespace Ifrit::Runtime::FrameGraphUtils