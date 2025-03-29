
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <functional>
#include <string>
#include <vector>

namespace Ifrit::Core
{

    // Migration of render graph from original RHI layer.
    // Intended to making automatic layout transitions and resource management
    // easier. Resource lifetime management and reuse will be considered in the
    // future.
    // Some references from: https://zhuanlan.zhihu.com/p/147207161

    using ResourceNodeId       = u32;
    using PassNodeId           = u32;
    using FgBuffer             = Graphics::Rhi::RhiBuffer;
    using FgTexture            = Graphics::Rhi::RhiTexture;
    using FgTextureSubResource = Graphics::Rhi::RhiImageSubResource;

    class FrameGraphCompiler;
    class FrameGraphExecutor;

    enum class FrameGraphResourceType
    {
        Undefined,
        ResourceBuffer,
        ResourceTexture,
    };

    enum class FrameGraphPassType
    {
        Compute,
        Graphics,
    };

    struct FrameGraphPassContext
    {
        const Graphics::Rhi::RhiCommandList* m_CmdList;
    };

    struct IFRIT_APIDECL ResourceNode
    {
    private:
        ResourceNodeId         id;
        String                 name;
        bool                   isImported;
        FrameGraphResourceType type;

        Ref<FgBuffer>          selfBuffer;
        Ref<FgTexture>         selfTexture;
        FgBuffer*              importedBuffer;
        FgTexture*             importedTexture;

        FgTextureSubResource   subResource;

    public:
        friend class FrameGraphCompiler;
        friend class FrameGraphExecutor;
        friend class FrameGraphBuilder;
        friend struct PassNode;

    public:
        ResourceNode& SetImportedResource(FgBuffer* buffer);
        ResourceNode& SetImportedResource(FgTexture* texture, const FgTextureSubResource& subResource);
    };

    struct IFRIT_APIDECL PassNode
    {
    private:
        PassNodeId                             id;
        FrameGraphPassType                     type;
        String                                 name;
        bool                                   isImported;
        Fn<void(const FrameGraphPassContext&)> passFunction;
        Vec<ResourceNodeId>                    inputResources;
        Vec<ResourceNodeId>                    outputResources;
        Vec<ResourceNodeId>                    dependentResources;

    public:
        friend class FrameGraphCompiler;
        friend class FrameGraphExecutor;
        friend class FrameGraphBuilder;

        PassNode& AddReadResource(const ResourceNode& res);
        PassNode& AddWriteResource(const ResourceNode& res);
        PassNode& SetExecutionFunction(Fn<void(const FrameGraphPassContext&)> func);

        // Legacy Interface, should be removed in the future.
        PassNode& AddDependentResource(const ResourceNode& res);
    };

    class IFRIT_APIDECL FrameGraphBuilder
    {
    private:
        Vec<ResourceNode*> m_resources;
        Vec<PassNode*>     m_passes;

    public:
        ~FrameGraphBuilder();

        ResourceNode& AddResource(const String& name);
        PassNode&     AddPass(const String& name, FrameGraphPassType type);

        friend class FrameGraphCompiler;
        friend class FrameGraphExecutor;
    };

    struct CompiledFrameGraph
    {
        struct ResourceBarriers
        {
            bool                            enableUAVBarrier        = false;
            bool                            enableTransitionBarrier = false;
            Graphics::Rhi::RhiResourceState srcState;
            Graphics::Rhi::RhiResourceState dstState = Graphics::Rhi::RhiResourceState::Undefined;
        };
        const FrameGraphBuilder*   m_graph                          = nullptr;
        Vec<u32>                   m_passTopoOrder                  = {};
        Vec<Vec<ResourceBarriers>> m_passResourceBarriers           = {};
        Vec<Vec<ResourceBarriers>> m_outputAliasedResourcesBarriers = {};
        Vec<Vec<u32>>              m_inputResourceDependencies      = {};
    };

    class IFRIT_APIDECL FrameGraphCompiler
    {
    private:
    public:
        CompiledFrameGraph Compile(const FrameGraphBuilder& graph);
    };

    class IFRIT_APIDECL FrameGraphExecutor
    {
    public:
        void ExecuteInSingleCmd(const Graphics::Rhi::RhiCommandList* cmd, const CompiledFrameGraph& compiledGraph);

    private:
        Graphics::Rhi::RhiResourceBarrier ToRhiResBarrier(const CompiledFrameGraph::ResourceBarriers& barrier, const ResourceNode& res, bool& valid);
    };

} // namespace Ifrit::Core