
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

#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/typing/Util.h"
#include <stdexcept>

using Ifrit::SizeCast;

namespace Ifrit::Runtime
{
    IFRIT_APIDECL ResourceNode& ResourceNode::SetImportedResource(FgBuffer* buffer)
    {
        isImported     = true;
        importedBuffer = buffer;
        type           = FrameGraphResourceType::ResourceBuffer;
        return *this;
    }

    IFRIT_APIDECL ResourceNode& ResourceNode::SetImportedResource(
        FgTexture* texture, const FgTextureSubResource& subResource)
    {
        isImported        = true;
        importedTexture   = texture;
        type              = FrameGraphResourceType::ResourceTexture;
        this->subResource = subResource;
        return *this;
    }

    IFRIT_APIDECL PassNode& PassNode::SetExecutionFunction(Fn<void(const FrameGraphPassContext&)> func)
    {
        passFunction = func;
        return *this;
    }

    IFRIT_APIDECL PassNode& PassNode::AddReadResource(const ResourceNode& res)
    {
        inputResources.push_back(res.id);
        return *this;
    }
    IFRIT_APIDECL PassNode& PassNode::AddWriteResource(const ResourceNode& res)
    {
        outputResources.push_back(res.id);
        return *this;
    }
    IFRIT_APIDECL PassNode& PassNode::AddReadWriteResource(const ResourceNode& res)
    {
        inputResources.push_back(res.id);
        outputResources.push_back(res.id);
        return *this;
    }
    IFRIT_APIDECL PassNode& PassNode::AddDependentResource(const ResourceNode& res)
    {
        dependentResources.push_back(res.id);
        return *this;
    }

    IFRIT_APIDECL void PassNode::Execute(const FrameGraphPassContext& ctx)
    {
        ctx.m_CmdList->BeginScope(String("Ifrit/RDG: [Common] ") + name);
        if (passFunction)
        {
            passFunction(ctx);
        }
        ctx.m_CmdList->EndScope();
    }

    // Specialized nodes
    IFRIT_APIDECL GraphicsPassNode::GraphicsPassNode(Uref<Graphics::Rhi::RhiGraphicsPass>&& pass)
        : m_pass(std::move(pass))
    {
    }

    IFRIT_APIDECL void GraphicsPassNode::Execute(const FrameGraphPassContext& ctx)
    {
        m_pass->SetRecordFunction(
            [this, &ctx](const Graphics::Rhi::RhiRenderPassContext* ct) { this->passFunction(ctx); });
        ctx.m_CmdList->BeginScope(String("Ifrit/RDG: [Draw] ") + name);
        m_pass->Run(ctx.m_CmdList, this->m_renderTargets, 0);
        ctx.m_CmdList->EndScope();
    }

    IFRIT_APIDECL ComputePassNode::ComputePassNode(Uref<Graphics::Rhi::RhiComputePass>&& pass) : m_pass(std::move(pass))
    {
    }

    IFRIT_APIDECL void ComputePassNode::Execute(const FrameGraphPassContext& ctx)
    {
        m_pass->SetRecordFunction(
            [this, &ctx](const Graphics::Rhi::RhiRenderPassContext* ct) { this->passFunction(ctx); });
        ctx.m_CmdList->BeginScope(String("Ifrit/RDG: [Compute] ") + name);
        m_pass->Run(ctx.m_CmdList, 0);
        ctx.m_CmdList->EndScope();
    }

    IFRIT_APIDECL FrameGraphBuilder::~FrameGraphBuilder()
    {
        for (auto& res : m_resources)
        {
            delete res;
        }
        for (auto& pass : m_passes)
        {
            delete pass;
        }
    }

    IFRIT_APIDECL ResourceNode& FrameGraphBuilder::AddResource(const String& name)
    {
        ResourceNode* node = new ResourceNode();
        node->id           = SizeCast<u32>(m_resources.size());
        node->type         = FrameGraphResourceType::Undefined;
        node->name         = name;
        node->isImported   = false;
        m_resources.push_back(node);

        // iInfo("Resource ID:{}  Name:{}", node.id, node.name);
        return *node;
    }

    IFRIT_APIDECL PassNode& FrameGraphBuilder::AddPass(const String& name, FrameGraphPassType type)
    {
        PassNode* node   = new PassNode();
        node->type       = type;
        node->id         = SizeCast<u32>(m_passes.size());
        node->name       = name;
        node->isImported = false;
        // node.inputResources     = inputs;
        // node.outputResources    = outputs;
        // node.dependentResources = dependencies;
        m_passes.push_back(node);
        return *node;
    }

    IFRIT_APIDECL ComputePassNode& FrameGraphBuilder::AddComputePass(
        const String& name, const String& shader, u32 pushConsts)
    {
        auto cp = m_Rhi->CreateComputePass2();
        cp->SetComputeShader(m_ShaderRegistry->GetShader(shader, 0));
        cp->SetPushConstSize(pushConsts * sizeof(u32));

        auto pass        = new ComputePassNode(std::move(cp));
        pass->id         = SizeCast<u32>(m_passes.size());
        pass->name       = name;
        pass->isImported = false;
        pass->type       = FrameGraphPassType::Compute;

        m_passes.push_back(pass);
        return *pass;
    }

    IFRIT_APIDECL GraphicsPassNode& FrameGraphBuilder::AddGraphicsPass(
        const String& name, const String& vs, const String& fs, u32 pushConsts, Graphics::Rhi::RhiRenderTargets* rts)
    {
        auto gp = m_Rhi->CreateGraphicsPass2();
        gp->SetVertexShader(m_ShaderRegistry->GetShader(vs, 0));
        gp->SetPixelShader(m_ShaderRegistry->GetShader(fs, 0));
        gp->SetPushConstSize(pushConsts * sizeof(u32));
        gp->SetRenderTargetFormat(rts->GetFormat());

        auto area = rts->GetRenderArea();
        gp->SetRenderArea(area.x, area.y, area.width, area.height);

        auto pass        = new GraphicsPassNode(std::move(gp));
        pass->id         = SizeCast<u32>(m_passes.size());
        pass->name       = name;
        pass->isImported = false;
        pass->type       = FrameGraphPassType::Graphics;
        pass->SetRenderTargets(rts);

        m_passes.push_back(pass);
        return *pass;
    }

    IFRIT_APIDECL GraphicsPassNode& FrameGraphBuilder::AddMeshGraphicsPass(
        const String& name, const String& ms, const String& fs, u32 pushConsts, Graphics::Rhi::RhiRenderTargets* rts)
    {
        auto gp = m_Rhi->CreateGraphicsPass2();
        gp->SetMeshShader(m_ShaderRegistry->GetShader(ms, 0));
        gp->SetPixelShader(m_ShaderRegistry->GetShader(fs, 0));
        gp->SetPushConstSize(pushConsts * sizeof(u32));
        gp->SetRenderTargetFormat(rts->GetFormat());

        auto area = rts->GetRenderArea();
        gp->SetRenderArea(area.x, area.y, area.width, area.height);

        auto rtx = rts->GetColorAttachment(0);
        gp->SetMsaaSamples(rtx->GetRenderTarget()->GetSamples());

        auto pass        = new GraphicsPassNode(std::move(gp));
        pass->id         = SizeCast<u32>(m_passes.size());
        pass->name       = name;
        pass->isImported = false;
        pass->type       = FrameGraphPassType::Graphics;
        pass->SetRenderTargets(rts);

        m_passes.push_back(pass);
        return *pass;
    }

    ResourceNode& FrameGraphBuilder::DeclareTexture(const String& name, const FrameGraphTextureDesc& desc)
    {
        auto& node = AddResource(name);
        node.type  = FrameGraphResourceType::ResourceTexture;
        node.SetManagedResource(desc);
        return node;
    }

    ResourceNode& FrameGraphBuilder::DeclareBuffer(const String& name, const FrameGraphBufferDesc& desc)
    {
        auto& node = AddResource(name);
        node.type  = FrameGraphResourceType::ResourceBuffer;
        node.SetManagedResource(desc);
        return node;
    }

    Graphics::Rhi::RhiUAVDesc FrameGraphBuilder::GetUAV(const ResourceNode& res)
    {
        iAssertion(!res.isImported, "FrameGraphBuilder: GetUAV() called on imported resource.");

        if (res.type == FrameGraphResourceType::ResourceBuffer)
        {
            iAssertion(res.selfBuffer,
                "FrameGraphBuilder: GetUAV() called on buffer resource that is not created. Lifetime is corrupted.");
            return m_Rhi->GetUAVDescriptor(res.selfBuffer);
        }
        else if (res.type == FrameGraphResourceType::ResourceTexture)
        {
            iAssertion(res.selfTexture,
                "FrameGraphBuilder: GetUAV() called on texture resource that is not created. Lifetime is corrupted.");
            return m_Rhi->GetUAVDescriptor(res.selfTexture);
        }
        iError("FrameGraphBuilder: GetUAV() called on resource that is not a buffer or texture.");
        std::abort();
        return 0;
    }

    Graphics::Rhi::RhiSRVDesc FrameGraphBuilder::GetSRV(const ResourceNode& res)
    {
        iAssertion(!res.isImported, "FrameGraphBuilder: GetSRV() called on imported resource.");
        if (res.type == FrameGraphResourceType::ResourceBuffer)
        {
            iAssertion(res.selfBuffer,
                "FrameGraphBuilder: GetSRV() called on buffer resource that is not created. Lifetime is corrupted.");
            return m_Rhi->GetSRVDescriptor(res.selfBuffer);
        }
        else if (res.type == FrameGraphResourceType::ResourceTexture)
        {
            iAssertion(res.selfTexture,
                "FrameGraphBuilder: GetSRV() called on texture resource that is not created. Lifetime is corrupted.");
            return m_Rhi->GetSRVDescriptor(res.selfTexture);
        }
        iError("FrameGraphBuilder: GetSRV() called on resource that is not a buffer or texture.");
        std::abort();
        return 0;
    }

    // Frame Graph compiler

    Graphics::Rhi::RhiResourceState GetInputResourceState(FrameGraphPassType passType, FrameGraphResourceType resType)
    {
        if (resType == FrameGraphResourceType::ResourceBuffer)
        {
            if (passType == FrameGraphPassType::Transfer)
            {
                return Graphics::Rhi::RhiResourceState::CopySrc;
            }
            else
            {
                return Graphics::Rhi::RhiResourceState::UnorderedAccess;
            }
        }
        else if (resType == FrameGraphResourceType::ResourceTexture)
        {
            if (passType == FrameGraphPassType::Graphics)
            {
                return Graphics::Rhi::RhiResourceState::ShaderRead;
            }
            else if (passType == FrameGraphPassType::Compute)
            {
                return Graphics::Rhi::RhiResourceState::UnorderedAccess;
            }
            else if (passType == FrameGraphPassType::Transfer)
            {
                return Graphics::Rhi::RhiResourceState::CopySrc;
            }
        }
        return Graphics::Rhi::RhiResourceState::Undefined;
    }

    Graphics::Rhi::RhiResourceState GetDesiredOutputLayout(
        FrameGraphPassType passType, FrameGraphResourceType resType, ResourceNode* image)
    {
        if (resType == FrameGraphResourceType::ResourceBuffer)
        {
            if (passType == FrameGraphPassType::Transfer)
            {
                return Graphics::Rhi::RhiResourceState::CopyDst;
            }
            else
            {
                return Graphics::Rhi::RhiResourceState::UnorderedAccess;
            }
        }
        else if (resType == FrameGraphResourceType::ResourceTexture)
        {
            if (passType == FrameGraphPassType::Graphics)
            {
                if (image->IsImported())
                {
                    if (image->GetTexture()->IsDepthTexture())
                    {
                        return Graphics::Rhi::RhiResourceState::DepthStencilRT;
                    }
                    else
                    {
                        return Graphics::Rhi::RhiResourceState::ColorRT;
                    }
                }
                else
                {
                    auto desc = image->GetManagedTextureDesc();
                    if (desc.m_Format == Graphics::Rhi::RhiImageFormat::RhiImgFmt_D32_SFLOAT)
                    {
                        return Graphics::Rhi::RhiResourceState::DepthStencilRT;
                    }
                    else
                    {
                        return Graphics::Rhi::RhiResourceState::ColorRT;
                    }
                }
            }
            else if (passType == FrameGraphPassType::Compute)
            {
                return Graphics::Rhi::RhiResourceState::UnorderedAccess;
            }
            else if (passType == FrameGraphPassType::Transfer)
            {
                return Graphics::Rhi::RhiResourceState::CopyDst;
            }
        }
        return Graphics::Rhi::RhiResourceState::Undefined;
    }

    IFRIT_APIDECL CompiledFrameGraph FrameGraphCompiler::Compile(const FrameGraphBuilder& graph)
    {
        using namespace Ifrit::Graphics::Rhi;

        CompiledFrameGraph compiledGraph = {};
        compiledGraph.m_inputBarriers    = {};
        compiledGraph.m_graph            = &graph;

        if (graph.m_compileMode == FrameGraphCompileMode::Unordered)
        {
            iError("Not supported any longer.");
            std::abort();
        }
        // Managed Resource Lifetime
        Vec<u32> resourceBeginUse;
        Vec<u32> resourceEndUse;
        for (u32 i = 0; i < graph.m_resources.size(); i++)
        {
            resourceBeginUse.push_back(graph.m_resources.size());
            resourceEndUse.push_back(0);
        }
        for (u32 i = 0; i < graph.m_passes.size(); i++)
        {
            auto& pass = graph.m_passes[i];
            for (auto& resId : pass->inputResources)
            {
                resourceBeginUse[resId] = std::min(resourceBeginUse[resId], i);
            }
            for (auto& resId : pass->outputResources)
            {
                resourceEndUse[resId] = std::max(resourceEndUse[resId], i);
            }
        }

        for (u32 i = 0; i < graph.m_resources.size(); i++)
        {
            if (graph.m_resources[i]->isImported)
            {
                continue;
            }
            auto res = graph.m_resources[i];
            graph.m_passes[resourceBeginUse[i]]->m_ResourceCreateRequest.push_back(i);
            graph.m_passes[resourceEndUse[i]]->m_ResourceReleaseRequest.push_back(i);
        }

        // Resource Barriers
        Vec<RhiResourceState>            resState(graph.m_resources.size(), RhiResourceState::Undefined);
        HashMap<void*, RhiResourceState> rawResourceState;
        HashMap<void*, bool>             rawResourceIsWriting;
        Vec<RhiResourceState>            managedResourceState(graph.m_resources.size(), RhiResourceState::Undefined);
        Vec<u32>                         managedResourceIsWriting(graph.m_resources.size(), 0);

        for (const auto& pass : graph.m_passes)
        {
            compiledGraph.m_inputBarriers.push_back({});
            // Make transitions for read resources
            for (auto& resId : pass->inputResources)
            {
                auto& res           = graph.m_resources[resId];
                auto  desiredLayout = GetInputResourceState(pass->type, res->type);

                // Get aliased resource state, if it's imported
                void* resPtr = nullptr;
                if (res->isImported)
                {
                    if (res->type == FrameGraphResourceType::ResourceBuffer)
                    {
                        resPtr = res->importedBuffer;
                    }
                    else if (res->type == FrameGraphResourceType::ResourceTexture)
                    {
                        resPtr = res->importedTexture;
                    }
                    if (rawResourceState.find(resPtr) == rawResourceState.end())
                    {
                        rawResourceState[resPtr]     = Graphics::Rhi::RhiResourceState::Undefined;
                        rawResourceIsWriting[resPtr] = false;
                    }
                }

                Graphics::Rhi::RhiResourceState rawResState;
                if (res->isImported)
                {
                    rawResState = rawResourceState[resPtr];
                }
                else
                {
                    rawResState = managedResourceState[resId];
                }

                // Check if input state meets the desired state
                if (desiredLayout != rawResState)
                {
                    if (desiredLayout == Graphics::Rhi::RhiResourceState::Undefined
                        && graph.m_resourceInitState == FrameGraphResourceInitState::Uninitialized)
                    {
                        // If the layout is managed by user, then we don't need to do anything
                        // Just set the state to undefined
                        CompiledFrameGraph::ResourceBarrier aliasBarrier;
                        aliasBarrier.m_ResId                 = resId;
                        aliasBarrier.enableTransitionBarrier = true;
                        aliasBarrier.srcState = Graphics::Rhi::RhiResourceState::AutoTraced; // rawResState;
                        aliasBarrier.dstState = desiredLayout;
                        compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                    }
                    else if (desiredLayout != Graphics::Rhi::RhiResourceState::Undefined)
                    {
                        // Here we need to make a transition barrier
                        CompiledFrameGraph::ResourceBarrier aliasBarrier;
                        aliasBarrier.m_ResId                 = resId;
                        aliasBarrier.enableTransitionBarrier = true;
                        aliasBarrier.srcState = Graphics::Rhi::RhiResourceState::AutoTraced; // rawResState;
                        aliasBarrier.dstState = desiredLayout;
                        compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                    }
                    else
                    {
                        if (res->IsImported() && rawResourceIsWriting[resPtr])
                        {
                            // If the resource is writing, then we need to make a uav barrier to prevent RAW
                            CompiledFrameGraph::ResourceBarrier aliasBarrier;
                            aliasBarrier.m_ResId          = resId;
                            aliasBarrier.enableUAVBarrier = true;
                            compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                        }
                        else if (!res->isImported && managedResourceIsWriting[resId] > 0)
                        {
                            // If the resource is writing, then we need to make a uav barrier to prevent RAW
                            CompiledFrameGraph::ResourceBarrier aliasBarrier;
                            aliasBarrier.m_ResId          = resId;
                            aliasBarrier.enableUAVBarrier = true;
                            compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                        }
                    }
                    if (res->isImported)
                    {
                        rawResourceState[resPtr]     = desiredLayout;
                        rawResourceIsWriting[resPtr] = 0;
                    }
                    else
                    {
                        managedResourceState[resId]     = desiredLayout;
                        managedResourceIsWriting[resId] = 0;
                    }
                }
            }

            // Make transitions for write resources
            for (auto& resId : pass->outputResources)
            {
                auto& res           = graph.m_resources[resId];
                auto  desiredLayout = GetDesiredOutputLayout(pass->type, res->type, res);

                // Get aliased resource state
                void* resPtr = nullptr;
                if (res->isImported)
                {
                    if (res->type == FrameGraphResourceType::ResourceBuffer)
                    {
                        resPtr = res->importedBuffer;
                    }
                    else if (res->type == FrameGraphResourceType::ResourceTexture)
                    {
                        resPtr = res->importedTexture;
                    }
                    if (rawResourceState.find(resPtr) == rawResourceState.end())
                    {
                        rawResourceState[resPtr]     = Graphics::Rhi::RhiResourceState::Undefined;
                        rawResourceIsWriting[resPtr] = true;
                    }
                }

                Graphics::Rhi::RhiResourceState rawResState;
                if (res->isImported)
                {
                    rawResState = rawResourceState[resPtr];
                }
                else
                {
                    rawResState = managedResourceState[resId];
                }
                // Check if input state meets the desired state
                if (desiredLayout != rawResState)
                {
                    if (desiredLayout == Graphics::Rhi::RhiResourceState::Undefined
                        && graph.m_resourceInitState == FrameGraphResourceInitState::Uninitialized)
                    {
                        // If the layout is managed by user, then we don't need to do anything
                        // Just set the state to undefined
                        CompiledFrameGraph::ResourceBarrier aliasBarrier;
                        aliasBarrier.m_ResId                 = resId;
                        aliasBarrier.enableTransitionBarrier = true;
                        aliasBarrier.srcState = Graphics::Rhi::RhiResourceState::AutoTraced; // rawResState;
                        aliasBarrier.dstState = desiredLayout;
                        compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                    }
                    else if (desiredLayout != Graphics::Rhi::RhiResourceState::Undefined)
                    {
                        // Here we need to make a transition barrier
                        CompiledFrameGraph::ResourceBarrier aliasBarrier;
                        aliasBarrier.m_ResId                 = resId;
                        aliasBarrier.enableTransitionBarrier = true;
                        aliasBarrier.srcState = Graphics::Rhi::RhiResourceState::AutoTraced; // rawResState;
                        aliasBarrier.dstState = desiredLayout;
                        compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                    }
                    if (res->isImported)
                    {
                        rawResourceState[resPtr]     = desiredLayout;
                        rawResourceIsWriting[resPtr] = 1;
                    }
                    else
                    {
                        managedResourceState[resId] = desiredLayout;
                        managedResourceIsWriting[resId] += 1;
                    }
                }
            }
        }
        return compiledGraph;
    }

    Graphics::Rhi::RhiResourceBarrier FrameGraphExecutor::ToRhiResBarrier(
        const CompiledFrameGraph::ResourceBarrier& barrier, const ResourceNode& res, bool& valid)
    {
        Graphics::Rhi::RhiResourceBarrier resBarrier;
        valid = false;
        if (barrier.enableTransitionBarrier)
        {
            resBarrier.m_type              = Graphics::Rhi::RhiBarrierType::Transition;
            resBarrier.m_transition.m_type = res.type == FrameGraphResourceType::ResourceBuffer
                ? Graphics::Rhi::RhiResourceType::Buffer
                : Graphics::Rhi::RhiResourceType::Texture;
            if (res.isImported)
            {
                if (res.type == FrameGraphResourceType::ResourceBuffer)
                {
                    resBarrier.m_transition.m_buffer = res.importedBuffer;
                }
                else
                {
                    resBarrier.m_transition.m_texture     = res.importedTexture;
                    resBarrier.m_transition.m_subResource = res.subResource;
                }
            }
            else
            {
                if (res.type == FrameGraphResourceType::ResourceBuffer)
                {
                    resBarrier.m_transition.m_buffer = res.selfBuffer;
                }
                else
                {
                    resBarrier.m_transition.m_texture     = res.selfTexture;
                    resBarrier.m_transition.m_subResource = res.subResource;
                }
            }

            resBarrier.m_transition.m_srcState = Graphics::Rhi::RhiResourceState::AutoTraced; // barrier.srcState;
            resBarrier.m_transition.m_dstState = barrier.dstState;
            valid                              = true;
        }
        else if (barrier.enableUAVBarrier)
        {
            resBarrier.m_type       = Graphics::Rhi::RhiBarrierType::UAVAccess;
            resBarrier.m_uav.m_type = res.type == FrameGraphResourceType::ResourceBuffer
                ? Graphics::Rhi::RhiResourceType::Buffer
                : Graphics::Rhi::RhiResourceType::Texture;
            if (res.type == FrameGraphResourceType::ResourceBuffer)
            {
                resBarrier.m_uav.m_buffer = res.importedBuffer;
            }
            else
            {
                resBarrier.m_uav.m_texture = res.importedTexture;
            }
            valid = true;
        }

        return resBarrier;
    }

    // Execute the compiled frame graph
    IFRIT_APIDECL void FrameGraphExecutor::ExecuteInSingleCmd(
        const Graphics::Rhi::RhiCommandList* cmd, const CompiledFrameGraph& compiledGraph)
    {
        cmd->BeginScope("Ifrit/RDG: Render Graph Execution");
        using namespace Ifrit::Graphics::Rhi;
        for (auto pass : compiledGraph.m_graph->m_passes)
        {
            // PreExecute
            for (u32 i = 0; i < pass->m_ResourceCreateRequest.size(); i++)
            {
                auto res = compiledGraph.m_graph->m_resources[pass->m_ResourceCreateRequest[i]];
                iAssertion(!res->isImported, "Resource should not be imported.");

                if (res->type == FrameGraphResourceType::ResourceBuffer)
                {
                    auto resAlloc      = compiledGraph.m_graph->m_ResourcePool->CreateBuffer(res->bufferDesc);
                    res->m_PooledResId = resAlloc.m_PooledResId;
                    res->selfBuffer    = resAlloc.m_Buffer;
                }
                else if (res->type == FrameGraphResourceType::ResourceTexture)
                {
                    auto resAlloc      = compiledGraph.m_graph->m_ResourcePool->CreateTexture(res->textureDesc);
                    res->m_PooledResId = resAlloc.m_PooledResId;
                    res->selfTexture   = resAlloc.m_Texture;
                    res->subResource   = { 0, 0, 1, 1 };
                }
            }

            // Execute
            Vec<RhiResourceBarrier> outputBarriers;
            for (auto& barrier : compiledGraph.m_inputBarriers[pass->id])
            {
                bool valid      = false;
                auto resBarrier = ToRhiResBarrier(barrier, *compiledGraph.m_graph->m_resources[barrier.m_ResId], valid);
                if (valid)
                    outputBarriers.push_back(resBarrier);
            }
            cmd->AddResourceBarrier(outputBarriers);

            FrameGraphPassContext passContext;
            passContext.m_CmdList = cmd;

            pass->FillContext(passContext);
            pass->Execute(passContext);

            // PostExecute
            for (u32 i = 0; i < pass->m_ResourceReleaseRequest.size(); i++)
            {
                auto res = compiledGraph.m_graph->m_resources[pass->m_ResourceReleaseRequest[i]];
                iAssertion(!res->isImported, "Resource should not be imported.");

                if (res->type == FrameGraphResourceType::ResourceBuffer)
                {
                    res->selfBuffer = nullptr;
                    compiledGraph.m_graph->m_ResourcePool->ReleaseBuffer(res->m_PooledResId);
                    res->m_PooledResId = RIndexedPtr(0);
                }
                else if (res->type == FrameGraphResourceType::ResourceTexture)
                {
                    res->selfTexture = nullptr;
                    compiledGraph.m_graph->m_ResourcePool->ReleaseTexture(res->m_PooledResId);
                    res->m_PooledResId = RIndexedPtr(0);
                }
            }
        }
        cmd->EndScope();
    }

} // namespace Ifrit::Runtime