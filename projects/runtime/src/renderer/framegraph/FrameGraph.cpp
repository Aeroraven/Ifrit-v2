
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
    IFRIT_APIDECL PassNode& PassNode::AddDependentResource(const ResourceNode& res)
    {
        dependentResources.push_back(res.id);
        return *this;
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

    // Frame Graph compiler

    // Layout transition:
    // (pass)->(output:srcLayout)->(input:dstLayout)
    Graphics::Rhi::RhiResourceState GetOutputResouceState(FrameGraphPassType passType, FrameGraphResourceType resType,
        FgTexture* image, FgBuffer* buffer, Graphics::Rhi::RhiResourceState parentState)
    {
        if (parentState != Graphics::Rhi::RhiResourceState::Undefined)
        {
            return parentState;
        }
        if (resType == FrameGraphResourceType::ResourceBuffer)
        {
            return Graphics::Rhi::RhiResourceState::UnorderedAccess;
        }
        else if (resType == FrameGraphResourceType::ResourceTexture)
        {
            if (passType == FrameGraphPassType::Graphics)
            {
                if (image->IsDepthTexture())
                {
                    return Graphics::Rhi::RhiResourceState::DepthStencilRT;
                }
                else
                {
                    return Graphics::Rhi::RhiResourceState::ColorRT;
                }
            }
            else if (passType == FrameGraphPassType::Compute)
            {
                // TODO: check if it's UAV or SRV
                return Graphics::Rhi::RhiResourceState::Common;
            }
        }
        return Graphics::Rhi::RhiResourceState::Undefined;
    }
    Graphics::Rhi::RhiResourceState GetInputResourceState(
        FrameGraphPassType passType, FrameGraphResourceType resType, FgTexture* image, FgBuffer* buffer)
    {
        if (resType == FrameGraphResourceType::ResourceBuffer)
        {
            return Graphics::Rhi::RhiResourceState::UnorderedAccess;
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
        }
        return Graphics::Rhi::RhiResourceState::Undefined;
    }

    Graphics::Rhi::RhiResourceState GetDesiredOutputLayout(
        FrameGraphPassType passType, FrameGraphResourceType resType, FgTexture* image, FgBuffer* buffer)
    {
        if (resType == FrameGraphResourceType::ResourceBuffer)
        {
            return Graphics::Rhi::RhiResourceState::UnorderedAccess;
        }
        else if (resType == FrameGraphResourceType::ResourceTexture)
        {
            if (passType == FrameGraphPassType::Graphics)
            {
                if (image->IsDepthTexture())
                {
                    return Graphics::Rhi::RhiResourceState::DepthStencilRT;
                }
                else
                {
                    return Graphics::Rhi::RhiResourceState::ColorRT;
                }
            }
            else if (passType == FrameGraphPassType::Compute)
            {
                return Graphics::Rhi::RhiResourceState::UnorderedAccess;
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

        Vec<RhiResourceState>            resState(graph.m_resources.size(), RhiResourceState::Undefined);

        HashMap<void*, RhiResourceState> rawResourceState;
        HashMap<void*, bool>             rawResourceIsWriting;

        for (const auto& pass : graph.m_passes)
        {
            compiledGraph.m_inputBarriers.push_back({});
            // Make transitions for read resources
            for (auto& resId : pass->inputResources)
            {
                auto& res = graph.m_resources[resId];
                auto  desiredLayout =
                    GetInputResourceState(pass->type, res->type, res->importedTexture, res->importedBuffer);

                // Get aliased resource state
                void* resPtr = nullptr;
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

                auto rawResState = rawResourceState[resPtr];
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
                        if (rawResourceIsWriting[resPtr])
                        {
                            // If the resource is writing, then we need to make a uav barrier to prevent RAW
                            CompiledFrameGraph::ResourceBarrier aliasBarrier;
                            aliasBarrier.m_ResId          = resId;
                            aliasBarrier.enableUAVBarrier = true;
                            compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                        }
                    }
                    rawResourceState[resPtr]     = desiredLayout;
                    rawResourceIsWriting[resPtr] = false;
                }
            }

            // Make transitions for write resources
            for (auto& resId : pass->outputResources)
            {
                auto& res = graph.m_resources[resId];
                auto  desiredLayout =
                    GetDesiredOutputLayout(pass->type, res->type, res->importedTexture, res->importedBuffer);

                // Get aliased resource state
                void* resPtr = nullptr;
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
                auto rawResState = rawResourceState[resPtr];

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
                    rawResourceState[resPtr] = desiredLayout;
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
            if (res.type == FrameGraphResourceType::ResourceBuffer)
            {
                resBarrier.m_transition.m_buffer = res.importedBuffer;
            }
            else
            {
                resBarrier.m_transition.m_texture     = res.importedTexture;
                resBarrier.m_transition.m_subResource = res.subResource;
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
        using namespace Ifrit::Graphics::Rhi;
        for (auto pass : compiledGraph.m_graph->m_passes)
        {
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

            pass->passFunction(passContext);
        }
    }

} // namespace Ifrit::Runtime