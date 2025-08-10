
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
#include "ifrit/rhi/common/RhiStructHelper.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/renderer/profiling/ProfileDataManager.h"
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
        ctx.m_CmdList->BeginScope(String("Ifrit.RDG.General: ") + name);
        if (passFunction)
        {
            passFunction(ctx);
        }
        ctx.m_CmdList->EndScope();
    }

    // Specialized nodes
    IFRIT_APIDECL GraphicsPassNode::GraphicsPassNode(Owner<RHI::RhiGraphicsPass>&& pass) : m_pass(std::move(pass)) {}

    IFRIT_APIDECL void GraphicsPassNode::Execute(const FrameGraphPassContext& ctx)
    {

        m_pass->SetRecordFunction([this, &ctx](const RHI::RhiRenderPassContext* ct) { this->passFunction(ctx); });
        ctx.m_CmdList->BeginScope(String("Ifrit.RDG.Draw: ") + name);
        m_pass->Run(ctx.m_CmdList, this->m_RhiRTs.get(), 0);
        ctx.m_CmdList->EndScope();
    }

    IFRIT_APIDECL void GraphicsPassNode::ComposeRenderTargets(RHI::RhiBackend* rhiBackend)
    {
        if (m_RTComposed)
            return;
        m_RhiRTs = rhiBackend->CreateRenderTargets();
        Vec<RHI::RhiColorAttachment*> crts;
        for (u32 i = 0; i < m_RenderTarget.size(); i++)
        {
            auto                res = m_RenderTarget[i];
            RHI::RhiClearValue2 clearValue(RHI::CreateRhiClearColorValue(m_ColorClearValue[i]));
            auto rt = rhiBackend->CreateRenderTarget(res->GetTexture(), clearValue, m_ColorLoadOp[i], 0, 0);
            crts.push_back(rt.get());
            m_RhiColorRTs.push_back(rt);

            if (i == 0)
                m_RhiRTs->SetRenderArea({ 0, 0, res->GetTexture()->GetWidth(), res->GetTexture()->GetHeight() });
        }
        m_RhiRTs->SetColorAttachments(crts);
        if (m_DepthTarget != nullptr)
        {
            RHI::RhiClearValue2 clearValue(RHI::CreateRhiClearDepthStencilValue(m_DepthClearValue, 0));
            auto                rt =
                rhiBackend->CreateRenderTargetDepthStencil(m_DepthTarget->GetTexture(), clearValue, m_DepthLoadOp);
            m_RhiDepthRT = rt;
            m_RhiRTs->SetDepthStencilAttachment(rt.get());
        }
        m_pass->SetRenderTargetFormat(m_RhiRTs->GetFormat());
        if (m_Scissor.width == 0 && m_Scissor.height == 0)
        {
            auto rdArea      = m_RhiRTs->GetRenderArea();
            m_Scissor.x      = rdArea.x;
            m_Scissor.y      = rdArea.y;
            m_Scissor.width  = rdArea.width;
            m_Scissor.height = rdArea.height;
            m_pass->SetRenderArea(m_Scissor.x, m_Scissor.y, m_Scissor.width, m_Scissor.height);
        }

        if (m_RhiColorRTs.size() == 0 && m_RhiDepthRT == nullptr)
        {
            IF_LOG_CRITICAL("FrameGraph", "No render targets are set for the pass: {}.", name);
            std::abort();
        }
        if (m_Scissor.width == 0 && m_Scissor.height == 0)
        {
            IF_LOG_CRITICAL("FrameGraph", "No render area is set for the pass: {}.", name);
        }
        m_RTComposed = true;
    }

    IFRIT_APIDECL GraphicsPassNode& GraphicsPassNode::AddRenderTarget(
        ResourceNode& res, LoadOp loadOp, Vector4f clearValue)
    {
        if (res.GetType() != FrameGraphResourceType::ResourceTexture)
        {
            IF_LOG_CRITICAL("FrameGraph", "Render target must be a texture resource.");
        }
        auto resPtr = &res;
        m_RenderTarget.push_back(resPtr);
        AddWriteResource(res);
        m_ColorClearValue.push_back(clearValue);
        m_ColorLoadOp.push_back(loadOp);

        return *this;
    }

    IFRIT_APIDECL GraphicsPassNode& GraphicsPassNode::AddDepthTarget(ResourceNode& res, LoadOp loadOp, f32 clearValue)
    {
        if (res.GetType() != FrameGraphResourceType::ResourceTexture)
        {
            IF_LOG_CRITICAL("FrameGraph", "Depth target must be a texture resource.");
        }
        auto resPtr   = &res;
        m_DepthTarget = resPtr;
        AddWriteResource(res);
        m_DepthLoadOp     = loadOp;
        m_DepthClearValue = clearValue;

        return *this;
    }

    IFRIT_APIDECL      ComputePassNode::ComputePassNode(Owner<RHI::RhiComputePass>&& pass) : m_pass(std::move(pass)) {}

    IFRIT_APIDECL void ComputePassNode::Execute(const FrameGraphPassContext& ctx)
    {
        m_pass->SetRecordFunction([this, &ctx](const RHI::RhiRenderPassContext* ct) { this->passFunction(ctx); });
        ctx.m_CmdList->BeginScope(String("Ifrit.RDG.Compute: ") + name);
        m_pass->Run(ctx.m_CmdList, 0);
        ctx.m_CmdList->EndScope();
    }

    IFRIT_APIDECL               FrameGraphBuilder::~FrameGraphBuilder() {}

    IFRIT_APIDECL ResourceNode& FrameGraphBuilder::AddResource(const String& name)
    {
        // ResourceNode* node = new ResourceNode();
        auto node        = MakeOwner<ResourceNode>();
        node->id         = SizeCast<u32>(m_resources.size());
        node->type       = FrameGraphResourceType::Undefined;
        node->name       = name;
        node->isImported = false;

        auto resPtr = node.get();
        m_resources.push_back(std::move(node));
        return *resPtr;
    }

    IFRIT_APIDECL PassNode& FrameGraphBuilder::AddPass(const String& name, FrameGraphPassType type)
    {
        // PassNode* node   = new PassNode();
        auto node        = MakeOwner<PassNode>();
        node->type       = type;
        node->id         = SizeCast<u32>(m_passes.size());
        node->name       = name;
        node->isImported = false;
        // node.inputResources     = inputs;
        // node.outputResources    = outputs;
        // node.dependentResources = dependencies;
        // m_passes.push_back(node);
        auto passPtr = node.get();
        m_passes.push_back(std::move(node));
        return *passPtr;
    }

    IFRIT_APIDECL ComputePassNode& FrameGraphBuilder::AddComputePass(
        const String& name, const ShaderVariantDesc& shader, u32 pushConsts)
    {
        auto cp = m_Rhi->CreateComputePass2();
        cp->SetComputeShader(m_ShaderRegistry->GetShader(shader));
        cp->SetPushConstSize(pushConsts * sizeof(u32));

        auto pass        = MakeOwner<ComputePassNode>(std::move(cp));
        pass->id         = SizeCast<u32>(m_passes.size());
        pass->name       = name;
        pass->isImported = false;
        pass->type       = FrameGraphPassType::Compute;

        auto passPtr = pass.get();
        m_passes.push_back(std::move(pass));
        return *passPtr;
    }

    IFRIT_APIDECL GraphicsPassNode& FrameGraphBuilder::AddGraphicsPass(const String& name, const ShaderVariantDesc& vs,
        const ShaderVariantDesc& fs, u32 pushConsts, RHI::RhiRasterizerTopology topology)
    {
        auto gp = m_Rhi->CreateGraphicsPass2();
        gp->SetVertexShader(m_ShaderRegistry->GetShader(vs));
        gp->SetPixelShader(m_ShaderRegistry->GetShader(fs));
        gp->SetPushConstSize(pushConsts * sizeof(u32));
        gp->SetRasterizerTopology(topology);

        auto pass        = MakeOwner<GraphicsPassNode>(std::move(gp));
        pass->id         = SizeCast<u32>(m_passes.size());
        pass->name       = name;
        pass->isImported = false;
        pass->type       = FrameGraphPassType::Graphics;

        auto passPtr = pass.get();
        m_passes.push_back(std::move(pass));
        return *passPtr;
    }

    IFRIT_APIDECL GraphicsPassNode& FrameGraphBuilder::AddMeshGraphicsPass(
        const String& name, const ShaderVariantDesc& ms, const ShaderVariantDesc& fs, u32 pushConsts)
    {
        auto gp = m_Rhi->CreateGraphicsPass2();
        gp->SetMeshShader(m_ShaderRegistry->GetShader(ms));
        gp->SetPixelShader(m_ShaderRegistry->GetShader(fs));
        gp->SetPushConstSize(pushConsts * sizeof(u32));

        // auto pass        = new GraphicsPassNode(std::move(gp));
        auto pass        = MakeOwner<GraphicsPassNode>(std::move(gp));
        pass->id         = SizeCast<u32>(m_passes.size());
        pass->name       = name;
        pass->isImported = false;
        pass->type       = FrameGraphPassType::Graphics;

        auto passPtr = pass.get();
        m_passes.push_back(std::move(pass));
        return *passPtr;
    }

    IFRIT_APIDECL ResourceNode& FrameGraphBuilder::DeclareTexture(const String& name, const FrameGraphTextureDesc& desc)
    {
        auto& node = AddResource(name);
        node.type  = FrameGraphResourceType::ResourceTexture;
        node.SetManagedResource(desc);
        return node;
    }

    IFRIT_APIDECL ResourceNode& FrameGraphBuilder::DeclareBuffer(const String& name, const FrameGraphBufferDesc& desc)
    {
        auto& node = AddResource(name);
        node.type  = FrameGraphResourceType::ResourceBuffer;
        node.SetManagedResource(desc);
        return node;
    }

    ResourceNode& FrameGraphBuilder::ImportTexture(
        const String& name, FgTexture* texture, const FgTextureSubResource& subResource)
    {
        if (!texture)
        {
            IF_LOG_ASSERTION("FrameGraph", false, "ImportTexture called with null texture.");
        }
        auto& node = AddResource(name);
        node.SetImportedResource(texture, subResource);
        return node;
    }
    ResourceNode& FrameGraphBuilder::ImportBuffer(const String& name, FgBuffer* buffer)
    {
        if (!buffer)
        {
            IF_LOG_ASSERTION("FrameGraph", false, "ImportBuffer called with null buffer.");
        }
        auto& node = AddResource(name);
        node.SetImportedResource(buffer);
        return node;
    }

    RHI::RhiUAVDesc FrameGraphBuilder::GetUAV(const ResourceNode& res) const
    {
        if (res.isImported)
        {
            if (res.type == FrameGraphResourceType::ResourceBuffer)
            {
                return m_Rhi->GetUAVDescriptor(res.importedBuffer);
            }
            else if (res.type == FrameGraphResourceType::ResourceTexture)
            {
                return m_Rhi->GetUAVDescriptor(res.importedTexture, res.subResource);
            }
        }
        if (res.type == FrameGraphResourceType::ResourceBuffer)
        {
            IF_LOG_ASSERTION("FrameGraph", res.selfBuffer,
                "GetUAV() called on buffer resource that is not created. Lifetime is corrupted. Resource: {}",
                res.name);
            return m_Rhi->GetUAVDescriptor(res.selfBuffer);
        }
        else if (res.type == FrameGraphResourceType::ResourceTexture)
        {
            IF_LOG_ASSERTION("FrameGraph", res.selfTexture,
                "GetUAV() called on texture resource that is not created. Lifetime is corrupted. Resource: {}",
                res.name);
            return m_Rhi->GetUAVDescriptor(res.selfTexture);
        }
        IF_LOG_CRITICAL("FrameGraph", "GetUAV() called on resource that is not a buffer or texture.");
        return 0;
    }

    RHI::RhiSRVDesc FrameGraphBuilder::GetSRV(const ResourceNode& res) const
    {
        if (res.isImported)
        {
            if (res.type == FrameGraphResourceType::ResourceBuffer)
            {
                return m_Rhi->GetSRVDescriptor(res.importedBuffer);
            }
            else if (res.type == FrameGraphResourceType::ResourceTexture)
            {
                return m_Rhi->GetSRVDescriptor(res.importedTexture, res.subResource);
            }
        }
        if (res.type == FrameGraphResourceType::ResourceBuffer)
        {
            IF_LOG_ASSERTION("FrameGraph", res.selfBuffer,
                "GetSRV() called on buffer resource that is not created. Lifetime is corrupted.");
            return m_Rhi->GetSRVDescriptor(res.selfBuffer);
        }
        else if (res.type == FrameGraphResourceType::ResourceTexture)
        {
            IF_LOG_ASSERTION("FrameGraph", res.selfTexture,
                "GetSRV() called on texture resource that is not created. Lifetime is corrupted.");
            return m_Rhi->GetSRVDescriptor(res.selfTexture);
        }
        IF_LOG_CRITICAL("FrameGraph", "GetSRV() called on resource that is not a buffer or texture.");
        std::abort();
        return 0;
    }

    RHI::RhiCBVDesc FrameGraphBuilder::GetCBV(const ResourceNode& res) const
    {
        if (res.isImported)
        {
            if (res.type == FrameGraphResourceType::ResourceBuffer)
            {
                return m_Rhi->GetCBVDescriptor(res.importedBuffer);
            }
        }
        if (res.type == FrameGraphResourceType::ResourceBuffer)
        {
            IF_LOG_ASSERTION("FrameGraph", res.selfBuffer,
                "GetCBV() called on buffer resource that is not created. Lifetime is corrupted.");
            return m_Rhi->GetCBVDescriptor(res.selfBuffer);
        }
        IF_LOG_CRITICAL("FrameGraph", "GetCBV() called on resource that is not a buffer.");
        std::abort();
        return 0;
    }

    IFRIT_APIDECL FrameGraphScope& FrameGraphBuilder::AddScopeBegin(const String& name)
    {
        Owner<FrameGraphScope> scope = MakeOwner<FrameGraphScope>();
        scope->m_Name                = name;
        scope->m_StartingPassId      = SizeCast<u32>(m_passes.size());
        auto scopeId                 = SizeCast<u32>(m_scopes.size());
        scope->m_ScopeId             = scopeId;
        auto ptr                     = scope.get();
        m_scopes.push_back(std::move(scope));
        return *ptr;
    }
    IFRIT_APIDECL void FrameGraphBuilder::AddScopeEnd(const FrameGraphScope& scope)
    {
        FrameGraphScope& scopex = *m_scopes[scope.m_ScopeId];
        scopex.m_EndingPassId   = std::max(scopex.m_StartingPassId, (u32)std::max(0, (i32)m_passes.size()));
    }

    IFRIT_APIDECL FrameGraphStatScope& FrameGraphBuilder::AddStatScopeBegin(const String& name)
    {
        Owner<FrameGraphStatScope> scope = MakeOwner<FrameGraphStatScope>();
        scope->m_Name                    = name;
        scope->m_StartingPassId          = SizeCast<u32>(m_passes.size());
        auto scopeId                     = SizeCast<u32>(m_statScopes.size());
        scope->m_ScopeId                 = scopeId;
        auto ptr                         = scope.get();
        //IF_LOG_INFO("FrameGraph", "Adding stat scope: {} at pass id: {}.", name, scope->m_StartingPassId);

        m_statScopes.push_back(std::move(scope));
        return *ptr;
    }
    IFRIT_APIDECL void FrameGraphBuilder::AddStatScopeEnd(const FrameGraphStatScope& scope)
    {
        FrameGraphStatScope& scopex = *m_statScopes[scope.m_ScopeId];
        scopex.m_EndingPassId       = std::max(scopex.m_StartingPassId, (u32)std::max(0, (i32)m_passes.size()));

        //IF_LOG_INFO("FrameGraph", "Ending stat scope: {} at pass id: {}.", scopex.m_Name, scopex.m_EndingPassId);
        if (scopex.m_EndingPassId <= scopex.m_StartingPassId)
        {
            IF_LOG_CRITICAL("FrameGraph", "Stat scope {} has no ending pass.", scopex.m_Name);
            throw std::runtime_error("Stat scope has no ending pass.");
        }
        if (scopex.m_EndingPassId > m_passes.size())
        {
            IF_LOG_CRITICAL(
                "FrameGraph", "Stat scope {} has ending pass that exceeds the number of passes.", scopex.m_Name);
            throw std::runtime_error("Stat scope has ending pass that exceeds the number of passes.");
        }
    }

    // Frame Graph compiler

    RHI::RhiResourceState GetInputResourceState(FrameGraphPassType passType, FrameGraphResourceType resType)
    {
        if (resType == FrameGraphResourceType::ResourceBuffer)
        {
            if (passType == FrameGraphPassType::Transfer)
            {
                return RHI::RhiResourceState::CopySrc;
            }
            else
            {
                return RHI::RhiResourceState::UnorderedAccess;
            }
        }
        else if (resType == FrameGraphResourceType::ResourceTexture)
        {
            if (passType == FrameGraphPassType::Graphics)
            {
                return RHI::RhiResourceState::ShaderRead;
            }
            else if (passType == FrameGraphPassType::Compute)
            {
                return RHI::RhiResourceState::UnorderedAccess;
            }
            else if (passType == FrameGraphPassType::Transfer)
            {
                return RHI::RhiResourceState::CopySrc;
            }
        }
        return RHI::RhiResourceState::Undefined;
    }

    RHI::RhiResourceState GetDesiredOutputLayout(
        FrameGraphPassType passType, FrameGraphResourceType resType, ResourceNode* image)
    {
        if (resType == FrameGraphResourceType::ResourceBuffer)
        {
            if (passType == FrameGraphPassType::Transfer)
            {
                return RHI::RhiResourceState::CopyDst;
            }
            else
            {
                return RHI::RhiResourceState::UnorderedAccess;
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
                        return RHI::RhiResourceState::DepthStencilRT;
                    }
                    else
                    {
                        return RHI::RhiResourceState::ColorRT;
                    }
                }
                else
                {
                    auto desc = image->GetManagedTextureDesc();
                    if (desc.m_Format == RHI::RhiImageFormat::RhiImgFmt_D32_SFLOAT)
                    {
                        return RHI::RhiResourceState::DepthStencilRT;
                    }
                    else
                    {
                        return RHI::RhiResourceState::ColorRT;
                    }
                }
            }
            else if (passType == FrameGraphPassType::Compute)
            {
                return RHI::RhiResourceState::UnorderedAccess;
            }
            else if (passType == FrameGraphPassType::Transfer)
            {
                return RHI::RhiResourceState::CopyDst;
            }
        }
        return RHI::RhiResourceState::Undefined;
    }

    IFRIT_APIDECL CompiledFrameGraph FrameGraphCompiler::Compile(const FrameGraphBuilder& graph)
    {
        using namespace Ifrit::RHI;

        CompiledFrameGraph compiledGraph = {};
        compiledGraph.m_inputBarriers    = {};
        compiledGraph.m_graph            = &graph;

        if (graph.m_compileMode == FrameGraphCompileMode::Unordered)
        {
            IF_LOG_CRITICAL("FrameGraph", "Not supported any longer.");
            std::abort();
        }

        // RDG event scopes
        compiledGraph.m_StartingScopes.clear();
        compiledGraph.m_EndingScopes.clear();
        compiledGraph.m_StartingScopes.resize(graph.m_passes.size() + 1);
        compiledGraph.m_EndingScopes.resize(graph.m_passes.size() + 1, 0);

        compiledGraph.m_StatStartingScopes.clear();
        compiledGraph.m_StatEndingScopes.clear();
        compiledGraph.m_StatStartingScopes.resize(graph.m_passes.size() + 1);
        compiledGraph.m_StatEndingScopes.resize(graph.m_passes.size() + 1);

        for (auto& scope : graph.m_scopes)
        {

            if (scope->m_StartingPassId <= graph.m_passes.size())
            {
                compiledGraph.m_StartingScopes[scope->m_StartingPassId].push_back(scope->m_Name);
            }
            if (scope->m_EndingPassId <= graph.m_passes.size())
            {
                compiledGraph.m_EndingScopes[scope->m_EndingPassId]++;
            }
        }
        for (auto& scope : graph.m_statScopes)
        {
            if (scope->m_StartingPassId <= graph.m_passes.size())
            {
                compiledGraph.m_StatStartingScopes[scope->m_StartingPassId].push_back(scope->m_ScopeId);
            }
            if (scope->m_EndingPassId <= graph.m_passes.size())
            {
                compiledGraph.m_StatEndingScopes[scope->m_EndingPassId].push_back(scope->m_ScopeId);
            }
        }

        // Managed Resource Lifetime
        Vec<u32> resourceBeginUse;
        Vec<u32> resourceEndUse;
        for (u32 i = 0; i < graph.m_resources.size(); i++)
        {
            resourceBeginUse.push_back(SizeCast<ResourceNodeId>(graph.m_passes.size()));
            resourceEndUse.push_back(0);
        }
        for (u32 i = 0; i < graph.m_passes.size(); i++)
        {
            auto& pass = graph.m_passes[i];
            for (auto& resId : pass->inputResources)
            {
                resourceBeginUse[resId] = std::min(resourceBeginUse[resId], i);
                resourceEndUse[resId]   = std::max(resourceEndUse[resId], i);
            }
            for (auto& resId : pass->outputResources)
            {
                resourceEndUse[resId]   = std::max(resourceEndUse[resId], i);
                resourceBeginUse[resId] = std::min(resourceBeginUse[resId], i);
            }
        }

        for (u32 i = 0; i < graph.m_resources.size(); i++)
        {
            if (graph.m_resources[i]->isImported)
            {
                continue;
            }
            auto res = graph.m_resources[i].get();
            // iInfo("FrameGraph: Resource {} is used from {} to {}.", res->name, resourceBeginUse[i],
            // resourceEndUse[i]);
            if (resourceBeginUse[i] == graph.m_passes.size())
            {
                continue;
            }
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
            // iInfo("Compiler: Compiling pass: {}.", pass->name);
            //  Make transitions for read resources
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
                        rawResourceState[resPtr]     = RHI::RhiResourceState::Undefined;
                        rawResourceIsWriting[resPtr] = false;
                    }
                }

                RHI::RhiResourceState rawResState;
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
                    if (desiredLayout == RHI::RhiResourceState::Undefined
                        && graph.m_resourceInitState == FrameGraphResourceInitState::Uninitialized)
                    {
                        // If the layout is managed by user, then we don't need to do anything
                        // Just set the state to undefined
                        CompiledFrameGraph::ResourceBarrier aliasBarrier;
                        aliasBarrier.m_ResId                 = resId;
                        aliasBarrier.enableTransitionBarrier = true;
                        aliasBarrier.srcState                = RHI::RhiResourceState::AutoTraced; // rawResState;
                        aliasBarrier.dstState                = desiredLayout;
                        compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                    }
                    else if (desiredLayout != RHI::RhiResourceState::Undefined)
                    {
                        // Here we need to make a transition barrier
                        CompiledFrameGraph::ResourceBarrier aliasBarrier;
                        aliasBarrier.m_ResId                 = resId;
                        aliasBarrier.enableTransitionBarrier = true;
                        aliasBarrier.srcState                = RHI::RhiResourceState::AutoTraced; // rawResState;
                        aliasBarrier.dstState                = desiredLayout;
                        compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                    }
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

            // Make transitions for write resources
            for (auto& resId : pass->outputResources)
            {
                auto& res           = graph.m_resources[resId];
                auto  desiredLayout = GetDesiredOutputLayout(pass->type, res->type, res.get());

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
                        rawResourceState[resPtr]     = RHI::RhiResourceState::Undefined;
                        rawResourceIsWriting[resPtr] = true;
                    }
                }

                RHI::RhiResourceState rawResState;
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
                    if (desiredLayout == RHI::RhiResourceState::Undefined
                        && graph.m_resourceInitState == FrameGraphResourceInitState::Uninitialized)
                    {
                        // If the layout is managed by user, then we don't need to do anything
                        // Just set the state to undefined
                        CompiledFrameGraph::ResourceBarrier aliasBarrier;
                        aliasBarrier.m_ResId                 = resId;
                        aliasBarrier.enableTransitionBarrier = true;
                        aliasBarrier.srcState                = RHI::RhiResourceState::AutoTraced; // rawResState;
                        aliasBarrier.dstState                = desiredLayout;
                        compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                    }
                    else if (desiredLayout != RHI::RhiResourceState::Undefined)
                    {
                        // Here we need to make a transition barrier
                        CompiledFrameGraph::ResourceBarrier aliasBarrier;
                        aliasBarrier.m_ResId                 = resId;
                        aliasBarrier.enableTransitionBarrier = true;
                        aliasBarrier.srcState                = RHI::RhiResourceState::AutoTraced; // rawResState;
                        aliasBarrier.dstState                = desiredLayout;

                        compiledGraph.m_inputBarriers.back().push_back(aliasBarrier);
                    }
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
        return compiledGraph;
    }

    RHI::RhiResourceBarrier FrameGraphExecutor::ToRhiResBarrier(
        const CompiledFrameGraph::ResourceBarrier& barrier, const ResourceNode& res, bool& valid)
    {
        RHI::RhiResourceBarrier resBarrier;
        valid = false;
        if (barrier.enableTransitionBarrier)
        {
            resBarrier.m_type              = RHI::RhiBarrierType::Transition;
            resBarrier.m_transition.m_type = res.type == FrameGraphResourceType::ResourceBuffer
                ? RHI::RhiResourceType::Buffer
                : RHI::RhiResourceType::Texture;
            if (res.isImported)
            {
                if (res.type == FrameGraphResourceType::ResourceBuffer)
                {
                    IF_LOG_ASSERTION("FrameGraph", res.importedBuffer,
                        "FrameGraphExecutor: Imported buffer resource is null. Lifetime is corrupted.");
                    resBarrier.m_transition.m_buffer = res.importedBuffer;
                }
                else
                {
                    IF_LOG_ASSERTION("FrameGraph", res.importedTexture,
                        "FrameGraphExecutor: Imported texture resource is null. Lifetime is corrupted.");
                    resBarrier.m_transition.m_texture     = res.importedTexture;
                    resBarrier.m_transition.m_subResource = res.subResource;
                }
            }
            else
            {
                if (res.type == FrameGraphResourceType::ResourceBuffer)
                {
                    IF_LOG_ASSERTION("FrameGraph", res.selfBuffer,
                        "FrameGraphExecutor: Buffer resource is null. Lifetime is corrupted.");
                    resBarrier.m_transition.m_buffer = res.selfBuffer;
                }
                else
                {
                    IF_LOG_ASSERTION("FrameGraph", res.selfTexture,
                        "FrameGraphExecutor: Texture resource is null. Lifetime is corrupted.");
                    resBarrier.m_transition.m_texture     = res.selfTexture;
                    resBarrier.m_transition.m_subResource = res.subResource;
                }
            }

            resBarrier.m_transition.m_srcState = RHI::RhiResourceState::AutoTraced; // barrier.srcState;
            resBarrier.m_transition.m_dstState = barrier.dstState;
            valid                              = true;
        }
        else if (barrier.enableUAVBarrier)
        {
            resBarrier.m_type       = RHI::RhiBarrierType::UAVAccess;
            resBarrier.m_uav.m_type = res.type == FrameGraphResourceType::ResourceBuffer
                ? RHI::RhiResourceType::Buffer
                : RHI::RhiResourceType::Texture;
            if (res.isImported)
            {
                if (res.type == FrameGraphResourceType::ResourceBuffer)
                {
                    IF_LOG_ASSERTION("FrameGraph", res.importedBuffer,
                        "FrameGraphExecutor: Buffer resource is null. Lifetime is corrupted.");
                    resBarrier.m_uav.m_buffer = res.importedBuffer;
                }
                else
                {
                    IF_LOG_ASSERTION("FrameGraph", res.importedTexture,
                        "FrameGraphExecutor: Texture resource is null. Lifetime is corrupted.");
                    resBarrier.m_uav.m_texture = res.importedTexture;
                }
            }
            else
            {
                if (res.type == FrameGraphResourceType::ResourceBuffer)
                {
                    IF_LOG_ASSERTION("FrameGraph", res.selfBuffer,
                        "FrameGraphExecutor: Buffer resource is null. Lifetime is corrupted.");
                    resBarrier.m_uav.m_buffer = res.selfBuffer;
                }
                else
                {
                    IF_LOG_ASSERTION("FrameGraph", res.selfTexture,
                        "FrameGraphExecutor: Texture resource is null. Lifetime is corrupted.");
                    resBarrier.m_uav.m_texture = res.selfTexture;
                }
            }

            valid = true;
        }

        return resBarrier;
    }

    // Execute the compiled frame graph
    IFRIT_APIDECL void FrameGraphExecutor::ExecuteInSingleCmd(
        const RHI::RhiCommandList* cmd, const CompiledFrameGraph& compiledGraph)
    {
        auto statManager = GetActiveApplication()->GetProfileDataManager();

        cmd->BeginScope("Ifrit.RDG: Execute Render Graph");
        using namespace Ifrit::RHI;
        // Begin event scopes, top level
        int scopesActive = 0;
        for (auto& scopeName : compiledGraph.m_StartingScopes[0])
        {
            cmd->BeginScope(scopeName);
            scopesActive++;
        }
        for (u32 i = 0; i < compiledGraph.m_EndingScopes[0]; i++)
        {
            cmd->EndScope();
            scopesActive--;
        }

        // begin stat scopes
        for (auto& scopeId : compiledGraph.m_StatStartingScopes[0])
        {
            auto& statScope = *compiledGraph.m_graph->m_statScopes[scopeId];
            statManager->ReportBeginEvent(cmd, statScope.m_Name);
        }

        // end stat scopes
        for (auto& scopeId : compiledGraph.m_StatEndingScopes[0])
        {
            auto& statScope = *compiledGraph.m_graph->m_statScopes[scopeId];
            statManager->ReportEndEvent(cmd, statScope.m_Name);
        }

        for (auto& pass : compiledGraph.m_graph->m_passes)
        {
            // PreExecute
            // iInfo("FrameGraphExecutor: Executing {}", pass->name);
            for (u32 i = 0; i < pass->m_ResourceCreateRequest.size(); i++)
            {
                auto res = compiledGraph.m_graph->m_resources[pass->m_ResourceCreateRequest[i]].get();
                IF_LOG_ASSERTION("FrameGraph", !res->isImported, "Resource should not be imported.");
                // iInfo("FrameGraphExecutor: Allocating {} ({})", res->name,res->id);
                if (res->type == FrameGraphResourceType::ResourceBuffer)
                {
                    auto resAlloc = compiledGraph.m_graph->m_ResourcePool->CreateBuffer(res->bufferDesc, res->name);
                    res->m_PooledResId = resAlloc.m_PooledResId;
                    res->selfBuffer    = resAlloc.m_Buffer;
                }
                else if (res->type == FrameGraphResourceType::ResourceTexture)
                {
                    auto resAlloc = compiledGraph.m_graph->m_ResourcePool->CreateTexture(res->textureDesc, res->name);
                    res->m_PooledResId = resAlloc.m_PooledResId;
                    res->selfTexture   = resAlloc.m_Texture;
                    res->subResource   = { 0, 0, 1, 1 };
                }
            }

            // After PreExecute
            pass->OnAfterResourceAllocated(m_RhiBackend);

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
            passContext.m_FgDesc  = compiledGraph.m_graph;

            pass->FillContext(passContext);
            pass->Execute(passContext);
            cmd->GlobalMemoryBarrier();
            // PostExecute
            for (u32 i = 0; i < pass->m_ResourceReleaseRequest.size(); i++)
            {
                auto res = compiledGraph.m_graph->m_resources[pass->m_ResourceReleaseRequest[i]].get();
                IF_LOG_ASSERTION("FrameGraph", !res->isImported, "Resource should not be imported.");
                // iInfo("FrameGraphExecutor: Recycling {} ({})", res->name, res->id);
                if (res->type == FrameGraphResourceType::ResourceBuffer)
                {
                    res->selfBuffer = nullptr;
                    compiledGraph.m_graph->m_ResourcePool->ReleaseBuffer(res->m_PooledResId);
                    res->m_PooledResId = FIndexedPtr(0);
                }
                else if (res->type == FrameGraphResourceType::ResourceTexture)
                {
                    res->selfTexture = nullptr;
                    compiledGraph.m_graph->m_ResourcePool->ReleaseTexture(res->m_PooledResId);
                    res->m_PooledResId = FIndexedPtr(0);
                }
            }

            // End scopes
            for (u32 i = 0; i < compiledGraph.m_EndingScopes[pass->id + 1]; i++)
            {
                cmd->EndScope();
                scopesActive--;
            }

            // Begin event scopes
            for (auto& scopeName : compiledGraph.m_StartingScopes[pass->id + 1])
            {
                cmd->BeginScope(scopeName);
                scopesActive++;
            }

            // End stat scopes
            for (auto& scopeId : compiledGraph.m_StatEndingScopes[pass->id + 1])
            {
                auto  scopeName = compiledGraph.m_graph->m_statScopes[scopeId]->m_Name;
                auto& statScope = *compiledGraph.m_graph->m_statScopes[scopeId];
                statManager->ReportEndEvent(cmd, statScope.m_Name);
            }

            // Begin stat scopes
            for (auto& scopeId : compiledGraph.m_StatStartingScopes[pass->id + 1])
            {
                auto& statScope = *compiledGraph.m_graph->m_statScopes[scopeId];
                statManager->ReportBeginEvent(cmd, statScope.m_Name);
            }
        }
        cmd->EndScope();
        while (scopesActive > 0)
        {
            cmd->EndScope();
            scopesActive--;
        }
    }

} // namespace Ifrit::Runtime