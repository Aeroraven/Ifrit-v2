
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

#include "ifrit/runtime/common/Pch.h"

#include "ifrit/runtime/rendercore/shadercore/ShaderRegistry.h"
#include "ifrit/runtime/rendercore/framegraph/FrameGraphResourcePool.h"

namespace Ifrit::Runtime
{

    // Migration of render graph from original RHI layer.
    // Intended to making automatic layout transitions and resource management
    // easier. Resource lifetime management and reuse will be considered in the
    // future.
    // Some references from: https://zhuanlan.zhihu.com/p/147207161

    using ResourceNodeId       = u32;
    using PassNodeId           = u32;
    using FgBuffer             = RHI::RhiBuffer;
    using FgTexture            = RHI::RhiTexture;
    using FgTextureSubResource = RHI::RhiImageSubResource;

    class FrameGraphCompiler;
    class FrameGraphExecutor;
    class IFrameGraphDescRegistry;

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
        Transfer
    };

    enum class FrameGraphCompileMode
    {
        Unordered, // Planned To be Removed
        Sequential
    };

    enum class FrameGraphResourceInitState
    {
        Manual,
        Uninitialized,
    };

    struct FrameGraphPassContext
    {
        const IFrameGraphDescRegistry* m_FgDesc;
        const RHI::RhiCommandListContext* m_CmdList;
        const RHI::RhiGraphicsPass*    m_GraphicsPass;
        const RHI::RhiComputePass*     m_ComputePass;
    };

    struct IFRIT_APIDECL ResourceNode
    {
    private:
        ResourceNodeId         id;
        String                 name;
        bool                   isImported;
        FrameGraphResourceType type;

        FgBuffer*              selfBuffer;
        FgTexture*             selfTexture;
        FgBuffer*              importedBuffer;
        FgTexture*             importedTexture;

        FgTextureSubResource   subResource;
        FrameGraphBufferDesc   bufferDesc;
        FrameGraphTextureDesc  textureDesc;
        FIndexedPtr            m_PooledResId;

    public:
        friend class FrameGraphCompiler;
        friend class FrameGraphExecutor;
        friend class FrameGraphBuilder;
        friend struct PassNode;

    public:
        // These two should be removed in the future.
        ResourceNode&          SetImportedResource(FgBuffer* buffer);
        ResourceNode&          SetImportedResource(FgTexture* texture, const FgTextureSubResource& subResource);

        FrameGraphResourceType GetType() const { return type; }

        FgBuffer*              GetBuffer() const
        {
            if (isImported)
            {
                return importedBuffer;
            }
            else
            {
                if (selfBuffer == nullptr)
                {
                    IF_LOG_ERROR("FrameGraph",
                        "GetBuffer() called on buffer resource that is not created. Lifetime is corrupted.");
                    std::abort();
                }
                return selfBuffer;
            }
        }
        FgTexture* GetTexture() const
        {
            if (isImported)
            {
                return importedTexture;
            }
            else
            {
                if (selfTexture == nullptr)
                {
                    IF_LOG_ERROR("FrameGraph",
                        "GetTexture() called on texture resource that is not created. Lifetime is corrupted.");
                    std::abort();
                }
                return selfTexture;
            }
        }
        bool                  IsImported() const { return isImported; }
        FrameGraphTextureDesc GetManagedTextureDesc() const { return textureDesc; }

        RHI::RhiImageFormat   GetTextureFormat()
        {
            if (type == FrameGraphResourceType::ResourceBuffer)
            {
                IF_LOG_ERROR("FrameGraph", "FrameGraphBuilder: GetTextureFormat() called on buffer resource.");
                std::abort();
            }
            if (isImported)
            {
                return importedTexture->GetImageFormat();
            }
            else
            {
                return textureDesc.m_Format;
            }
        }

        u32 GetHeight() const
        {
            if (type == FrameGraphResourceType::ResourceBuffer)
            {
                IF_LOG_ERROR("FrameGraph", "FrameGraphBuilder: GetHeight() called on buffer resource.");
                std::abort();
            }
            if (isImported)
            {
                return importedTexture->GetHeight();
            }
            else
            {
                return textureDesc.m_Height;
            }
        }

        u32 GetWidth() const
        {
            if (type == FrameGraphResourceType::ResourceBuffer)
            {
                IF_LOG_ERROR("FrameGraph", "FrameGraphBuilder: GetWidth() called on buffer resource.");
                std::abort();
            }
            if (isImported)
            {
                return importedTexture->GetWidth();
            }
            else
            {
                return textureDesc.m_Width;
            }
        }

        u32 GetDepth() const
        {
            if (type == FrameGraphResourceType::ResourceBuffer)
            {
                IF_LOG_ERROR("FrameGraph", "FrameGraphBuilder: GetDepth() called on buffer resource.");
                std::abort();
            }
            if (isImported)
            {
                return importedTexture->GetDepth();
            }
            else
            {
                return textureDesc.m_Depth;
            }
        }

    private:
        void SetManagedResource(FrameGraphBufferDesc desc)
        {
            bufferDesc = desc;
            isImported = false;
        };
        void SetManagedResource(FrameGraphTextureDesc desc)
        {
            textureDesc = desc;
            isImported  = false;
        };
    };

    using FGTextureNode    = ResourceNode;
    using FGBufferNode     = ResourceNode;
    using FGTextureNodeRef = ResourceNode*;
    using FGBufferNodeRef  = ResourceNode*;

    struct IFRIT_APIDECL PassNode
    {
    protected:
        PassNodeId                             id;
        FrameGraphPassType                     type;
        String                                 name;
        bool                                   isImported;
        Fn<void(const FrameGraphPassContext&)> passFunction;
        Vec<ResourceNodeId>                    inputResources;
        Vec<ResourceNodeId>                    outputResources;
        Vec<ResourceNodeId>                    dependentResources;

        Vec<ResourceNodeId>                    m_ResourceCreateRequest;
        Vec<ResourceNodeId>                    m_ResourceReleaseRequest;

    public:
        friend class FrameGraphCompiler;
        friend class FrameGraphExecutor;
        friend class FrameGraphBuilder;

        PassNode& AddReadResource(const ResourceNode& res);
        PassNode& AddWriteResource(const ResourceNode& res);
        PassNode& AddReadWriteResource(const ResourceNode& res);

        // Legacy Interface, should be removed in the future.
        PassNode& AddDependentResource(const ResourceNode& res);
        PassNode& SetExecutionFunction(Fn<void(const FrameGraphPassContext&)> func);
        virtual ~PassNode() {}

    protected:
        virtual void        Execute(const FrameGraphPassContext& ctx);
        virtual void        OnAfterResourceAllocated(RHI::RhiBackend* rhiBackend) {}
        inline virtual void FillContext(FrameGraphPassContext& passContext)
        {
            passContext.m_ComputePass  = nullptr;
            passContext.m_GraphicsPass = nullptr;
        }
    };

    struct IFRIT_APIDECL ComputePassNode : public PassNode, NonCopyable
    {
    protected:
        Owner<RHI::RhiComputePass> m_pass;

    protected:
        virtual void Execute(const FrameGraphPassContext& ctx) override;

    public:
        ComputePassNode(Owner<RHI::RhiComputePass>&& pass);
        virtual ~ComputePassNode() {}
        inline RHI::RhiComputePass* GetPass() { return m_pass.get(); }
        inline virtual void         FillContext(FrameGraphPassContext& passContext)
        {
            passContext.m_ComputePass  = m_pass.get();
            passContext.m_GraphicsPass = nullptr;
        }
        friend class FrameGraphBuilder;
    };

    struct IFRIT_APIDECL GraphicsPassNode : public PassNode, NonCopyable
    {
        using LoadOp = RHI::RhiRenderTargetLoadOp;

    protected:
        Owner<RHI::RhiGraphicsPass>         m_pass;

        Vec<ResourceNode*>                  m_RenderTarget;
        Vec<LoadOp>                         m_ColorLoadOp;
        Vec<Vector4f>                       m_ColorClearValue;
        ResourceNode*                       m_DepthTarget = nullptr;
        LoadOp                              m_DepthLoadOp;
        f32                                 m_DepthClearValue;

        Vec<Ref<RHI::RhiColorAttachment>>   m_RhiColorRTs;
        Ref<RHI::RhiDepthStencilAttachment> m_RhiDepthRT;
        Ref<RHI::RhiRenderTargets>          m_RhiRTs;
        RHI::RhiScissor                     m_Scissor    = { 0, 0, 0, 0 };
        bool                                m_RTComposed = false;

    protected:
        virtual void Execute(const FrameGraphPassContext& ctx) override;
        void         ComposeRenderTargets(RHI::RhiBackend* rhiBackend);
        virtual void OnAfterResourceAllocated(RHI::RhiBackend* rhiBackend) override
        {
            ComposeRenderTargets(rhiBackend);
        }

    public:
        GraphicsPassNode(Owner<RHI::RhiGraphicsPass>&& pass);
        inline RHI::RhiGraphicsPass* GetPass() { return m_pass.get(); }
        inline virtual void          FillContext(FrameGraphPassContext& passContext)
        {
            passContext.m_ComputePass  = nullptr;
            passContext.m_GraphicsPass = m_pass.get();
        }

        GraphicsPassNode& AddRenderTarget(
            ResourceNode& res, LoadOp loadOp = LoadOp::Clear, Vector4f clearValue = { 0, 0, 0, 0 });
        GraphicsPassNode& AddDepthTarget(ResourceNode& res, LoadOp loadOp = LoadOp::Clear, f32 clearValue = 1.0f);

        friend class FrameGraphBuilder;
    };

    struct FrameGraphScope
    {
        String m_Name;
        u32    m_ScopeId;
        u32    m_StartingPassId = ~0u;
        u32    m_EndingPassId   = ~0u;
    };
    struct FrameGraphStatScope
    {
        String m_Name;
        u32    m_ScopeId;
        u32    m_StartingPassId = ~0u;
        u32    m_EndingPassId   = ~0u;
    };

    class IFRIT_APIDECL IFrameGraphDescRegistry
    {
    public:
        virtual RHI::RhiUAVDesc GetUAV(const ResourceNode& res) const = 0;
        virtual RHI::RhiSRVDesc GetSRV(const ResourceNode& res) const = 0;
        virtual RHI::RhiCBVDesc GetCBV(const ResourceNode& res) const = 0;
    };

    class IFRIT_APIDECL FrameGraphBuilder : public IFrameGraphDescRegistry, public NonCopyable
    {
    private:
        Vec<Owner<ResourceNode>>        m_resources;
        Vec<Owner<PassNode>>            m_passes;
        Vec<Owner<FrameGraphScope>>     m_scopes;
        Vec<Owner<FrameGraphStatScope>> m_statScopes;
        FrameGraphCompileMode           m_compileMode       = FrameGraphCompileMode::Sequential;
        FrameGraphResourceInitState     m_resourceInitState = FrameGraphResourceInitState::Manual;
        ShaderRegistry*                 m_ShaderRegistry    = nullptr;
        RHI::RhiBackend*                m_Rhi               = nullptr;

        FrameGraphResourcePool*         m_ResourcePool = nullptr;

    public:
        FrameGraphBuilder(ShaderRegistry* shaderRegistry, RHI::RhiBackend* rhi, FrameGraphResourcePool* resourcePool)
            : m_ShaderRegistry(shaderRegistry), m_Rhi(rhi), m_ResourcePool(resourcePool)
        {
        }
        ~FrameGraphBuilder();

        ResourceNode&     AddResource(const String& name);
        PassNode&         AddPass(const String& name, FrameGraphPassType type);
        void              SetResourceInitState(FrameGraphResourceInitState state) { m_resourceInitState = state; }

        ComputePassNode&  AddComputePass(const String& name, const ShaderVariantDesc& shader, u32 pushConsts);
        GraphicsPassNode& AddGraphicsPass(const String& name, const ShaderVariantDesc& vs, const ShaderVariantDesc& fs,
            u32 pushConsts, RHI::RhiRasterizerTopology topology = RHI::RhiRasterizerTopology::TriangleList);
        GraphicsPassNode& AddMeshGraphicsPass(
            const String& name, const ShaderVariantDesc& ms, const ShaderVariantDesc& fs, u32 pushConsts);

        ResourceNode& DeclareTexture(const String& name, const FrameGraphTextureDesc& desc);
        ResourceNode& DeclareBuffer(const String& name, const FrameGraphBufferDesc& desc);

        ResourceNode& ImportTexture(
            const String& name, FgTexture* texture, const FgTextureSubResource& subResource = { 0, 0, 1, 1 });
        ResourceNode&           ImportBuffer(const String& name, FgBuffer* buffer);

        RHI::RhiUAVDesc         GetUAV(const ResourceNode& res) const override;
        RHI::RhiSRVDesc         GetSRV(const ResourceNode& res) const override;
        RHI::RhiCBVDesc         GetCBV(const ResourceNode& res) const override;

        inline RHI::RhiBackend* GetRhi() const { return m_Rhi; }
        inline ShaderRegistry*  GetShaderRegistry() const { return m_ShaderRegistry; }

        FrameGraphScope&        AddScopeBegin(const String& name);
        void                    AddScopeEnd(const FrameGraphScope& scope);

        FrameGraphStatScope&    AddStatScopeBegin(const String& name);
        void                    AddStatScopeEnd(const FrameGraphStatScope& scope);

        friend class FrameGraphCompiler;
        friend class FrameGraphExecutor;
    };

    struct CompiledFrameGraph
    {
        struct ResourceBarrier
        {
            u32                   m_ResId                 = ~0u;
            bool                  enableUAVBarrier        = false;
            bool                  enableTransitionBarrier = false;
            RHI::RhiResourceState srcState;
            RHI::RhiResourceState dstState = RHI::RhiResourceState::Undefined;
        };
        FrameGraphResourceInitState m_resourceInitState  = FrameGraphResourceInitState::Manual;
        const FrameGraphBuilder*    m_graph              = nullptr;
        Vec<Vec<ResourceBarrier>>   m_inputBarriers      = {};
        Vec<Vec<String>>            m_StartingScopes     = {};
        Vec<u32>                    m_EndingScopes       = {};
        Vec<Vec<u32>>               m_StatStartingScopes = {};
        Vec<Vec<u32>>               m_StatEndingScopes   = {};
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
        FrameGraphExecutor(RHI::RhiBackend* rhiBackend) : m_RhiBackend(rhiBackend) {}
        void ExecuteInSingleCmd(const RHI::RhiCommandListContext* cmd, const CompiledFrameGraph& compiledGraph);

    private:
        RHI::RhiBackend*        m_RhiBackend = nullptr;
        RHI::RhiResourceBarrier ToRhiResBarrier(
            const CompiledFrameGraph::ResourceBarrier& barrier, const ResourceNode& res, bool& valid);
    };

} // namespace Ifrit::Runtime