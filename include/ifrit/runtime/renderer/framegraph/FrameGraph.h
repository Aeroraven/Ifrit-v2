
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
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/constfunc/ConstFunc.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <functional>
#include <string>
#include <vector>

#include "ifrit/runtime/material/ShaderRegistry.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphResourcePool.h"

namespace Ifrit::Runtime
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
        const Graphics::Rhi::RhiCommandList*  m_CmdList;
        const Graphics::Rhi::RhiGraphicsPass* m_GraphicsPass;
        const Graphics::Rhi::RhiComputePass*  m_ComputePass;
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
        RIndexedPtr            m_PooledResId;

    public:
        friend class FrameGraphCompiler;
        friend class FrameGraphExecutor;
        friend class FrameGraphBuilder;
        friend struct PassNode;

    public:
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
                    iError(
                        "FrameGraphBuilder: GetBuffer() called on buffer resource that is not created. Lifetime is corrupted.");
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
                    iError(
                        "FrameGraphBuilder: GetTexture() called on texture resource that is not created. Lifetime is corrupted.");
                    std::abort();
                }
                return selfTexture;
            }
        }
        bool                          IsImported() const { return isImported; }
        FrameGraphTextureDesc         GetManagedTextureDesc() const { return textureDesc; }

        Graphics::Rhi::RhiImageFormat GetTextureFormat()
        {
            if (type == FrameGraphResourceType::ResourceBuffer)
            {
                iError("FrameGraphBuilder: GetTextureFormat() called on buffer resource.");
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

    protected:
        virtual void        Execute(const FrameGraphPassContext& ctx);
        virtual void        OnAfterResourceAllocated(Graphics::Rhi::RhiBackend* rhiBackend) {}
        inline virtual void FillContext(FrameGraphPassContext& passContext)
        {
            passContext.m_ComputePass  = nullptr;
            passContext.m_GraphicsPass = nullptr;
        }
    };

    struct IFRIT_APIDECL ComputePassNode : public PassNode, NonCopyable
    {
    protected:
        Uref<Graphics::Rhi::RhiComputePass> m_pass;

    protected:
        ComputePassNode(Uref<Graphics::Rhi::RhiComputePass>&& pass);
        virtual void Execute(const FrameGraphPassContext& ctx) override;

    public:
        inline Graphics::Rhi::RhiComputePass* GetPass() { return m_pass.get(); }
        inline virtual void                   FillContext(FrameGraphPassContext& passContext)
        {
            passContext.m_ComputePass  = m_pass.get();
            passContext.m_GraphicsPass = nullptr;
        }
        friend class FrameGraphBuilder;
    };

    struct IFRIT_APIDECL GraphicsPassNode : public PassNode, NonCopyable
    {
        using LoadOp = Graphics::Rhi::RhiRenderTargetLoadOp;

    protected:
        Uref<Graphics::Rhi::RhiGraphicsPass>          m_pass;

        Vec<ResourceNode*>                            m_RenderTarget;
        Vec<LoadOp>                                   m_ColorLoadOp;
        Vec<Vector4f>                                 m_ColorClearValue;
        ResourceNode*                                 m_DepthTarget = nullptr;
        LoadOp                                        m_DepthLoadOp;
        f32                                           m_DepthClearValue;

        Vec<Ref<Graphics::Rhi::RhiColorAttachment>>   m_RhiColorRTs;
        Ref<Graphics::Rhi::RhiDepthStencilAttachment> m_RhiDepthRT;
        Ref<Graphics::Rhi::RhiRenderTargets>          m_RhiRTs;
        Graphics::Rhi::RhiScissor                     m_Scissor    = { 0, 0, 0, 0 };
        bool                                          m_RTComposed = false;

    protected:
        GraphicsPassNode(Uref<Graphics::Rhi::RhiGraphicsPass>&& pass);
        virtual void Execute(const FrameGraphPassContext& ctx) override;
        void         ComposeRenderTargets(Graphics::Rhi::RhiBackend* rhiBackend);
        virtual void OnAfterResourceAllocated(Graphics::Rhi::RhiBackend* rhiBackend) override
        {
            ComposeRenderTargets(rhiBackend);
        }

    public:
        inline Graphics::Rhi::RhiGraphicsPass* GetPass() { return m_pass.get(); }
        inline virtual void                    FillContext(FrameGraphPassContext& passContext)
        {
            passContext.m_ComputePass  = nullptr;
            passContext.m_GraphicsPass = m_pass.get();
        }

        GraphicsPassNode& AddRenderTarget(
            ResourceNode& res, LoadOp loadOp = LoadOp::Clear, Vector4f clearValue = { 0, 0, 0, 0 });
        GraphicsPassNode& AddDepthTarget(ResourceNode& res, LoadOp loadOp = LoadOp::Clear, f32 clearValue = 1.0f);

        friend class FrameGraphBuilder;
    };

    class IFRIT_APIDECL FrameGraphBuilder
    {
    private:
        Vec<ResourceNode*>          m_resources;
        Vec<PassNode*>              m_passes;
        FrameGraphCompileMode       m_compileMode       = FrameGraphCompileMode::Sequential;
        FrameGraphResourceInitState m_resourceInitState = FrameGraphResourceInitState::Manual;
        ShaderRegistry*             m_ShaderRegistry    = nullptr;
        Graphics::Rhi::RhiBackend*  m_Rhi               = nullptr;

        FrameGraphResourcePool*     m_ResourcePool = nullptr;

    public:
        FrameGraphBuilder(
            ShaderRegistry* shaderRegistry, Graphics::Rhi::RhiBackend* rhi, FrameGraphResourcePool* resourcePool)
            : m_ShaderRegistry(shaderRegistry), m_Rhi(rhi), m_ResourcePool(resourcePool)
        {
        }
        ~FrameGraphBuilder();

        ResourceNode&     AddResource(const String& name);
        PassNode&         AddPass(const String& name, FrameGraphPassType type);
        void              SetResourceInitState(FrameGraphResourceInitState state) { m_resourceInitState = state; }

        ComputePassNode&  AddComputePass(const String& name, const String& shader, u32 pushConsts);
        GraphicsPassNode& AddGraphicsPass(const String& name, const String& vs, const String& fs, u32 pushConsts);
        GraphicsPassNode& AddMeshGraphicsPass(const String& name, const String& ms, const String& fs, u32 pushConsts);

        ResourceNode&     DeclareTexture(const String& name, const FrameGraphTextureDesc& desc);
        ResourceNode&     DeclareBuffer(const String& name, const FrameGraphBufferDesc& desc);

        Graphics::Rhi::RhiUAVDesc         GetUAV(const ResourceNode& res);
        Graphics::Rhi::RhiSRVDesc         GetSRV(const ResourceNode& res);

        inline Graphics::Rhi::RhiBackend* GetRhi() const { return m_Rhi; }
        inline ShaderRegistry*            GetShaderRegistry() const { return m_ShaderRegistry; }

        friend class FrameGraphCompiler;
        friend class FrameGraphExecutor;
    };

    struct CompiledFrameGraph
    {
        struct ResourceBarrier
        {
            u32                             m_ResId                 = ~0u;
            bool                            enableUAVBarrier        = false;
            bool                            enableTransitionBarrier = false;
            Graphics::Rhi::RhiResourceState srcState;
            Graphics::Rhi::RhiResourceState dstState = Graphics::Rhi::RhiResourceState::Undefined;
        };
        FrameGraphResourceInitState m_resourceInitState = FrameGraphResourceInitState::Manual;
        const FrameGraphBuilder*    m_graph             = nullptr;
        Vec<Vec<ResourceBarrier>>   m_inputBarriers     = {};
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
        FrameGraphExecutor(Graphics::Rhi::RhiBackend* rhiBackend) : m_RhiBackend(rhiBackend) {}
        void ExecuteInSingleCmd(const Graphics::Rhi::RhiCommandList* cmd, const CompiledFrameGraph& compiledGraph);

    private:
        Graphics::Rhi::RhiBackend*        m_RhiBackend = nullptr;
        Graphics::Rhi::RhiResourceBarrier ToRhiResBarrier(
            const CompiledFrameGraph::ResourceBarrier& barrier, const ResourceNode& res, bool& valid);
    };

} // namespace Ifrit::Runtime