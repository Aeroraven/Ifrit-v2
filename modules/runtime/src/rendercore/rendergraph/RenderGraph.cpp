#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"
#include "ifrit/core/console/ConsoleObject.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/LogUtils.h"
#include "ifrit/core/algo/BuddyAllocator.h"
#include "ifrit/core/typing/EnumUtils.h"

#include "ifrit/core/base/containers/Maps.h"
#include "ifrit/core/base/containers/Queue.h"

#include "ifrit/core/algo/Graph.h"
#include "ifrit/core/algo/StlStringUtils.h"
#include "ifrit/core/file/FileOps.h"

#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.GlobalVars.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Utils.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Resources.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Passes.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Context.h"

namespace Ifrit::Runtime::RenderCore::RDG
{
    // ===== RDG Graphs =====
    struct RDGGraphContext;

    struct RDGGraphBuilderInternal
    {
        RDGGraphBuilderArgs mArgs;
        RDGGraphContext     mCtx;
    };

    RDGGraphBuilder::RDGGraphBuilder(const RDGGraphBuilderArgs& args) : mData(new RDGGraphBuilderInternal())
    {
        mData->mCtx.mArgs     = args;
        mData->mCtx.mSetupCtx = MakeOwner<RDGSetupContext>(&mData->mCtx);
    }

    RDGGraphBuilder::~RDGGraphBuilder()
    {
        delete mData;
        mData = nullptr;
    }

    RDGTextureHandle RDGGraphBuilder::DeclareTexture(const String& name, const RDGTextureDesc& desc)
    {
        auto texture = MakeOwner<RDGTextureResource>(this, name, ERDGResourceFlag::None, desc);
        mData->mCtx.mResources.push_back(nullptr);
        auto resourceId                    = static_cast<u32>(mData->mCtx.mResources.size() - 1);
        mData->mCtx.mResources[resourceId] = std::move(texture);

        RDGTextureHandle handle(*this, resourceId);
        return handle;
    }
    RDGBufferHandle RDGGraphBuilder::DeclareBuffer(const String& name, const RDGBufferDesc& desc)
    {
        auto buffer = MakeOwner<RDGBufferResource>(this, name, ERDGResourceFlag::None, desc);
        mData->mCtx.mResources.push_back(nullptr);
        auto resourceId                    = static_cast<u32>(mData->mCtx.mResources.size() - 1);
        mData->mCtx.mResources[resourceId] = std::move(buffer);

        RDGBufferHandle handle(*this, resourceId);
        return handle;
    }
    RDGTextureHandle RDGGraphBuilder::ImportTexture(RHI::RhiTextureRef texture)
    {
        auto textureResource = MakeOwner<RDGTextureResource>(this, "texture", ERDGResourceFlag::Imported, texture);
        mData->mCtx.mResources.push_back(nullptr);
        auto resourceId                    = static_cast<u32>(mData->mCtx.mResources.size() - 1);
        mData->mCtx.mResources[resourceId] = std::move(textureResource);
        RDGTextureHandle handle(*this, resourceId);
        return handle;
    }
    RDGBufferHandle RDGGraphBuilder::ImportBuffer(RHI::RhiBufferRef buffer)
    {
        auto bufferResource = MakeOwner<RDGBufferResource>(this, "buffer", ERDGResourceFlag::Imported, buffer);
        mData->mCtx.mResources.push_back(nullptr);
        auto resourceId                    = static_cast<u32>(mData->mCtx.mResources.size() - 1);
        mData->mCtx.mResources[resourceId] = std::move(bufferResource);
        RDGBufferHandle handle(*this, resourceId);
        return handle;
    }
    void RDGGraphBuilder::Compile() { mData->mCtx.Compile(); }

    u32  RDGGraphBuilder::PreAllocatePass()
    {
        auto index = static_cast<u32>(mData->mCtx.mPasses.size());
        mData->mCtx.mPasses.emplace_back();
        return index;
    }
    void RDGGraphBuilder::ActivatePassInSetupContext(u32 index)
    {
        RDG_ASSERTION(index < mData->mCtx.mPasses.size(), "Pass index out of range in RDGGraphBuilder::ActivatePass");
        auto& pass = mData->mCtx.mPasses[index];
        RDG_NOTNULL(pass, "Pass is null in RDGGraphBuilder::ActivatePass");
        auto  passType = pass->GetType();
        auto& setupCtx = static_cast<RDGSetupContext&>(GetSetupContext());
        setupCtx.SetCurrentPass(index);
    }
    RDGPassData& RDGGraphBuilder::GetPassData(u32 index)
    {
        RDG_ASSERTION(index < mData->mCtx.mPasses.size(), "Pass index out of range in RDGGraphBuilder::GetPassData");
        auto& pass = mData->mCtx.mPasses[index];
        RDG_NOTNULL(pass, "Pass is null in RDGGraphBuilder::GetPassData");
        return pass->GetPassData();
    }
    IRDGGraphBuilderSetupContext&  RDGGraphBuilder::GetSetupContext() { return *mData->mCtx.mSetupCtx; }
    RDGGraphBuilderExecuteContext& RDGGraphBuilder::GetExecuteContext()
    {
        // TODO
        RDG_NOT_IMPLEMENTED();
    }

    RDGPassHandle RDGGraphBuilder::AddPassInternal(u32 idx, const String& name, TSinkArg<RDGPassData> passData,
        ERDGPassType type, TSinkArg<Fn<void()>> fnSetup, TSinkArg<Fn<void()>> fnExecute)
    {
        auto pass =
            MakeOwner<RDGPass>(idx, this, name, type, std::move(fnSetup), std::move(fnExecute), std::move(passData));
        mData->mCtx.mPasses[idx] = std::move(pass);

        ActivatePassInSetupContext(idx);
        mData->mCtx.mPasses[idx]->ExecuteSetup();
        RDGPassHandle handle(*this, idx);
        return handle;
    }

    void RDGGraphBuilder::DumpDebugFile(const String& path, ERDGDebugVisualizationMode mode)
    {
        if (mode == ERDGDebugVisualizationMode::PassDAG)
        {
            mData->mCtx.Visualize_DumpCompiledDOTGraph(path);
        }
        else if (mode == ERDGDebugVisualizationMode::PassDAGWithResources)
        {
            mData->mCtx.Visualize_DumpCompiledDOTGraphWithResources(path);
        }
        else if (mode == ERDGDebugVisualizationMode::PhysicalResourceAlloc)
        {
            mData->mCtx.Visualize_DumpPhysicalResourcesAllocation();
        }
        else
        {
            RDG_LOG_ERROR("Unsupported RDGDebugVisualizationMode in RDGGraphBuilder::DumpDebugFile");
        }
    }

    // ===== RDG Setup Context Impls =====

    void RDGSetupContext::AddFullResourceUsageToPass(u32 resourceIdx, ERDGResourceAccess access,
        ERDGReadWriteMode rwMode, const TOptional<RHI::RhiClearColorValue>& clearColor,
        const TOptional<RHI::RhiClearDepthStencilValue>& clearDepthStencil, RHI::ERhiRenderTargetLoadOp rtLoadOp)

    {
        RDG_ASSERTION(mCurrentPassId != ~0u, "No current pass set in RDGSetupContext::AddFullResourceUsageToPass");
        RDG_ASSERTION(resourceIdx < mCtx->mResources.size(),
            "Resource index out of range in RDGSetupContext::AddFullResourceUsageToPass");
        auto& resource = mCtx->mResources[resourceIdx];
        RDG_NOTNULL(resource, "Resource is null in RDGSetupContext::AddFullResourceUsageToPass");

        RDGPassResourceUsage usage;
        usage.mResourceIndex         = resourceIdx;
        usage.mOverallAccess.mAccess = access;
        usage.mOverallAccess.mRWMode = rwMode;

        if (OptionalNotEmpty(clearColor))
        {
            usage.mClearColor = *clearColor;
        }
        if (OptionalNotEmpty(clearDepthStencil))
        {
            usage.mClearDepthStencil = *clearDepthStencil;
        }
        usage.mRTLoadOp = rtLoadOp;

        auto& pass = mCtx->mPasses[mCurrentPassId];
        RDG_NOTNULL(pass, "Pass is null in RDGSetupContext::AddFullResourceUsageToPass");
        pass->AddResourceUsage(usage);
    }

    IRDGAccess_ResourceView* RDGSetupContext::CreateSRV(
        RDGTextureHandle handle, const TOptional<RHI::RhiImageSubResource>& subRes)
    {
        RDG_ASSERTION(!OptionalNotEmpty(subRes), "Subresource view is not supported yet in RDGSetupContext::CreateSRV");
        auto  resourceId  = handle.mIndex;
        auto& resource    = mCtx->mResources[resourceId];
        auto  texResource = ForcedCheckedCast<RDGTextureResource>(resource.get());
        AddFullResourceUsageToPass(
            resourceId, ERDGResourceAccessFlag::SRVRead, ERDGReadWriteModeFlag::Read, NullOpt, NullOpt);
        return texResource->GetOrCreateSRV(subRes);
    }
    IRDGAccess_ResourceView* RDGSetupContext::CreateSRV(RDGBufferHandle handle)
    {
        auto  resourceId  = handle.mIndex;
        auto& resource    = mCtx->mResources[resourceId];
        auto  bufResource = ForcedCheckedCast<RDGBufferResource>(resource.get());
        AddFullResourceUsageToPass(
            resourceId, ERDGResourceAccessFlag::SRVRead, ERDGReadWriteModeFlag::Read, NullOpt, NullOpt);
        return bufResource->GetSRV();
    }
    IRDGAccess_ResourceView* RDGSetupContext::CreateUAV(
        RDGTextureHandle handle, const TOptional<RHI::RhiImageSubResource>& subRes, ERDGReadWriteMode mode)
    {
        RDG_ASSERTION(!OptionalNotEmpty(subRes), "Subresource view is not supported yet in RDGSetupContext::CreateUAV");
        auto               resourceId  = handle.mIndex;
        auto&              resource    = mCtx->mResources[resourceId];
        auto               texResource = ForcedCheckedCast<RDGTextureResource>(resource.get());

        ERDGResourceAccess access = 0;
        if (HasFlagBit(mode, ERDGReadWriteModeFlag::Read))
            access = SetFlagBit(access, ERDGResourceAccessFlag::UAVRead);
        if (HasFlagBit(mode, ERDGReadWriteModeFlag::Write))
            access = SetFlagBit(access, ERDGResourceAccessFlag::UAVWrite);

        AddFullResourceUsageToPass(resourceId, access, mode, NullOpt, NullOpt);
        return texResource->GetOrCreateUAV(subRes);
    }
    IRDGAccess_ResourceView* RDGSetupContext::CreateUAV(RDGBufferHandle handle, ERDGReadWriteMode mode)
    {
        auto               resourceId  = handle.mIndex;
        auto&              resource    = mCtx->mResources[resourceId];
        auto               bufResource = ForcedCheckedCast<RDGBufferResource>(resource.get());
        ERDGResourceAccess access      = 0;
        if (HasFlagBit(mode, ERDGReadWriteModeFlag::Read))
            access = SetFlagBit(access, ERDGResourceAccessFlag::UAVRead);
        if (HasFlagBit(mode, ERDGReadWriteModeFlag::Write))
            access = SetFlagBit(access, ERDGResourceAccessFlag::UAVWrite);
        AddFullResourceUsageToPass(resourceId, access, mode, NullOpt, NullOpt);
        return bufResource->GetUAV();
    }
    IRDGAccess_Buffer* RDGSetupContext::AsIndirectArg(RDGBufferHandle handle)
    {
        auto  resourceId  = handle.mIndex;
        auto& resource    = mCtx->mResources[resourceId];
        auto  bufResource = ForcedCheckedCast<RDGBufferResource>(resource.get());
        AddFullResourceUsageToPass(
            resourceId, ERDGResourceAccessFlag::IndirectArgRead, ERDGReadWriteModeFlag::Read, NullOpt, NullOpt);
        return bufResource->GetBufferAccess();
    }

    IRDGAccess_Texture* RDGSetupContext::AsRenderTarget(RDGTextureHandle handle,
        TOptional<RHI::RhiImageSubResource> clearValue, RHI::ERhiRenderTargetLoadOp loadOp,
        TOptional<RHI::RhiClearColorValue> clearColor)
    {
        auto  resourceId  = handle.mIndex;
        auto& resource    = mCtx->mResources[resourceId];
        auto  texResource = ForcedCheckedCast<RDGTextureResource>(resource.get());
        AddFullResourceUsageToPass(resourceId, ERDGResourceAccessFlag::RenderTarget, ERDGReadWriteModeFlag::ReadWrite,
            clearColor, NullOpt, loadOp);
        return texResource->GetTextureAccess();
    }
    IRDGAccess_Texture* RDGSetupContext::AsDepthStencil(RDGTextureHandle handle,
        TOptional<RHI::RhiImageSubResource> clearValue, RHI::ERhiRenderTargetLoadOp loadOp, TOptional<f32> clearDepth,
        TOptional<u32> clearStencil)
    {
        auto                           resourceId  = handle.mIndex;
        auto&                          resource    = mCtx->mResources[resourceId];
        auto                           texResource = ForcedCheckedCast<RDGTextureResource>(resource.get());
        RHI::RhiClearDepthStencilValue clearDepthStencil;
        if (OptionalNotEmpty(clearValue))
        {
            clearDepthStencil.m_Depth   = clearDepth.value_or(1.0f);
            clearDepthStencil.m_Stencil = clearStencil.value_or(0);
        }
        else
        {
            clearDepthStencil.m_Depth   = 1.0f;
            clearDepthStencil.m_Stencil = 0;
        }

        AddFullResourceUsageToPass(resourceId, ERDGResourceAccessFlag::DepthStencil, ERDGReadWriteModeFlag::ReadWrite,
            NullOpt, clearDepthStencil, loadOp);
        return texResource->GetTextureAccess();
    }

} // namespace Ifrit::Runtime::RenderCore::RDG

#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.ContextImpl.inl"