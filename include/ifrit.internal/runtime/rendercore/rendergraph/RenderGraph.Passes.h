#pragma once
#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"

#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.GlobalVars.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Utils.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/LogUtils.h"

namespace Ifrit::Runtime::RenderCore::RDG
{
    using RDGPassData = Ifrit::Reflection::Object;

    // ===== RDG Passes =====

    struct RDGPassIdentifier
    {
        ERDGQueueType mQueueType = ERDGQueueType::Graphics;
        u32           mIndex     = ~0u;
    };

    struct RDGPassResourceUsage
    {
        struct AccessPattern
        {
            ERDGResourceAccess mAccess = 0;
            ERDGReadWriteMode  mRWMode = ERDGReadWriteModeFlag::None;
        };
        u32                                         mResourceIndex = ~0u;
        AccessPattern                               mOverallAccess;
        THashMap<RDGSubresourcePair, AccessPattern> mSubresourceAccesses;
        bool                                        mDetailedSubresourceTracking = false;

        // Additional
        RHI::ERhiRenderTargetLoadOp                 mRTLoadOp = RHI::ERhiRenderTargetLoadOp::DontCare;
        RHI::RhiClearColorValue                     mClearColor;
        RHI::RhiClearDepthStencilValue              mClearDepthStencil;

        void                                        MergeUsageFrom(const RDGPassResourceUsage& other);
    };

    struct RDGDownstreamPassDependency
    {
        u32      mDownstreamPassIndex = ~0u;
        Vec<u32> mResourceDependencies;
    };

    struct RDGUpstreamPassDependency
    {
        u32      mUpstreamPassIndex = ~0u;
        Vec<u32> mResourceDependencies;
    };

    class RDGPass
    {
    public:
        RDGPass(u32 id, RDGGraphBuilder* builder, const String& name, ERDGPassType type, TSinkArg<Fn<void()>> setup,
            TSinkArg<Fn<void()>> execute, TSinkArg<RDGPassData> passData) noexcept
            : mId(id)
            , mBuilder(builder)
            , mName(name)
            , mType(type)
            , mSetupFn(std::move(setup))
            , mExecuteFn(std::move(execute))
            , mPassData(std::move(passData))
        {
        }
        virtual ~RDGPass() = default;

        IF_FORCEINLINE ERDGPassType             GetType() const noexcept { return mType; }
        IF_FORCEINLINE const RDGPassIdentifier& GetIdentifier() const noexcept { return mIdentifier; }
        IF_FORCEINLINE String                   GetName() const { return mName; }
        IF_FORCEINLINE u32                      GetId() const noexcept { return mId; }
        IF_FORCEINLINE const Vec<RDGPassResourceUsage>& GetResourceUsages() const noexcept { return mResourceUsages; }
        IF_FORCEINLINE RDGPassData&                     GetPassData() noexcept { return mPassData; }
        IF_FORCEINLINE bool                             IsEnabled() const noexcept { return mEnabled; }

        IF_FORCEINLINE void SetIdentifier(const RDGPassIdentifier& id) noexcept { mIdentifier = id; }
        IF_FORCEINLINE void SetEnabled(bool enabled) noexcept { mEnabled = enabled; }

        IF_FORCEINLINE void ExecuteSetup() { mSetupFn(); }

        void                AddResourceUsage(const RDGPassResourceUsage& usage)
        {

            auto inId = usage.mResourceIndex;
            for (auto& existUsage : mResourceUsages)
            {
                if (existUsage.mResourceIndex == inId)
                {
                    existUsage.MergeUsageFrom(usage);
                    return;
                }
            }
            mResourceUsages.push_back(usage);
        }

    private:
        String                           mName;
        ERDGPassType                     mType = ERDGPassType::Invalid;
        Fn<void()>                       mSetupFn;
        Fn<void()>                       mExecuteFn;
        RDGGraphBuilder*                 mBuilder = nullptr;

        RDGPassIdentifier                mIdentifier;
        u32                              mId;
        Vec<RDGPassResourceUsage>        mResourceUsages;
        Vec<RDGDownstreamPassDependency> mDownstreamPassDeps;
        Vec<RDGUpstreamPassDependency>   mUpstreamPassDeps;

        Vec<u32>                         mResourceCreationRequests;
        Vec<u32>                         mResourceReleaseRequests;
        bool                             mEnabled = false;

        RDGPassData                      mPassData;
    };

    // ===== Merge Pass Impls =====
    void RDGPassResourceUsage::MergeUsageFrom(const RDGPassResourceUsage& other)
    {
        // Merge access pattern
        for (const auto& [subres, access] : other.mSubresourceAccesses)
        {
            auto it = mSubresourceAccesses.find(subres);
            if (it != mSubresourceAccesses.end())
            {
                it->second.mAccess = static_cast<ERDGResourceAccess>(it->second.mAccess | access.mAccess);
                it->second.mRWMode = static_cast<ERDGReadWriteMode>(it->second.mRWMode | access.mRWMode);
            }
            else
            {
                mSubresourceAccesses[subres] = access;
            }
        }
        mOverallAccess.mAccess = static_cast<ERDGResourceAccess>(mOverallAccess.mAccess | other.mOverallAccess.mAccess);
        mOverallAccess.mRWMode = static_cast<ERDGReadWriteMode>(mOverallAccess.mRWMode | other.mOverallAccess.mRWMode);

        bool sourceHasClearColorData = HasFlagBit(mOverallAccess.mAccess, ERDGResourceAccessFlag::RenderTarget);
        bool destHasClearColorData   = HasFlagBit(other.mOverallAccess.mAccess, ERDGResourceAccessFlag::RenderTarget);
        bool sourceHasClearDepthData = HasFlagBit(mOverallAccess.mAccess, ERDGResourceAccessFlag::DepthStencil);
        bool destHasClearDepthData   = HasFlagBit(other.mOverallAccess.mAccess, ERDGResourceAccessFlag::DepthStencil);
        if (sourceHasClearColorData && destHasClearColorData)
        {
            if (mClearColor != other.mClearColor || mRTLoadOp != other.mRTLoadOp)
            {
                RDG_LOG_WARNING(
                    "Merging two different clear color values in RDGPassResourceUsage::MergeUsageFrom, overwriting.");
            }
        }
        if (destHasClearColorData)
        {
            mRTLoadOp   = other.mRTLoadOp;
            mClearColor = other.mClearColor;
        }
        if (sourceHasClearDepthData && destHasClearDepthData)
        {
            if (mClearDepthStencil.m_Depth != other.mClearDepthStencil.m_Depth
                || mClearDepthStencil.m_Stencil != other.mClearDepthStencil.m_Stencil)
            {
                RDG_LOG_WARNING(
                    "Merging two different clear depth/stencil values in RDGPassResourceUsage::MergeUsageFrom, overwriting.");
            }
        }
        if (destHasClearDepthData)
        {
            mClearDepthStencil = other.mClearDepthStencil;
        }

        // Validate invalid combinations
        bool isRTV_DSV = HasFlagBit(mOverallAccess.mAccess, ERDGResourceAccessFlag::RenderTarget)
            || HasFlagBit(mOverallAccess.mAccess, ERDGResourceAccessFlag::DepthStencil);
        bool isUAV_SRV = HasFlagBit(mOverallAccess.mAccess, ERDGResourceAccessFlag::UAVRead)
            || HasFlagBit(mOverallAccess.mAccess, ERDGResourceAccessFlag::UAVWrite)
            || HasFlagBit(mOverallAccess.mAccess, ERDGResourceAccessFlag::SRVRead);

        if (isRTV_DSV && isUAV_SRV)
        {
            RDG_LOG_CRITICAL(
                "Invalid combination of RenderTarget access and Read mode in RDGPassResourceUsage::MergeUsageFrom.");
        }
    }

} // namespace Ifrit::Runtime::RenderCore::RDG