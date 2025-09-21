#pragma once
#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"

#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.GlobalVars.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Utils.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Resources.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Passes.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/LogUtils.h"

namespace Ifrit::Runtime::RenderCore::RDG
{
    // ===== RDG Graphs =====
    struct RDGTransition
    {
        u32                      mResourceIdx = ~0u;
        RHI::RhiImageSubResource mSubresource;
        RHI::ERhiResourceState   mStateBefore   = RHI::ERhiResourceState::Undefined;
        RHI::ERhiResourceState   mStateAfter    = RHI::ERhiResourceState::Undefined;
        ERDGQueueType            mSrcQueue      = ERDGQueueType::Graphics;
        ERDGQueueType            mDstQueue      = ERDGQueueType::Graphics;
        bool                     mWholeResource = true;
    };

    struct RDGTrackedResourceState
    {
        u32                            mLastAccessedPassIndex = ~0u;
        RHI::ERhiResourceState         mCurrentState          = RHI::ERhiResourceState::Undefined;
        ERDGQueueType                  mCurrentQueue          = ERDGQueueType::Graphics;
        ERDGReadWriteMode              mCurrentRWMode         = ERDGReadWriteModeFlag::None;

        static RDGTrackedResourceState FromResourceUsage(const RDGPassResourceUsage& usage, ERDGQueueType queue)
        {
            RDGTrackedResourceState state;
            state.mCurrentQueue = queue;

            using AccessModeTp = ERDGReadWriteMode;

            struct RequiredAccessMode
            {
                RHI::ERhiResourceState mState;
                ERDGReadWriteMode      mRWMode;
            };

            auto fnMergeState = [](RHI::ERhiResourceState  src,
                                    RHI::ERhiResourceState pending) -> RHI::ERhiResourceState {
                constexpr Array<RHI::ERhiResourceState, 3> kUnorderedAccessFlags = {
                    RHI::ERhiResourceState::UnorderedAccess,
                    RHI::ERhiResourceState::UnorderedAccess_Read,
                    RHI::ERhiResourceState::UnorderedAccess_Write,
                };

                if (src == RHI::ERhiResourceState::Undefined)
                    return pending;
                if (pending == RHI::ERhiResourceState::Undefined || src == pending
                    || src == RHI::ERhiResourceState::Common)
                    return src;
                if (pending == RHI::ERhiResourceState::Common)
                    return pending;

                bool isSrcUAV     = (src == RHI::ERhiResourceState::UnorderedAccess
                    || src == RHI::ERhiResourceState::UnorderedAccess_Read
                    || src == RHI::ERhiResourceState::UnorderedAccess_Write);
                bool isPendingUAV = (pending == RHI::ERhiResourceState::UnorderedAccess
                    || pending == RHI::ERhiResourceState::UnorderedAccess_Read
                    || pending == RHI::ERhiResourceState::UnorderedAccess_Write);
                if (isSrcUAV && isPendingUAV && src != pending)
                {
                    return RHI::ERhiResourceState::UnorderedAccess;
                }
                return RHI::ERhiResourceState::Common;
            };

            auto fnTranslateFromAccessMode = [&fnMergeState](ERDGResourceAccess access) -> RHI::ERhiResourceState {
                RHI::ERhiResourceState desired = RHI::ERhiResourceState::Undefined;
                if (HasFlagBit(access, ERDGResourceAccessFlag::CopyDst))
                {
                    desired = fnMergeState(desired, RHI::ERhiResourceState::CopyDst);
                }
                if (HasFlagBit(access, ERDGResourceAccessFlag::CopySrc))
                {
                    desired = fnMergeState(desired, RHI::ERhiResourceState::CopySrc);
                }
                if (HasFlagBit(access, ERDGResourceAccessFlag::IndirectArgRead))
                {
                    desired = fnMergeState(desired, RHI::ERhiResourceState::Common);
                }
                if (HasFlagBit(access, ERDGResourceAccessFlag::RenderTarget))
                {
                    desired = fnMergeState(desired, RHI::ERhiResourceState::ColorRT);
                }
                if (HasFlagBit(access, ERDGResourceAccessFlag::DepthStencil))
                {
                    desired = fnMergeState(desired, RHI::ERhiResourceState::DepthStencilRT);
                }
                if (HasFlagBit(access, ERDGResourceAccessFlag::SRVRead))
                {
                    desired = fnMergeState(desired, RHI::ERhiResourceState::ShaderRead);
                }
                if (HasFlagBit(access, ERDGResourceAccessFlag::UAVRead))
                {
                    desired = fnMergeState(desired, RHI::ERhiResourceState::UnorderedAccess_Read);
                }
                if (HasFlagBit(access, ERDGResourceAccessFlag::UAVWrite))
                {
                    desired = fnMergeState(desired, RHI::ERhiResourceState::UnorderedAccess_Write);
                }
                // Check warnings
                bool isUAV = HasFlagBit(access, ERDGResourceAccessFlag::UAVRead)
                    || HasFlagBit(access, ERDGResourceAccessFlag::UAVWrite);
                bool isRTV = HasFlagBit(access, ERDGResourceAccessFlag::RenderTarget)
                    || HasFlagBit(access, ERDGResourceAccessFlag::DepthStencil);
                bool isSRV = HasFlagBit(access, ERDGResourceAccessFlag::SRVRead);
                if ((isUAV || isSRV) && isRTV) IF_UNLIKELY
                {
                    RDG_LOG_WARNING("Resource access pattern has both UAV/SRV and RTV/DSV access");
                }
                return desired;
            };

            auto fnFromAccessPattern = [&fnTranslateFromAccessMode, &fnMergeState](
                                           const RDGPassResourceUsage::AccessPattern& pattern,
                                           RequiredAccessMode&                        requiredAccess) -> void {
                // Merge RW mode
                if (HasFlagBit(pattern.mRWMode, ERDGReadWriteModeFlag::Read))
                {
                    requiredAccess.mRWMode = SetFlagBit(requiredAccess.mRWMode, ERDGReadWriteModeFlag::Read);
                }
                if (HasFlagBit(pattern.mRWMode, ERDGReadWriteModeFlag::Write))
                {
                    requiredAccess.mRWMode = SetFlagBit(requiredAccess.mRWMode, ERDGReadWriteModeFlag::Write);
                }
                // Merge state
                auto desiredState     = fnTranslateFromAccessMode(pattern.mAccess);
                requiredAccess.mState = fnMergeState(requiredAccess.mState, desiredState);
            };

            RequiredAccessMode requiredAccess;
            requiredAccess.mState  = RHI::ERhiResourceState::Undefined;
            requiredAccess.mRWMode = ERDGReadWriteModeFlag::None;
            if (usage.mDetailedSubresourceTracking)
            {
                for (const auto& [subres, access] : usage.mSubresourceAccesses)
                {
                    fnFromAccessPattern(access, requiredAccess);
                }
            }
            else
            {
                fnFromAccessPattern(usage.mOverallAccess, requiredAccess);
            }
            state.mCurrentState  = requiredAccess.mState;
            state.mCurrentRWMode = requiredAccess.mRWMode;
            return state;
        }
    };

    struct RDGGraphContext;

    struct RDGPhysicalResourcePool : public NonCopyable
    {
        template <typename T> struct TTrackedResource : public NonCopyable
        {
            Owner<T>            mResource;
            RDGResourceLifetime mLifetime;
            u32                 mInfoId;

            // Moving operations
            TTrackedResource() noexcept : mResource(nullptr), mLifetime(), mInfoId(~0u) {}
            TTrackedResource(TTrackedResource&& rhs) noexcept
                : mResource(std::move(rhs.mResource)), mLifetime(rhs.mLifetime), mInfoId(rhs.mInfoId)
            {
                rhs.mInfoId   = ~0u;
                rhs.mLifetime = RDGResourceLifetime();
            }
            TTrackedResource& operator=(TTrackedResource&& rhs) noexcept
            {
                if (this != &rhs)
                {
                    mResource     = std::move(rhs.mResource);
                    mLifetime     = rhs.mLifetime;
                    mInfoId       = rhs.mInfoId;
                    rhs.mInfoId   = ~0u;
                    rhs.mLifetime = RDGResourceLifetime();
                }
                return *this;
            }
        };
        struct ResourceInfo
        {
            ERDGResourceType mType;
            u32              mIndex;
        };

        Vec<ResourceInfo>                         mResourceInfos;
        Vec<TTrackedResource<RDGPhysicalTexture>> mTextures;
        Vec<TTrackedResource<RDGPhysicalBuffer>>  mBuffers;

        u32 RegisterTransientBuffer(const RDGBufferDesc& desc, RDGResourceLifetime lifetime,
            const Vec<Array<u32, ERDGQueueType::Count>>& dependencies);
        u32 RegisterTransientTexture(const RDGTextureDesc& desc, RDGResourceLifetime lifetime,
            const Vec<Array<u32, ERDGQueueType::Count>>& dependencies);
    };

    class RDGSetupContext : public IRDGGraphBuilderSetupContext
    {
    public:
        RDGSetupContext(RDGGraphContext* ctx) noexcept : mCtx(ctx) { RDG_NOTNULL(ctx, "RDGGraphContext is null"); }
        virtual ~RDGSetupContext() = default;

        virtual IRDGAccess_ResourceView* CreateSRV(
            RDGTextureHandle handle, const TOptional<RHI::RhiImageSubResource>& subRes) override final;
        virtual IRDGAccess_ResourceView* CreateSRV(RDGBufferHandle handle) override final;
        virtual IRDGAccess_ResourceView* CreateUAV(RDGTextureHandle handle,
            const TOptional<RHI::RhiImageSubResource>& subRes, ERDGReadWriteMode mode) override final;
        virtual IRDGAccess_ResourceView* CreateUAV(RDGBufferHandle handle, ERDGReadWriteMode mode) override final;

        virtual IRDGAccess_Buffer*       AsIndirectArg(RDGBufferHandle handle) override final;
        virtual IRDGAccess_Texture*      AsRenderTarget(RDGTextureHandle handle,
                 TOptional<RHI::RhiImageSubResource> clearValue, RHI::ERhiRenderTargetLoadOp loadOp,
                 TOptional<RHI::RhiClearColorValue> clearColor) override final;
        virtual IRDGAccess_Texture*      AsDepthStencil(RDGTextureHandle handle,
                 TOptional<RHI::RhiImageSubResource> clearValue, RHI::ERhiRenderTargetLoadOp loadOp,
                 TOptional<f32> clearDepth, TOptional<u32> clearStencil) override final;

        void        AddFullResourceUsageToPass(u32 resourceIdx, ERDGResourceAccess access, ERDGReadWriteMode rwMode,
                   const TOptional<RHI::RhiClearColorValue>&        clearColor,
                   const TOptional<RHI::RhiClearDepthStencilValue>& clearDepthStencil,
                   RHI::ERhiRenderTargetLoadOp                      rtLoadOp = RHI::ERhiRenderTargetLoadOp::DontCare);

        inline void SetCurrentPass(u32 index) { mCurrentPassId = index; }

    private:
        RDGGraphContext* mCtx           = nullptr;
        u32              mCurrentPassId = ~0u;
    };

    struct RDGGraphContext : public NonCopyable
    {
        RDGGraphBuilderArgs                   mArgs;
        Vec<Owner<RDGResource>>               mResources;
        Vec<Owner<RDGPass>>                   mPasses;
        Array<Vec<u32>, ERDGQueueType::Count> mPassIndicesPerQueue;

        THashMap<ERDGPassType, ERDGQueueType> mPassTypeToQueueType;
        Vec<Array<u32, ERDGQueueType::Count>> mCrossQueueDependenciesFull;
        Vec<Array<u32, ERDGQueueType::Count>> mCrossQueueDependenciesVital;
        Vec<u32>                              mValidPassIds;
        Vec<Array<u32, ERDGQueueType::Count>> mPassSyncPoints;

        Vec<RDGTransition>                    mTransitions;
        Vec<Vec<u32>>                         mTransitionBeginRequests_Pass;
        Vec<Vec<u32>>                         mTransitionEndRequests_Pass;
        Vec<u32>                              mTransitionBeginRequests_GraphStart;
        Vec<u32>                              mTransitionEndRequests_GraphEnd;

        Owner<RDGPhysicalResourcePool>        mPhysicalResourcePool;
        Owner<RDGSetupContext>                mSetupCtx;

        void                                  Compile();
        void                                  Compile_SetupParameters();
        void                                  Compile_PassIdentifierAssignment();
        void                                  Compile_ResourceAccessAnalysis();
        void                                  Compile_SetupDependencyGraph();
        void                                  Compile_FilterPasses();
        void                                  Compile_CollectSyncPoints();
        void                                  Compile_BuildManagedResourceAllocations();
        void                                  Compile_BuildTransitions();

        void                                  Visualize_DumpCompiledDOTGraph(const String& filepath) const;
        void                                  Visualize_DumpCompiledDOTGraphWithResources(const String& filepath) const;
        void                                  Visualize_DumpPhysicalResourcesAllocation() const;

        void                                  Execute();
        void                                  Execute_AllocatePhysicalResources();

        RDGGraphContext() = default;
    };

    struct RDGGraphRecordingContext
    {
        THashMap<u32, Array<Owner<RHI::RhiCommandListBase>, ERDGQueueType::Count>> mCmdListsPerQueue;
    };

} // namespace Ifrit::Runtime::RenderCore::RDG