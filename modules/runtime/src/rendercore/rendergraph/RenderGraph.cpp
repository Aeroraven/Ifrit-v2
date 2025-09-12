#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"
#include "ifrit/core/console/ConsoleObject.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/LogUtils.h"
#include "ifrit/core/algo/BuddyAllocator.h"

namespace Ifrit::Runtime::RenderCore::RDG
{

    enum ERDGMemoryHeapType
    {
        DeviceLocal = 0,
        HostShared  = 1,
        Count       = 2,
    };

    struct RDGSubresourcePair
    {
        u32  mMipLevel   = 0;
        u32  mArrayLayer = 0;

        bool operator==(const RDGSubresourcePair& other) const
        {
            return mMipLevel == other.mMipLevel && mArrayLayer == other.mArrayLayer;
        }
    };
} // namespace Ifrit::Runtime::RenderCore::RDG

namespace std
{
    template <> struct hash<Ifrit::Runtime::RenderCore::RDG::RDGSubresourcePair>
    {
        size_t operator()(const Ifrit::Runtime::RenderCore::RDG::RDGSubresourcePair& pair) const noexcept
        {
            size_t h1 = std::hash<Ifrit::u32>{}(pair.mMipLevel);
            size_t h2 = std::hash<Ifrit::u32>{}(pair.mArrayLayer);
            return h1 ^ (h2 << 1); // or use boost::hash_combine
        }
    };
} // namespace std

namespace Ifrit::Runtime::RenderCore::RDG
{
    static TConsoleVariable<u32> cvRDGEnableAsyncCompute("cv.RDG.AsyncCompute", 1,
        "Enable Async Compute Queue in Render Graph: 0 - Disabled, 1 - Manual, 2 - Forced", CVF_ReadOnly);
    static TConsoleVariable<u32> cvRDGEnableAsyncTransfer("cv.RDG.AsyncTransfer", 1,
        "Enable Async Transfer Queue in Render Graph: 0 - Disabled, 1 - Manual, 2 - Forced", CVF_ReadOnly);
    static TConsoleVariable<u32> cvRDGResourceReusingStrategy("cv.RDG.ResourceReusingStrategy", 2,
        "Resource Reusing Strategy: 0 - No Reuse, 1 - Resource Reusing, 2 - Memory Aliasing", CVF_ReadOnly);

    struct RDGGraphBuilderInternal
    {
        RDGGraphBuilderArgs mArgs;
    };

    // ===== RDG Resources =====
    struct RDGResourceLifetime
    {
        Array<u32, ERDGQueueType::Count> mLastUsePassIndex;
        u32                              mFirstUsePassIndex = ~0u;

        constexpr RDGResourceLifetime() noexcept
        {
            mLastUsePassIndex[ERDGQueueType::Graphics]      = ~0u;
            mLastUsePassIndex[ERDGQueueType::AsyncCompute]  = ~0u;
            mLastUsePassIndex[ERDGQueueType::AsyncTransfer] = ~0u;
        }
    };

    struct RDGManagedResourceAllocationRequest
    {
        u32                mResourceIndex = ~0u;

        // For aliasing
        ERDGMemoryHeapType mHeapType = ERDGMemoryHeapType::DeviceLocal;
        u32                mHeapId   = ~0u;
        u64                mOffset   = ~0ull;
    };

    struct RDGResourceAccess_UAV : public IRDGAccess_ResourceView
    {
        RHI::RhiUAVRef           mUAV;
        RHI::RhiDescriptorHandle GetDescriptor() override
        {
            RDG_NOTNULL(mUAV, "UAV is null in RDGResourceAccess_UAV");
            return mUAV->GetHandle();
        }
    };
    struct RDGResourceAccess_SRV : public IRDGAccess_ResourceView
    {
        RHI::RhiSRVRef           mSRV;
        RHI::RhiDescriptorHandle GetDescriptor() override
        {
            RDG_NOTNULL(mSRV, "SRV is null in RDGResourceAccess_SRV");
            return mSRV->GetHandle();
        }
    };
    struct RDGResourceAccess_Texture : public IRDGAccess_Texture
    {
        RHI::RhiTextureRef mTex;
        RHI::RhiTextureRef GetTexture() override
        {
            RDG_NOTNULL(mTex, "Texture is null in RDGResourceAccess_Texture");
            return mTex;
        }
    };
    struct RDGResourceAccess_Buffer : public IRDGAccess_Buffer
    {
        RHI::RhiBufferRef mBuf;
        RHI::RhiBufferRef GetBuffer() override
        {
            RDG_NOTNULL(mBuf, "Buffer is null in RDGResourceAccess_Buffer");
            return mBuf;
        }
    };

    class RDGResource
    {
    public:
        RDGResource(RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags) noexcept
            : mBuilder(builder), mName(name), mFlags(flags)
        {
        }
        virtual ~RDGResource() = default;

        IF_FORCEINLINE RDGResourceLifetime& GetLifetime() { return mLifetime; }
        IF_FORCEINLINE bool               IsImported() const { return HasFlagBit(mFlags, ERDGResourceFlag::Imported); }
        IF_FORCEINLINE const String&      GetName() const { return mName; }
        IF_FORCEINLINE ERDGMemoryHeapType GetHeapType() const { return mHeapType; }

        virtual u64                       GetRequiredMemorySize() const = 0;

    private:
        String                              mName;
        RDGResourceLifetime                 mLifetime;
        ERDGResourceFlags                   mFlags    = 0;
        ERDGMemoryHeapType                  mHeapType = ERDGMemoryHeapType::DeviceLocal;
        RDGGraphBuilder*                    mBuilder  = nullptr;
        RDGManagedResourceAllocationRequest mAllocationRequest;
    };

    class RDGTextureResource : public RDGResource
    {
    public:
        RDGTextureResource(
            RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags, const RDGTextureDesc& desc) noexcept
            : RDGResource(builder, name, flags), mDesc(desc), mImportedTexture(nullptr)
        {
        }
        RDGTextureResource(RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags,
            RHI::RhiTextureRef importedTex) noexcept
            : RDGResource(builder, name, flags), mImportedTexture(importedTex)
        {
            RDG_NOTNULL(importedTex, "Imported texture is null in RDGTextureResource");
            RDG_ASSERTION(HasFlagBit(flags, ERDGResourceFlag::Imported),
                "Imported texture must be marked as Imported in RDGTextureResource");
        }
        virtual ~RDGTextureResource() = default;

        virtual u64 GetRequiredMemorySize() const override
        {
            RDG_NOT_IMPLEMENTED();
            return 0;
        }

    private:
        RDGTextureDesc                                    mDesc;
        RHI::RhiTextureRef                                mImportedTexture = nullptr;

        RDGResourceAccess_Texture                         mTexAccess;
        HashMap<RHI::RhiImageSubResource, RHI::RhiSRVRef> mSRVs;
        HashMap<RHI::RhiImageSubResource, RHI::RhiUAVRef> mUAVs;
    };

    class RDGBufferResource : public RDGResource
    {
    public:
        RDGBufferResource(
            RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags, const RDGBufferDesc& desc) noexcept
            : RDGResource(builder, name, flags), mDesc(desc), mImportedBuffer(nullptr)
        {
        }
        RDGBufferResource(RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags,
            RHI::RhiBufferRef importedBuf) noexcept
            : RDGResource(builder, name, flags), mImportedBuffer(importedBuf)
        {
            RDG_NOTNULL(importedBuf, "Imported buffer is null in RDGBufferResource");
            RDG_ASSERTION(HasFlagBit(flags, ERDGResourceFlag::Imported),
                "Imported buffer must be marked as Imported in RDGBufferResource");
        }
        virtual ~RDGBufferResource() = default;

        virtual u64 GetRequiredMemorySize() const override
        {
            RDG_NOT_IMPLEMENTED();
            return 0;
        }

    private:
        RDGBufferDesc            mDesc;
        RHI::RhiBufferRef        mImportedBuffer = nullptr;
        RDGResourceAccess_Buffer mBufAccess;
        RHI::RhiSRVRef           mSRV;
        RHI::RhiUAVRef           mUAV;
    };

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
            ERDGReadWriteMode  mRWMode = ERDGReadWriteMode::None;
        };
        u32                                        mResourceIndex = ~0u;
        AccessPattern                              mOverallAccess;
        HashMap<RDGSubresourcePair, AccessPattern> mSubresourceAccesses;
        bool                                       mDetailedSubresourceTracking = false;
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
        RDGPass(RDGGraphBuilder* builder, const String& name, ERDGPassType type, Fn<void()> setup,
            Fn<void()> execute) noexcept
            : mBuilder(builder), mName(name), mType(type), mSetupFn(setup), mExecuteFn(execute)
        {
        }
        virtual ~RDGPass() = default;

        IF_FORCEINLINE ERDGPassType             GetType() const { return mType; }
        IF_FORCEINLINE void                     SetIdentifier(const RDGPassIdentifier& id) { mIdentifier = id; }
        IF_FORCEINLINE const RDGPassIdentifier& GetIdentifier() const { return mIdentifier; }
        IF_FORCEINLINE const Vec<RDGPassResourceUsage>& GetResourceUsages() const { return mResourceUsages; }
        IF_FORCEINLINE bool                             IsEnabled() const { return mEnabled; }
        IF_FORCEINLINE void                             SetEnabled(bool enabled) { mEnabled = enabled; }

    private:
        String                           mName;
        ERDGPassType                     mType = ERDGPassType::Invalid;
        Fn<void()>                       mSetupFn;
        Fn<void()>                       mExecuteFn;
        RDGGraphBuilder*                 mBuilder = nullptr;

        RDGPassIdentifier                mIdentifier;
        Vec<RDGPassResourceUsage>        mResourceUsages;
        Vec<RDGDownstreamPassDependency> mDownstreamPassDeps;
        Vec<RDGUpstreamPassDependency>   mUpstreamPassDeps;

        Vec<u32>                         mResourceCreationRequests;
        Vec<u32>                         mResourceReleaseRequests;
        bool                             mEnabled = false;
    };

    // ===== RDG Graphs =====

    struct RDGGraphContext
    {
        RDGGraphBuilderArgs                   mArgs;
        Vec<Owner<RDGResource>>               mResources;
        Vec<Owner<RDGPass>>                   mPasses;
        Array<Vec<u32>, ERDGQueueType::Count> mPassIndicesPerQueue;

        HashMap<ERDGPassType, ERDGQueueType>  mPassTypeToQueueType;
        Vec<Array<u32, ERDGQueueType::Count>> mCrossQueueDependencies;
        Vec<u32>                              mValidPassIds;

        void                                  Compile()
        {
            // Step: DAG build
            Compile_SetupParameters();
            Compile_PassIdentifierAssignment();
            Compile_ResourceAccessAnalysis();
            Compile_SetupDependencyGraph();
            Compile_FilterPasses();

            // Step: Rebuild optimized DAG
            Compile_PassIdentifierAssignment();
            Compile_ResourceAccessAnalysis();
            Compile_SetupDependencyGraph();

            // Step: Build resource allocations & transitions
            Compile_BuildManagedResourceAllocations();
            Compile_BuildTransitions();
        }
        void Compile_SetupParameters();
        void Compile_PassIdentifierAssignment();
        void Compile_ResourceAccessAnalysis();
        void Compile_SetupDependencyGraph();
        void Compile_FilterPasses();
        void Compile_BuildManagedResourceAllocations();
        void Compile_BuildTransitions();
    };

    struct RDGGraphRecordingContext
    {
        HashMap<u32, Array<Owner<RHI::RhiCommandListBase>, ERDGQueueType::Count>> mCmdListsPerQueue;
    };

    RDGGraphBuilder::RDGGraphBuilder(const RDGGraphBuilderArgs& args) : mData(new RDGGraphBuilderInternal(args)) {}

    RDGGraphBuilder::~RDGGraphBuilder()
    {
        delete mData;
        mData = nullptr;
    }

    // ===== RDG Graph Context Impls =====

    void RDGGraphContext::Compile_SetupParameters()
    {
        // Map pass types to queue types
        auto rhiBackend          = RHI::GetRhiBackend();
        auto rhiCaps             = rhiBackend->GetCapabilities();
        auto rhiHasAsyncCompute  = rhiCaps.bAsyncComputeEnable;
        auto rhiHasAsyncTransfer = rhiCaps.bAsyncTransferEnable;

        bool enableAsyncCompute =
            mArgs.mEnableAsyncCompute && (cvRDGEnableAsyncCompute.GetValue() != 0) && rhiHasAsyncCompute;
        bool enableAsyncTransfer =
            mArgs.mEnableAsyncTransfer && (cvRDGEnableAsyncTransfer.GetValue() != 0) && rhiHasAsyncTransfer;
        bool forcedAsyncCompute =
            mArgs.mEnableAsyncCompute && (cvRDGEnableAsyncCompute.GetValue() == 2) && rhiHasAsyncCompute;
        bool forcedAsyncTransfer =
            mArgs.mEnableAsyncTransfer && (cvRDGEnableAsyncTransfer.GetValue() == 2) && rhiHasAsyncTransfer;

        mPassTypeToQueueType[ERDGPassType::Graphics] = ERDGQueueType::Graphics;
        mPassTypeToQueueType[ERDGPassType::Compute] =
            forcedAsyncCompute ? ERDGQueueType::AsyncCompute : ERDGQueueType::Graphics;
        mPassTypeToQueueType[ERDGPassType::Transfer] =
            forcedAsyncTransfer ? ERDGQueueType::AsyncTransfer : ERDGQueueType::Graphics;
        mPassTypeToQueueType[ERDGPassType::AsyncCompute] =
            enableAsyncCompute ? ERDGQueueType::AsyncCompute : ERDGQueueType::Graphics;
        mPassTypeToQueueType[ERDGPassType::AsyncTransfer] =
            enableAsyncTransfer ? ERDGQueueType::AsyncTransfer : ERDGQueueType::Graphics;

        // Assume all passes are valid initially
        mValidPassIds.resize(mPasses.size());
        for (u32 i = 0; i < mPasses.size(); ++i)
        {
            mValidPassIds[i] = i;
        }
    }

    void RDGGraphContext::Compile_PassIdentifierAssignment()
    {
        // Assign pass identifiers and group passes per queue
        for (u32 i = 0; i < ERDGQueueType::Count; ++i)
        {
            mPassIndicesPerQueue[i].clear();
        }
        for (u32 i = 0; i < mValidPassIds.size(); ++i)
        {
            auto  passIndex = mValidPassIds[i];
            auto& pass      = mPasses[passIndex];
            auto  it        = mPassTypeToQueueType.find(pass->GetType());
            RDG_ASSERTION(it != mPassTypeToQueueType.end(), "Unsupported pass type in RDGGraphContext::Compile");

            auto              queueType = it->second;
            RDGPassIdentifier passIdentifier;
            passIdentifier.mQueueType = queueType;
            passIdentifier.mIndex     = static_cast<u32>(mPassIndicesPerQueue[queueType].size());
            pass->SetIdentifier(passIdentifier);

            mPassIndicesPerQueue[queueType].emplace_back(passIndex);
        }
    }

    void RDGGraphContext::Compile_ResourceAccessAnalysis()
    {
        for (auto& res : mResources)
        {
            res->GetLifetime() = RDGResourceLifetime();
        }

        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto& pass           = mPasses[mValidPassIds[i]];
            auto  passIdentifier = pass->GetIdentifier();
            for (const auto& resUsage : pass->GetResourceUsages())
            {
                auto& res      = mResources[resUsage.mResourceIndex];
                auto& lifetime = res->GetLifetime();
                if (lifetime.mFirstUsePassIndex == ~0u)
                {
                    lifetime.mFirstUsePassIndex = i;
                }
                lifetime.mLastUsePassIndex[passIdentifier.mQueueType] = i;
            }
        }
    }

    void RDGGraphContext::Compile_SetupDependencyGraph()
    {
        using QueueUpstreamDeps      = Array<u32, ERDGQueueType::Count>;
        using ResourceLastUsedPassId = Array<u32, ERDGQueueType::Count>;
        Vec<QueueUpstreamDeps>           passCrossQueueDependencies(mPasses.size());
        Vec<ResourceLastUsedPassId>      resourceLastUsedPassId(mResources.size());
        Array<u32, ERDGQueueType::Count> currentLastPassInQueue = { ~0u, ~0u, ~0u };

        auto                             fnGetMax = [](u32 a, u32 b) -> u32 {
            if (a == ~0u)
                return b;
            if (b == ~0u)
                return a;
            return (a > b) ? a : b;
        };

        // Initialize vectors
        for (auto& deps : passCrossQueueDependencies)
        {
            deps.fill(~0u);
        }
        for (auto& lastUsed : resourceLastUsedPassId)
        {
            lastUsed.fill(~0u);
        }

        // Build dependencies
        for (u32 i = 0; i < mValidPassIds.size(); ++i)
        {
            auto  passIndex      = mValidPassIds[i];
            auto& pass           = mPasses[passIndex];
            auto& passIdentifier = pass->GetIdentifier();

            for (const auto& resUsage : pass->GetResourceUsages())
            {
                auto& res          = mResources[resUsage.mResourceIndex];
                auto& queueResDeps = resourceLastUsedPassId[resUsage.mResourceIndex];
                for (u32 q = 0; q < ERDGQueueType::Count; ++q)
                {
                    passCrossQueueDependencies[passIndex][q] =
                        fnGetMax(passCrossQueueDependencies[passIndex][q], queueResDeps[q]);
                }
                queueResDeps[passIdentifier.mQueueType] = passIndex;
            }

            passCrossQueueDependencies[passIndex][passIdentifier.mQueueType] =
                currentLastPassInQueue[passIdentifier.mQueueType];
            currentLastPassInQueue[passIdentifier.mQueueType] = passIndex;
        }

        // Transitive reduce the DAG
        auto fnRemovalRedundantEdges = [&](u32& a, u32& b) -> void {
            // remove edges C->B if there is a path C->A->B
            auto  aPrevGraphics = passCrossQueueDependencies[a][ERDGQueueType::Graphics];
            auto  aPrevCompute  = passCrossQueueDependencies[a][ERDGQueueType::AsyncCompute];
            auto  aPrevTransfer = passCrossQueueDependencies[a][ERDGQueueType::AsyncTransfer];
            auto& bPrevGraphics = passCrossQueueDependencies[b][ERDGQueueType::Graphics];
            auto& bPrevCompute  = passCrossQueueDependencies[b][ERDGQueueType::AsyncCompute];
            auto& bPrevTransfer = passCrossQueueDependencies[b][ERDGQueueType::AsyncTransfer];

            if (bPrevGraphics <= aPrevGraphics && aPrevGraphics != ~0u)
            {
                bPrevGraphics = ~0u;
            }
            if (bPrevCompute <= aPrevCompute && aPrevCompute != ~0u)
            {
                bPrevCompute = ~0u;
            }
            if (bPrevTransfer <= aPrevTransfer && aPrevTransfer != ~0u)
            {
                bPrevTransfer = ~0u;
            }
        };

        for (u32 i = 0; i < mValidPassIds.size(); ++i)
        {
            auto passIndex        = mValidPassIds[i];
            auto passDepsGraphics = passCrossQueueDependencies[passIndex][ERDGQueueType::Graphics];
            auto passDepsCompute  = passCrossQueueDependencies[passIndex][ERDGQueueType::AsyncCompute];
            auto passDepsTransfer = passCrossQueueDependencies[passIndex][ERDGQueueType::AsyncTransfer];
            if (passDepsGraphics != ~0u)
            {
                fnRemovalRedundantEdges(passDepsGraphics, passIndex);
            }
            if (passDepsCompute != ~0u)
            {
                fnRemovalRedundantEdges(passDepsCompute, passIndex);
            }
            if (passDepsTransfer != ~0u)
            {
                fnRemovalRedundantEdges(passDepsTransfer, passIndex);
            }
        }
        mCrossQueueDependencies = std::move(passCrossQueueDependencies);
    }

    void RDGGraphContext::Compile_FilterPasses()
    {
        // Util fns
        auto fnGetMax = [](u32 a, u32 b) -> u32 {
            if (a == ~0u)
                return b;
            if (b == ~0u)
                return a;
            return (a > b) ? a : b;
        };
        auto fnCheckResourceRWMode = [&](const RDGPassResourceUsage& res, ERDGReadWriteMode mode) -> bool {
            if (res.mDetailedSubresourceTracking)
            {
                for (const auto& [subres, access] : res.mSubresourceAccesses)
                {
                    if (HasFlagBit(access.mRWMode, mode))
                    {
                        return true;
                    }
                }
                return false;
            }
            else
            {
                return HasFlagBit(res.mOverallAccess.mRWMode, mode);
            }
        };

        // Collect passes that write to external resources
        Vec<u32>     validPasses;
        HashSet<u32> influencingResources;
        for (auto i = 0; i < mPasses.size(); ++i)
        {
            auto& pass = mPasses[i];
            for (auto& resUsage : pass->GetResourceUsages())
            {
                auto& res        = mResources[resUsage.mResourceIndex];
                bool  isWriting  = fnCheckResourceRWMode(resUsage, ERDGReadWriteMode::Write);
                bool  isExternal = res->IsImported();
                if (isWriting && isExternal)
                {

                    influencingResources.insert(resUsage.mResourceIndex);
                    validPasses.emplace_back(i);
                    break;
                }
            }
        }
        // Backtrack to find all passes that influence the above passes
        Vec<u32> outgoingEdges(mPasses.size(), 0);
        for (auto i = 0; i < mPasses.size(); ++i)
        {
            auto& passDeps = mCrossQueueDependencies[i];
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (passDeps[q] != ~0u)
                {
                    outgoingEdges[passDeps[q]]++;
                }
            }
        }
        Queue<u32> pendingPasses;
        for (auto i = 0; i < outgoingEdges.size(); ++i)
        {
            if (outgoingEdges[i] == 0)
            {
                pendingPasses.push(i);
            }
        }
        while (!pendingPasses.empty())
        {
            auto passIdx = pendingPasses.front();
            pendingPasses.pop();

            bool isValidPass = false;
            for (auto& resUsage : mPasses[passIdx]->GetResourceUsages())
            {
                auto isWriting = fnCheckResourceRWMode(resUsage, ERDGReadWriteMode::Write);
                if (influencingResources.find(resUsage.mResourceIndex) != influencingResources.end() && isWriting)
                {
                    isValidPass = true;
                    break;
                }
            }
            if (isValidPass)
            {
                for (auto& resUsage : mPasses[passIdx]->GetResourceUsages())
                {
                    auto it = influencingResources.find(resUsage.mResourceIndex);
                    if (it != influencingResources.end())
                    {
                        if (fnCheckResourceRWMode(resUsage, ERDGReadWriteMode::Read))
                        {
                            influencingResources.insert(resUsage.mResourceIndex);
                        }
                    }
                }
                mPasses[passIdx]->SetEnabled(true);
            }
            auto& passDeps = mCrossQueueDependencies[passIdx];
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (passDeps[q] != ~0u)
                {
                    outgoingEdges[passDeps[q]]--;
                    if (outgoingEdges[passDeps[q]] == 0)
                    {
                        pendingPasses.push(passDeps[q]);
                    }
                }
            }
        }
        // Record passes to be enabled
        Vec<u32> passesToKeep;
        for (auto i = 0; i < mPasses.size(); ++i)
        {
            if (mPasses[i]->IsEnabled())
            {
                passesToKeep.emplace_back(i);
            }
        }
        mValidPassIds = std::move(passesToKeep);
    }

    void RDGGraphContext::Compile_BuildManagedResourceAllocations()
    {
        // stat max memory required
        Array<u64, ERDGMemoryHeapType::Count> maxHeapSize = { 0, 0 };
        Vec<Vec<u32>>                         passSuccessors(mPasses.size());
        Vec<u32>                              passIncomingEdges(mPasses.size(), 0);
        Queue<u32>                            pendingPasses;
        Vec<u32>                              resourceQueueActiveStates(mResources.size(), 0);
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto  passIndex = mValidPassIds[i];
            auto& passDeps  = mCrossQueueDependencies[passIndex];
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (passDeps[q] != ~0u)
                {
                    passSuccessors[passDeps[q]].emplace_back(passIndex);
                    passIncomingEdges[passIndex]++;
                }
            }
            if (passIncomingEdges[passIndex] == 0)
            {
                pendingPasses.push(passIndex);
            }
        }
        for (auto i = 0; i < mResources.size(); ++i)
        {
            auto& res        = mResources[i];
            auto  isImported = res->IsImported();
            if (!isImported)
            {
                auto& lifetime = res->GetLifetime();
                for (u32 q = 0; q < ERDGQueueType::Count; ++q)
                {
                    if (lifetime.mLastUsePassIndex[q] != ~0u)
                    {
                        resourceQueueActiveStates[i]++;
                    }
                }
            }
        }
        // Topological traverse the DAG
        while (!pendingPasses.empty())
        {
            auto passIndex = pendingPasses.front();
            pendingPasses.pop();

            for (auto& resUsage : mPasses[passIndex]->GetResourceUsages())
            {
                auto& res        = mResources[resUsage.mResourceIndex];
                auto  isImported = res->IsImported();
                auto  memSize    = res->GetRequiredMemorySize();
                if (isImported)
                    continue;
            }
        }
    }

} // namespace Ifrit::Runtime::RenderCore::RDG