#pragma once
#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Context.h"

#include "ifrit/core/base/containers/Queue.h"
#include "ifrit/core/algo/Graph.h"
#include "ifrit/core/algo/StlStringUtils.h"
#include "ifrit/core/file/FileOps.h"
namespace Ifrit::Runtime::RenderCore::RDG
{
    // ===== RDG Graph Context Impls =====

    void RDGGraphContext::Compile()
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
        Compile_CollectSyncPoints();
        Compile_BuildManagedResourceAllocations();
        Compile_BuildTransitions();
    }

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

        // Setup resource pool
        mPhysicalResourcePool = MakeOwner<RDGPhysicalResourcePool>();
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

            // print pass dependencies

            RDG_LOG_DEBUG("[{}] Pass {} (Type {}) depends on:", i, pass->GetName(), (u32)pass->GetType());
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (passCrossQueueDependencies[passIndex][q] != ~0u)
                {
                    auto depPassIndex = passCrossQueueDependencies[passIndex][q];
                    RDG_LOG_DEBUG("  - Queue {}: Pass {}", q, mPasses[depPassIndex]->GetName());
                }
            }
        }
        mCrossQueueDependenciesFull = passCrossQueueDependencies;

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
        mCrossQueueDependenciesVital = std::move(passCrossQueueDependencies);
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
        auto fnCheckResourceRWMode = [&](const RDGPassResourceUsage& res, ERDGReadWriteModeFlag mode) -> bool {
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
        Vec<u32>      validPasses;
        THashSet<u32> influencingResources;
        for (auto i = 0; i < mPasses.size(); ++i)
        {
            auto& pass = mPasses[i];
            for (auto& resUsage : pass->GetResourceUsages())
            {
                auto& res        = mResources[resUsage.mResourceIndex];
                bool  isWriting  = fnCheckResourceRWMode(resUsage, ERDGReadWriteModeFlag::Write);
                bool  isExternal = res->IsImported();
                if (isWriting && isExternal)
                {
                    // print influencing resource
                    RDG_LOG_DEBUG("Pass {} is writing to external resource {} (Idx {})", pass->GetName(),
                        res->GetName(), resUsage.mResourceIndex);
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
            auto& passDeps = mCrossQueueDependenciesVital[i];
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (passDeps[q] != ~0u)
                {
                    outgoingEdges[passDeps[q]]++;
                }
            }
        }
        TQueue<u32> pendingPasses;
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

            RDG_LOG_DEBUG("Considering pass {} for enabling", mPasses[passIdx]->GetName());

            bool isValidPass = false;
            for (auto& resUsage : mPasses[passIdx]->GetResourceUsages())
            {
                auto isWriting = fnCheckResourceRWMode(resUsage, ERDGReadWriteModeFlag::Write);
                RDG_LOG_DEBUG("  - Resource {} (Idx {}) is {}writing", mResources[resUsage.mResourceIndex]->GetName(),
                    resUsage.mResourceIndex, isWriting ? "" : "not ");

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
                    if (it == influencingResources.end())
                    {
                        if (fnCheckResourceRWMode(resUsage, ERDGReadWriteModeFlag::Read))
                        {
                            // print influencing resource
                            RDG_LOG_DEBUG("Pass {} is reading from influencing resource {} (Idx {})",
                                mPasses[passIdx]->GetName(), mResources[resUsage.mResourceIndex]->GetName(),
                                resUsage.mResourceIndex);
                            influencingResources.insert(resUsage.mResourceIndex);
                        }
                    }
                }
                mPasses[passIdx]->SetEnabled(true);
            }
            auto& passDeps = mCrossQueueDependenciesVital[passIdx];
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

    void RDGGraphContext::Compile_CollectSyncPoints()
    {
        // Find out sync points
        Vec<Array<u32, ERDGQueueType::Count>> passSyncPoints(mPasses.size());
        for (u32 i = 0; i < mValidPassIds.size(); ++i)
        {
            auto passIndex = mValidPassIds[i];
            passSyncPoints[passIndex].fill(~0u);
        }
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto  passIndex = mValidPassIds[i];
            auto& passDeps  = mCrossQueueDependenciesFull[passIndex];
            auto  passQueue = mPasses[passIndex]->GetIdentifier().mQueueType;
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                auto prev = passDeps[q];
                if (prev != ~0u)
                {
                    passSyncPoints[prev][passQueue] = GetMinPassIndex(passSyncPoints[prev][passQueue], passIndex);
                }
            }
        }

        // Reverse propagate the sync points using topological sort
        Vec<u32>      passIncomingEdges(mPasses.size(), 0);
        Vec<Vec<u32>> passSuccessors(mPasses.size());
        TQueue<u32>   pendingPasses;
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto  passIndex = mValidPassIds[i];
            auto& passDeps  = mCrossQueueDependenciesFull[passIndex];
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (passDeps[q] != ~0u)
                {
                    passIncomingEdges[passDeps[q]]++;
                    passSuccessors[passDeps[q]].emplace_back(passIndex);
                }
            }
        }
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto passIndex = mValidPassIds[i];
            if (passIncomingEdges[passIndex] == 0)
            {
                pendingPasses.push(passIndex);
            }
        }
        while (!pendingPasses.empty())
        {
            auto passIndex = pendingPasses.front();
            pendingPasses.pop();

            for (auto& succ : passSuccessors[passIndex])
            {
                for (u32 q = 0; q < ERDGQueueType::Count; ++q)
                {
                    passSyncPoints[passIndex][q] =
                        GetMinPassIndex(passSyncPoints[passIndex][q], passSyncPoints[succ][q]);
                }
            }
            auto& predecessor = mCrossQueueDependenciesFull[passIndex];
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (predecessor[q] != ~0u)
                {
                    passIncomingEdges[predecessor[q]]--;
                    if (passIncomingEdges[predecessor[q]] == 0)
                    {
                        pendingPasses.push(predecessor[q]);
                    }
                }
            }
        }
        mPassSyncPoints = std::move(passSyncPoints);
    }

    void RDGGraphContext::Compile_BuildManagedResourceAllocations()
    {
        for (auto& res : mResources)
        {
            bool isTransient = !res->IsImported();
            if (isTransient)
            {
                auto resType = res->GetType();
                if (resType == ERDGResourceType::Buffer)
                {
                    auto downcasted = static_cast<RDGBufferResource*>(res.get());
                    auto id         = mPhysicalResourcePool->RegisterTransientBuffer(
                        downcasted->GetDesc(), downcasted->GetLifetime(), mCrossQueueDependenciesFull);
                    downcasted->SetTransientResourceId(id);
                }
                else if (resType == ERDGResourceType::Texture)
                {
                    auto downcasted = static_cast<RDGTextureResource*>(res.get());
                    auto id         = mPhysicalResourcePool->RegisterTransientTexture(
                        downcasted->GetDesc(), downcasted->GetLifetime(), mCrossQueueDependenciesFull);
                    downcasted->SetTransientResourceId(id);
                }
                else
                {
                    RDG_ASSERTION(
                        false, "Unsupported resource type in RDGGraphContext::Compile_BuildManagedResourceAllocations");
                }
            }
        }
    }

    void RDGGraphContext::Compile_BuildTransitions()
    {
        Vec<RDGTrackedResourceState> resourceStates(mResources.size());
        for (auto i = 0; i < resourceStates.size(); ++i)
        {
            auto& res         = mResources[i];
            auto& resLifetime = res->GetLifetime();
            auto  startPassId = resLifetime.mFirstUsePassIndex;
            if (startPassId == ~0u)
            {
                continue;
            }
            auto startPassType = mPasses[startPassId]->GetType();

            resourceStates[i].mCurrentState = res->GetInitialState();
            resourceStates[i].mCurrentQueue = mPassTypeToQueueType[startPassType];
        }

        // Topological sort
        Vec<u32>      passIncomingEdges(mPasses.size(), 0);
        Vec<Vec<u32>> passSuccessors(mPasses.size());
        TQueue<u32>   pendingPasses;
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto  passIndex = mValidPassIds[i];
            auto& passDeps  = mCrossQueueDependenciesVital[passIndex];
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (passDeps[q] != ~0u)
                {
                    passIncomingEdges[passIndex]++;
                    passSuccessors[passDeps[q]].emplace_back(passIndex);
                }
            }
        }
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto passIndex = mValidPassIds[i];
            if (passIncomingEdges[passIndex] == 0)
            {
                pendingPasses.push(passIndex);
            }
        }
        mTransitionEndRequests_Pass.clear();
        mTransitionBeginRequests_Pass.clear();
        mTransitionBeginRequests_GraphStart.clear();
        mTransitionEndRequests_GraphEnd.clear();
        mTransitionBeginRequests_Pass.resize(mPasses.size());
        mTransitionEndRequests_Pass.resize(mPasses.size());

        while (!pendingPasses.empty())
        {
            auto passIndex = pendingPasses.front();
            pendingPasses.pop();
            auto& pass = mPasses[passIndex];

            for (auto& resUsage : pass->GetResourceUsages())
            {
                auto  resIdx   = resUsage.mResourceIndex;
                auto& resState = resourceStates[resIdx];
                auto& res      = mResources[resIdx];

                auto  desiredState =
                    RDGTrackedResourceState::FromResourceUsage(resUsage, pass->GetIdentifier().mQueueType);

                // Check if UAV barrier is needed
                bool lastRead   = HasFlagBit(resState.mCurrentRWMode, ERDGReadWriteModeFlag::Read);
                bool lastWrite  = HasFlagBit(resState.mCurrentRWMode, ERDGReadWriteModeFlag::Write);
                bool nextRead   = HasFlagBit(desiredState.mCurrentRWMode, ERDGReadWriteModeFlag::Read);
                bool nextWrite  = HasFlagBit(desiredState.mCurrentRWMode, ERDGReadWriteModeFlag::Write);
                bool isRAW      = lastWrite && nextRead;
                bool isDataDeps = isRAW || lastWrite;

                bool isLastCommonState =
                    (resState.mCurrentState == RHI::ERhiResourceState::Common || IsStateInUAV(resState.mCurrentState));
                bool isDesiredCommonState = (desiredState.mCurrentState == RHI::ERhiResourceState::Common
                    || IsStateInUAV(desiredState.mCurrentState));

                bool needUAVBarrier = (isLastCommonState && isDesiredCommonState && isDataDeps);

                // Check for layout transition
                bool isToCommonLayout      = (!isLastCommonState && isDesiredCommonState);
                bool isFromCommonLayout    = (isLastCommonState && !isDesiredCommonState);
                bool isAllUAVLayout        = (isLastCommonState && isDesiredCommonState);
                bool isStateChange         = (!isFromCommonLayout && !isToCommonLayout && !isAllUAVLayout
                    && resState.mCurrentState != desiredState.mCurrentState);
                bool needTransitionBarrier = isStateChange || isToCommonLayout || isFromCommonLayout;

                // Check for queue transition
                bool needQueueTransition = resState.mCurrentQueue != desiredState.mCurrentQueue;
                RDG_ASSERTION(
                    !(needUAVBarrier && needTransitionBarrier), "Barrier handling logic corrupted, {}", res->GetName());

                if (needUAVBarrier || needTransitionBarrier || needQueueTransition)
                {
                    RDGTransition transition;
                    transition.mResourceIdx   = resIdx;
                    transition.mStateBefore   = resState.mCurrentState;
                    transition.mStateAfter    = desiredState.mCurrentState;
                    transition.mSrcQueue      = resState.mCurrentQueue;
                    transition.mDstQueue      = desiredState.mCurrentQueue;
                    transition.mWholeResource = true;
                    mTransitions.emplace_back(transition);

                    // Update resource state
                    resState.mCurrentState  = desiredState.mCurrentState;
                    resState.mCurrentQueue  = desiredState.mCurrentQueue;
                    resState.mCurrentRWMode = desiredState.mCurrentRWMode;

                    u32 lastPassIdx                 = resState.mLastAccessedPassIndex;
                    resState.mLastAccessedPassIndex = passIndex;

                    // Register transition request to passes
                    u32 transitionIdx = static_cast<u32>(mTransitions.size() - 1);
                    if (lastPassIdx == ~0u)
                    {
                        mTransitionBeginRequests_GraphStart.emplace_back(transitionIdx);
                    }
                    else
                    {
                        mTransitionBeginRequests_Pass[lastPassIdx].emplace_back(transitionIdx);
                    }
                    mTransitionEndRequests_Pass[passIndex].emplace_back(transitionIdx);
                }
                else
                {
                    resState.mCurrentRWMode = desiredState.mCurrentRWMode;
                    resState.mCurrentState  = desiredState.mCurrentState;
                }
            }
            for (auto& succ : passSuccessors[passIndex])
            {
                passIncomingEdges[succ]--;
                if (passIncomingEdges[succ] == 0)
                {
                    pendingPasses.push(succ);
                }
            }
        }
        // Recover initial states for external resources
        for (auto i = 0; i < resourceStates.size(); ++i)
        {
            auto& resState = resourceStates[i];
            auto& res      = mResources[i];
            if (res->IsImported() && resState.mCurrentState != res->GetInitialState())
            {
                RDGTransition transition;
                transition.mResourceIdx   = i;
                transition.mStateBefore   = resState.mCurrentState;
                transition.mStateAfter    = res->GetInitialState();
                transition.mSrcQueue      = resState.mCurrentQueue;
                transition.mDstQueue      = ERDGQueueType::Graphics;
                transition.mWholeResource = true;
                mTransitions.emplace_back(transition);

                u32 lastPassIdx = resState.mLastAccessedPassIndex;

                u32 transitionIdx = static_cast<u32>(mTransitions.size() - 1);
                if (lastPassIdx == ~0u)
                {
                    mTransitionBeginRequests_GraphStart.emplace_back(transitionIdx);
                }
                else
                {
                    mTransitionBeginRequests_Pass[lastPassIdx].emplace_back(transitionIdx);
                }
                mTransitionEndRequests_GraphEnd.emplace_back(transitionIdx);
            }
        }
    }

    void RDGGraphContext::Visualize_DumpCompiledDOTGraph(const String& filepath) const
    {
        Vec<Vec<u32>> passSuccessors(mPasses.size());
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto  passIndex = mValidPassIds[i];
            auto& passDeps  = mCrossQueueDependenciesFull[passIndex];
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (passDeps[q] != ~0u)
                {
                    passSuccessors[passDeps[q]].emplace_back(passIndex);
                }
            }
        }

        Graph g;
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto   passIndex = mValidPassIds[i];
            auto&  pass      = mPasses[passIndex];
            String nodeName  = ToString(pass->GetId()) + pass->GetName();
            g.AddNode(nodeName);
        }
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto   passIndex = mValidPassIds[i];
            auto&  pass      = mPasses[passIndex];
            String srcName   = ToString(pass->GetId()) + pass->GetName();
            for (auto& succ : passSuccessors[passIndex])
            {
                auto&  succPass = mPasses[succ];
                String dstName  = ToString(succPass->GetId()) + succPass->GetName();
                g.AddEdge(srcName, dstName);
            }
        }
        auto dotContent = g.ToDotString();
        WriteTextFile(filepath, dotContent);
    }
    void RDGGraphContext::Visualize_DumpCompiledDOTGraphWithResources(const String& filepath) const
    {
        Graph                 g;

        // Maps to store node names for consistency
        THashMap<u32, String> passIdToNodeName;
        THashMap<u32, String> resourceIdToNodeName;

        // Add all valid passes as nodes
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto   passIndex    = mValidPassIds[i];
            auto&  pass         = mPasses[passIndex];
            String passNodeName = "P" + ToString(pass->GetId()) + "_" + pass->GetName();

            // Clean up name for DOT format (remove spaces, special chars)
            for (auto& c : passNodeName)
            {
                if (!std::isalnum(c) && c != '_')
                    c = '_';
            }

            passIdToNodeName[passIndex] = passNodeName;

            // Add pass node with attributes
            THashMap<String, String> passAttributes;
            passAttributes["shape"] = "box";
            passAttributes["style"] = "filled";

            // Color by queue type
            auto queueType = pass->GetIdentifier().mQueueType;
            switch (queueType)
            {
                case ERDGQueueType::Graphics:
                    passAttributes["fillcolor"] = "lightgreen";
                    break;
                case ERDGQueueType::AsyncCompute:
                    passAttributes["fillcolor"] = "orange";
                    break;
                case ERDGQueueType::AsyncTransfer:
                    passAttributes["fillcolor"] = "pink";
                    break;
            }

            // Create label with pass info
            String queueName        = (queueType == ERDGQueueType::Graphics) ? "GFX"
                       : (queueType == ERDGQueueType::AsyncCompute)          ? "COMP"
                                                                             : "XFER";
            passAttributes["label"] = pass->GetName() + "\\n(" + queueName + ")";

            g.AddNode(passNodeName, passAttributes);
            g.SetNodeShape(passNodeName, EGraphVisualizeNodeShape::Box);
        }

        // Collect all resources used by valid passes
        THashSet<u32>           usedResources;
        THashMap<u32, Vec<u32>> resourceToReadingPasses;
        THashMap<u32, Vec<u32>> resourceToWritingPasses;

        auto fnCheckResourceRWMode = [&](const RDGPassResourceUsage& resUsage, ERDGReadWriteModeFlag mode) -> bool {
            if (resUsage.mDetailedSubresourceTracking)
            {
                for (const auto& [subres, access] : resUsage.mSubresourceAccesses)
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
                return HasFlagBit(resUsage.mOverallAccess.mRWMode, mode);
            }
        };

        // Analyze resource usage
        for (auto i = 0; i < mValidPassIds.size(); ++i)
        {
            auto  passIndex = mValidPassIds[i];
            auto& pass      = mPasses[passIndex];

            for (const auto& resUsage : pass->GetResourceUsages())
            {
                u32 resIdx = resUsage.mResourceIndex;
                usedResources.insert(resIdx);

                // Check if pass reads or writes the resource
                bool isReading = fnCheckResourceRWMode(resUsage, ERDGReadWriteModeFlag::Read);
                bool isWriting = fnCheckResourceRWMode(resUsage, ERDGReadWriteModeFlag::Write);

                if (isReading)
                    resourceToReadingPasses[resIdx].emplace_back(passIndex);
                if (isWriting)
                    resourceToWritingPasses[resIdx].emplace_back(passIndex);
            }
        }

        // Add resource nodes
        for (u32 resIdx : usedResources)
        {
            auto&  res         = mResources[resIdx];
            String resNodeName = "R" + ToString(resIdx) + "_" + res->GetName();

            // Clean up name for DOT format
            for (auto& c : resNodeName)
            {
                if (!std::isalnum(c) && c != '_')
                    c = '_';
            }

            resourceIdToNodeName[resIdx] = resNodeName;

            // Add resource node with attributes
            THashMap<String, String> resAttributes;
            resAttributes["shape"] = "ellipse";
            resAttributes["style"] = "filled";

            // Color by resource type and external status
            if (res->IsImported())
            {
                resAttributes["fillcolor"] = "yellow";
                resAttributes["penwidth"]  = "3";
            }
            else
            {
                resAttributes["fillcolor"] = "lightcyan";
            }

            // Create label with resource info
            String resourceType    = (res->GetType() == ERDGResourceType::Texture) ? "Tex" : "Buf";
            String importStatus    = res->IsImported() ? " (Ext)" : "";
            resAttributes["label"] = res->GetName() + "\\n(" + resourceType + importStatus + ")";

            g.AddNode(resNodeName, resAttributes);
            g.SetNodeShape(resNodeName, EGraphVisualizeNodeShape::Circle);
        }

        // Add Pass -> Resource edges (writes)
        for (const auto& [resIdx, writingPasses] : resourceToWritingPasses)
        {
            String resNodeName = resourceIdToNodeName[resIdx];
            for (u32 passIdx : writingPasses)
            {
                String passNodeName = passIdToNodeName[passIdx];
                g.AddEdge(passNodeName, resNodeName);

                // Add edge attributes for write operations
                g.AddAttribute(passNodeName + "->" + resNodeName, "color", "red");
                g.AddAttribute(passNodeName + "->" + resNodeName, "penwidth", "2");
                g.AddAttribute(passNodeName + "->" + resNodeName, "label", "W");
            }
        }

        // Add Resource -> Pass edges (reads)
        for (const auto& [resIdx, readingPasses] : resourceToReadingPasses)
        {
            String resNodeName = resourceIdToNodeName[resIdx];
            for (u32 passIdx : readingPasses)
            {
                String passNodeName = passIdToNodeName[passIdx];
                g.AddEdge(resNodeName, passNodeName);

                // Add edge attributes for read operations
                g.AddAttribute(resNodeName + "->" + passNodeName, "color", "blue");
                g.AddAttribute(resNodeName + "->" + passNodeName, "penwidth", "1");
                g.AddAttribute(resNodeName + "->" + passNodeName, "label", "R");
            }
        }

        // Generate DOT content and write to file
        String dotContent = g.ToDotString();

        // Add some graph-level styling by modifying the DOT string
        String styledDotContent = dotContent;

        // Insert graph attributes after the opening brace
        size_t insertPos = styledDotContent.find('{');
        if (insertPos != String::npos)
        {
            String graphAttrs = "\n  rankdir=LR;\n";
            graphAttrs += "  node [fontname=\"Arial\", fontsize=10];\n";
            graphAttrs += "  edge [fontname=\"Arial\", fontsize=8];\n";
            graphAttrs += "  labelloc=t;\n\n";

            styledDotContent.insert(insertPos + 1, graphAttrs);
        }

        WriteTextFile(filepath, styledDotContent);
    }
    void RDGGraphContext::Visualize_DumpPhysicalResourcesAllocation() const
    {
        RDG_LOG_INFO("=== RDG Physical Resource Allocation ===");
        for (const auto& buf : mPhysicalResourcePool->mBuffers)
        {
            RDG_LOG_INFO("Buffer Id {}:  Usage Flags {:x}", buf.mInfoId, buf.mResource->GetDesc().mSize,
                (u32)buf.mResource->GetDesc().mUsage);
            RDG_LOG_INFO("  Lifetime: First Use Pass Idx {}, Last Use Pass Idx [GFX {}, COMP {}, XFER {}]",
                buf.mLifetime.mFirstUsePassIndex, buf.mLifetime.mLastUsePassIndex[ERDGQueueType::Graphics],
                buf.mLifetime.mLastUsePassIndex[ERDGQueueType::AsyncCompute],
                buf.mLifetime.mLastUsePassIndex[ERDGQueueType::AsyncTransfer]);
        }

        for (const auto& tex : mPhysicalResourcePool->mTextures)
        {
            RDG_LOG_INFO("Texture Id {}:  Usage Flags {:x}", tex.mInfoId, (u32)tex.mResource->GetDesc().mUsage);
            RDG_LOG_INFO("  Lifetime: First Use Pass Idx {}, Last Use Pass Idx [GFX {}, COMP {}, XFER {}]",
                tex.mLifetime.mFirstUsePassIndex, tex.mLifetime.mLastUsePassIndex[ERDGQueueType::Graphics],
                tex.mLifetime.mLastUsePassIndex[ERDGQueueType::AsyncCompute],
                tex.mLifetime.mLastUsePassIndex[ERDGQueueType::AsyncTransfer]);
        }

        RDG_LOG_INFO("========================================");

        RDG_LOG_INFO("=== RDG Virtual to Physical Resource Mapping ===");
        for (u32 i = 0; i < mResources.size(); ++i)
        {
            auto& res = mResources[i];
            if (!res->IsImported())
            {
                if (res->GetType() == ERDGResourceType::Buffer)
                {
                    auto downcasted = static_cast<RDGBufferResource*>(res.get());
                    RDG_LOG_INFO("Virtual Buffer Resource Idx {} (Name {}) -> Physical Buffer Id {}", i, res->GetName(),
                        downcasted->GetTransientResourceId());
                }
                else if (res->GetType() == ERDGResourceType::Texture)
                {
                    auto downcasted = static_cast<RDGTextureResource*>(res.get());
                    RDG_LOG_INFO("Virtual Texture Resource Idx {} (Name {}) -> Physical Texture Id {}", i,
                        res->GetName(), downcasted->GetTransientResourceId());
                }
            }
        }
        RDG_LOG_INFO("========================================");
    }
    // ===== Resource Allocation =====

    u32 RDGPhysicalResourcePool::RegisterTransientBuffer(const RDGBufferDesc& desc, RDGResourceLifetime lifetime,
        const Vec<Array<u32, ERDGQueueType::Count>>& dependencies)
    {
        for (auto& cachedBuf : mBuffers)
        {
            if (cachedBuf.mResource->GetDesc() == desc)
            {
                auto lifetimeOverlap = cachedBuf.mLifetime.CheckOverlappingWith(lifetime, dependencies);
                if (lifetimeOverlap != ERDGLifetimeOverlapTestResult::Overlap)
                {
                    cachedBuf.mLifetime.ExtendLifetime(lifetime);
                    return cachedBuf.mInfoId;
                }
            }
        }
        // Otherwise create a new buffer
        ResourceInfo info;
        info.mType  = ERDGResourceType::Buffer;
        info.mIndex = static_cast<u32>(mBuffers.size());

        TTrackedResource<RDGPhysicalBuffer> newBuffer;
        mBuffers.push_back(std::move(newBuffer));

        auto& newBuf     = mBuffers.back();
        newBuf.mInfoId   = SizeCast<u32>(mResourceInfos.size());
        newBuf.mResource = MakeOwner<RDGPhysicalBuffer>(desc);
        newBuf.mLifetime = lifetime;

        mResourceInfos.push_back(info);
        return newBuf.mInfoId;
    }
    u32 RDGPhysicalResourcePool::RegisterTransientTexture(const RDGTextureDesc& desc, RDGResourceLifetime lifetime,
        const Vec<Array<u32, ERDGQueueType::Count>>& dependencies)
    {
        for (auto& cachedTex : mTextures)
        {
            if (cachedTex.mResource->GetDesc() == desc)
            {
                auto lifetimeOverlap = cachedTex.mLifetime.CheckOverlappingWith(lifetime, dependencies);
                if (lifetimeOverlap != ERDGLifetimeOverlapTestResult::Overlap)
                {
                    cachedTex.mLifetime.ExtendLifetime(lifetime);
                    return cachedTex.mInfoId;
                }
            }
        }
        // Otherwise create a new texture
        ResourceInfo info;
        info.mType  = ERDGResourceType::Texture;
        info.mIndex = static_cast<u32>(mTextures.size());

        TTrackedResource<RDGPhysicalTexture> newTexture;
        mTextures.push_back(std::move(newTexture));

        auto& newTex     = mTextures.back();
        newTex.mInfoId   = SizeCast<u32>(mResourceInfos.size());
        newTex.mResource = MakeOwner<RDGPhysicalTexture>(desc);
        newTex.mLifetime = lifetime;

        mResourceInfos.push_back(info);
        return newTex.mInfoId;
    }
} // namespace Ifrit::Runtime::RenderCore::RDG