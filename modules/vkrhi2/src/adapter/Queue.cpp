#include "ifrit/vkrhi2/adapter/Queue.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/vkrhi2/adapter/CommandBuffer.h"

namespace Ifrit::RHI::VulkanRHI2
{
    // ===== Semaphore =====
    IFRIT_APIDECL VA_TimelineSemaphore::VA_TimelineSemaphore(VA_Device* ctx) : mContext(ctx)
    {
        VkSemaphoreTypeCreateInfo timelineCI{};
        timelineCI.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timelineCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineCI.initialValue  = 0;
        VkSemaphoreCreateInfo semaphoreCI{};
        semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreCI.pNext = &timelineCI;
        VA_AssertResult(vkCreateSemaphore(mContext->GetVulkanDevice(), &semaphoreCI, nullptr, &mSemaphore),
            "Failed to create timeline semaphore");
    }

    IFRIT_APIDECL VA_TimelineSemaphore::~VA_TimelineSemaphore()
    {
        vkDestroySemaphore(mContext->GetVulkanDevice(), mSemaphore, nullptr);
    }

    // ===== Queue =====
    struct VA_QueueInternal : public NonCopyable
    {
        VA_Device*                     mDevice = nullptr;
        VkQueue                        mQueue;
        Owner<VA_TimelineSemaphore>    mSemaphore;
        u32                            mFamilyIndex;
        Mutex                          mSubmitMutex;
        ERhiCommandListPipelineType    mPipeType;

        Vec<Owner<VA_CommandListPool>> mAcquiredCommandPools;
        Vec<VA_CommandListPool*>       mAvailableCommandPools;

        Vec<Ref<VA_CommandTask>>       mQueuedTasks;
    };

    IFRIT_VKRHI2_API VA_Queue::VA_Queue(VA_Device* device, ERhiCommandListPipelineType pipeType, u32 familyIndex)
    {
        mInternal = new VA_QueueInternal();
        vkGetDeviceQueue(device->GetVulkanDevice(), familyIndex, 0, &mInternal->mQueue);
        mInternal->mSemaphore   = MakeOwner<VA_TimelineSemaphore>(device);
        mInternal->mFamilyIndex = familyIndex;
        mInternal->mPipeType    = pipeType;
        mInternal->mDevice      = device;
    }

    IFRIT_VKRHI2_API VA_Queue::~VA_Queue()
    {

        IF_LOG_INFO("VA_Queue", "Deleted Queue");
        delete mInternal;
    }

    IFRIT_VKRHI2_API u32  VA_Queue::GetFamilyIndex() { return mInternal->mFamilyIndex; }

    IFRIT_VKRHI2_API void VA_Queue::SubmitCommandNative(VA_CommandListNative* cmd,
        Vec<Ref<VA_CommandSubmission>> toWait, VkFence fenceToSignal, VkSemaphore swapchainSemaToSignal,
        Ref<VA_CommandSubmission> desiredToSignalInfo)
    {

        IF_LOG_ASSERTION("VA_Queue", cmd->GetState() == EVA_CommandListNativeState::ReadyToSubmit,
            "command buffer recording is not finished");

        auto                      desiredToSignalVal = desiredToSignalInfo->mValue;

        Vec<VkSemaphore>          waitSemaphoreHandles;
        Vec<uint64_t>             waitValues;
        Vec<VkPipelineStageFlags> waitStages;

        for (int i = 0; i < toWait.size(); i++)
        {
            waitSemaphoreHandles.push_back(toWait[i]->mSemaphore);
            waitValues.push_back(toWait[i]->mValue);
            waitStages.push_back(toWait[i]->mWaitStage);
        }

        Vec<u64>         signalValues;
        Vec<VkSemaphore> signalSemaphores;
        signalValues.push_back(desiredToSignalVal);
        signalSemaphores.push_back(mInternal->mSemaphore->GetSemaphore());

        if (swapchainSemaToSignal)
        {
            signalValues.push_back(0);
            signalSemaphores.push_back(swapchainSemaToSignal);
        }

        VkCommandBuffer commandBuffer = cmd->GetCmd();

        VkSubmitInfo    submitInfo{};
        submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount   = SizeCast<int>(waitSemaphoreHandles.size());
        submitInfo.pWaitSemaphores      = waitSemaphoreHandles.data();
        submitInfo.pWaitDstStageMask    = waitStages.data();
        submitInfo.commandBufferCount   = 1;
        submitInfo.pCommandBuffers      = &commandBuffer;
        submitInfo.signalSemaphoreCount = SizeCast<int>(signalSemaphores.size());
        submitInfo.pSignalSemaphores    = signalSemaphores.data();

        VkTimelineSemaphoreSubmitInfo timelineInfo{};
        timelineInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineInfo.waitSemaphoreValueCount   = SizeCast<int>(waitValues.size());
        timelineInfo.pWaitSemaphoreValues      = waitValues.data();
        timelineInfo.signalSemaphoreValueCount = SizeCast<int>(signalValues.size());
        timelineInfo.pSignalSemaphoreValues    = signalValues.data();

        submitInfo.pNext = &timelineInfo;
        VkFence vfence   = VK_NULL_HANDLE;
        if (fenceToSignal)
        {
            vfence = fenceToSignal;
            // IF_LOG_INFO("VA_Queue", "Submitting command buffer with fence {}", (void*)fenceToSignal);
        }
        VA_AssertResult(vkQueueSubmit(mInternal->mQueue, 1, &submitInfo, vfence), "Failed to submit command buffer");

        cmd->SetSubmitState();
    }

    IFRIT_VKRHI2_API VA_CommandListPool* VA_Queue::AcquireCommandPool()
    {
        ScopedLock lock(mInternal->mSubmitMutex);

        if (!mInternal->mAvailableCommandPools.empty())
        {
            auto pool = std::move(mInternal->mAvailableCommandPools.back());
            mInternal->mAvailableCommandPools.pop_back();
            return pool;
        }
        auto pool = MakeOwner<VA_CommandListPool>(
            mInternal->mDevice, mInternal->mPipeType, mInternal->mQueue, mInternal->mFamilyIndex);
        mInternal->mAcquiredCommandPools.push_back(std::move(pool));
        return mInternal->mAcquiredCommandPools.back().get();
    }
    IFRIT_VKRHI2_API void VA_Queue::ReleaseCommandPool(VA_CommandListPool* pool)
    {
        ScopedLock lock(mInternal->mSubmitMutex);
        mInternal->mAvailableCommandPools.push_back(pool);
    }

    IFRIT_VKRHI2_API Ref<VA_CommandSubmission> VA_Queue::PrepareSubmissionInfo()
    {
        Ref<VA_CommandSubmission> ret = MakeRef<VA_CommandSubmission>();
        ret->mSemaphore               = mInternal->mSemaphore->GetSemaphore();
        ret->mValue                   = mInternal->mSemaphore->FetchAndAdd();
        ret->mWaitStage               = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        return ret;
    }

    IFRIT_VKRHI2_API void VA_Queue::ProcessQueuedTasks()
    {
        ScopedLock lock(mInternal->mSubmitMutex);
        for (auto& task : mInternal->mQueuedTasks)
        {
            if (task->mType == EVA_CommandTaskType::CPUWait)
            {
                task->SetComplete();
                continue;
            }
            else if (task->mType == EVA_CommandTaskType::CommandSubmission)
            {
                if (task->mState != EVA_CommandTaskState::Recorded)
                {
                    IF_LOG_CRITICAL("VA_Queue", "Task is not ready to submit");
                }
                // Submit
                SubmitCommandNative(
                    task->mCmd, task->mToWait, task->mExternalFence, task->mExternalSemaphore, task->mToSignal);
                task->mState = EVA_CommandTaskState::Submitted;

                for (auto& cb : task->mCompletionCallbacks)
                {
                    cb();
                }
            }
        }
        mInternal->mQueuedTasks.clear();
    }

    IFRIT_VKRHI2_API void VA_Queue::EnqueueCommandTask(Ref<VA_CommandTask> task)
    {
        ScopedLock lock(mInternal->mSubmitMutex);
        if (task->mType == EVA_CommandTaskType::CPUWait)
        {
            mInternal->mQueuedTasks.push_back(task);
            return;
        }

        if (task->mState != EVA_CommandTaskState::Recorded)
        {
            IF_LOG_CRITICAL("VA_Queue", "Task is not ready to submit");
        }
        task->mToSignal = PrepareSubmissionInfo();
        mInternal->mQueuedTasks.push_back(task);
    }

    IFRIT_VKRHI2_API VkQueue VA_Queue::GetNativeQueue() const { return mInternal->mQueue; }

    IFRIT_VKRHI2_API void    VA_Queue::RecycleCmdLists()
    {
        ScopedLock lock(mInternal->mSubmitMutex);
        for (auto& pool : mInternal->mAcquiredCommandPools)
        {
            pool->RecycleCommandBuffers();
        }
    }

} // namespace Ifrit::RHI::VulkanRHI2