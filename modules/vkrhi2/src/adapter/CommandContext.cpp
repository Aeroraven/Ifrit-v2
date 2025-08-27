#include "ifrit/vkrhi2/adapter/CommandContext.h"
#include "ifrit/vkrhi2/adapter/CommandBuffer.h"
#include "ifrit/vkrhi2/adapter/CommandSubmission.h"

namespace Ifrit::RHI::VulkanRHI2
{
    struct VA_CommandListNativeInternal
    {
        VA_Device*                     mDevice;
        VA_Queue*                      mQueue;
        VA_CommandListPool*            mPool             = nullptr;
        VA_CommandListContext*         mImmediateContext = nullptr;

        Vec<Ref<VA_CommandTask>>       mCurrentSubTasks;
        EVA_CommandTaskState           mCurrentState = EVA_CommandTaskState::Invalid;

        Vec<Ref<VA_CommandSubmission>> mExternalToWait;
        VkFence                        mExternalFence = VK_NULL_HANDLE;
        VkSemaphore                    mExternalSema  = VK_NULL_HANDLE;
    };

    IFRIT_VKRHI2_API VA_CommandListContext::VA_CommandListContext(
        VA_Device* device, VA_Queue* queue, VA_CommandListContext* immediateContext)
    {
        mInternal                    = new VA_CommandListNativeInternal();
        mInternal->mDevice           = device;
        mInternal->mQueue            = queue;
        mInternal->mPool             = queue->AcquireCommandPool();
        mInternal->mImmediateContext = immediateContext;
    };

    IFRIT_VKRHI2_API VA_CommandListContext::~VA_CommandListContext()
    {
        if (mInternal->mPool && mInternal->mQueue)
        {
            mInternal->mQueue->ReleaseCommandPool(mInternal->mPool);
            mInternal->mPool = nullptr;
        }
        delete mInternal;
        mInternal = nullptr;
    }

    IFRIT_VKRHI2_API void VA_CommandListContext::NewTaskSection()
    {
        EndTaskSection();
        mInternal->mCurrentState = EVA_CommandTaskState::Wait;
        mInternal->mCurrentSubTasks.push_back(MakeRef<VA_CommandTask>());
    }

    IFRIT_VKRHI2_API void VA_CommandListContext::EndTaskSection()
    {
        if (mInternal->mCurrentSubTasks.size())
        {
            auto& lastTask = mInternal->mCurrentSubTasks.back();
            IF_LOG_ASSERTION(
                "VA_CommandListContext", lastTask->mState == mInternal->mCurrentState, "Task state corrupted");
            IF_LOG_ASSERTION("VA_CommandListContext", lastTask->mState == EVA_CommandTaskState::Execute,
                "Task not ended properly, missing EndTaskSection call?");
            if (lastTask->mState == EVA_CommandTaskState::Execute)
            {
                lastTask->mState = EVA_CommandTaskState::Recorded;
            }
            lastTask->mCmd->End();
            mInternal->mCurrentState = EVA_CommandTaskState::Recorded;
        }
    }

    IFRIT_VKRHI2_API VA_CommandTask* VA_CommandListContext::GetTaskSection(EVA_CommandTaskState desiredState)
    {
        if (mInternal->mCurrentSubTasks.empty() || mInternal->mCurrentState > desiredState)
        {
            NewTaskSection();
        }
        auto& lastTask           = mInternal->mCurrentSubTasks.back();
        lastTask->mState         = desiredState;
        mInternal->mCurrentState = desiredState;
        return lastTask.get();
    }

    IFRIT_VKRHI2_API VA_CommandListNative* VA_CommandListContext::GetCommandBuffer()
    {
        auto subTask = GetTaskSection(EVA_CommandTaskState::Execute);
        if (subTask->mCmd == nullptr)
        {
            auto pool    = mInternal->mPool;
            auto poolCmd = pool->AllocateCommandBuffer();
            poolCmd->Begin();
            subTask->mCmd = poolCmd;
        }
        return subTask->mCmd;
    }

    IFRIT_VKRHI2_API void VA_CommandListContext::RegisterDependencies(Vec<Ref<VA_CommandSubmission>> toWait)
    {
        for (auto& sub : toWait)
        {
            mInternal->mExternalToWait.push_back(sub);
        }
    }
    IFRIT_VKRHI2_API void VA_CommandListContext::RegisterExternalDependencies(VkFence extFence, VkSemaphore extSema)
    {
        mInternal->mExternalFence = extFence;
        mInternal->mExternalSema  = extSema;
    }

    IFRIT_VKRHI2_API Ref<VA_CommandSubmission> VA_CommandListContext::FlushAllTaskSections()
    {
        Ref<VA_CommandSubmission> ret   = nullptr;
        auto                      queue = mInternal->mQueue;
        EndTaskSection();
        if (mInternal->mCurrentSubTasks.size() == 0)
        {
            IF_LOG_WARNING("VA_CommandListContext", "No tasks to flush");
            return ret;
        }

        mInternal->mCurrentSubTasks[0]->mToWait                = mInternal->mExternalToWait;
        mInternal->mCurrentSubTasks.back()->mExternalFence     = mInternal->mExternalFence;
        mInternal->mCurrentSubTasks.back()->mExternalSemaphore = mInternal->mExternalSema;

        mInternal->mExternalToWait.clear();
        mInternal->mExternalFence = VK_NULL_HANDLE;
        mInternal->mExternalSema  = VK_NULL_HANDLE;

        for (int i = 0; i < mInternal->mCurrentSubTasks.size(); i++)
        {
            auto& subTask = mInternal->mCurrentSubTasks[i];
            if (i != 0)
            {
                subTask->mToWait.push_back(mInternal->mCurrentSubTasks[i - 1]->mToSignal);
            }
            queue->EnqueueCommandTask(subTask);
        }
        ret = mInternal->mCurrentSubTasks.back()->mToSignal;
        mInternal->mCurrentSubTasks.clear();
        mInternal->mCurrentState = EVA_CommandTaskState::Invalid;
        return ret;
    }

    // Public API
    IFRIT_VKRHI2_API Ref<VA_CommandSubmission> VA_CommandListContext::FlushCommands() { return FlushAllTaskSections(); }

} // namespace Ifrit::RHI::VulkanRHI2