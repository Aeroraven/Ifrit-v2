#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/core/tasks/TaskScheduler.h"
#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{

    class VA_CommandListNative;
    class VA_Device;

    enum class EVA_CommandTaskState
    {
        Invalid,
        Wait,
        Execute,
        Recorded,
        Submitted,
    };

    enum class EVA_CommandTaskType
    {
        CommandSubmission,
        CPUWait,
    };

    class VA_CommandSubmission : public RHI::RhiTaskSubmission
    {
    public:
        VkSemaphore mSemaphore            = VK_NULL_HANDLE;
        VkFence     mFence                = VK_NULL_HANDLE;
        u64         mValue                = 0;
        VkFlags     mWaitStage            = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        bool        mIsSwapchainSemaphore = false;
    };

    struct VA_CommandTask
    {
        EVA_CommandTaskType            mType  = EVA_CommandTaskType::CommandSubmission;
        EVA_CommandTaskState           mState = EVA_CommandTaskState::Invalid;
        VA_CommandListNative*          mCmd   = nullptr;
        Vec<Ref<VA_CommandSubmission>> mToWait;
        Ref<VA_CommandSubmission>      mToSignal;

        VkFence                        mExternalFence;
        VkSemaphore                    mExternalSemaphore;

        Vec<Fn<void()>>                mCompletionCallbacks;

        Mutex                          mCpuWaitLock;
        ConditionalVariable            mCpuWaitCond;
        bool                           mCpuWaitDone = false;

        void                           SetComplete()
        {
            if (mType == EVA_CommandTaskType::CPUWait)
            {
                UniqueLock lock(mCpuWaitLock);
                mCpuWaitDone = true;
                mCpuWaitCond.notify_all();
            }
        }
        void Wait()
        {
            if (mType == EVA_CommandTaskType::CPUWait)
            {
                UniqueLock lock(mCpuWaitLock);
                mCpuWaitCond.wait(lock, [this]() { return mCpuWaitDone; });
            }
        }

        static Ref<VA_CommandTask> CreateCpuWaitTask()
        {
            auto ret       = MakeRef<VA_CommandTask>();
            ret->mType     = EVA_CommandTaskType::CPUWait;
            ret->mState    = EVA_CommandTaskState::Submitted;
            ret->mCmd      = nullptr;
            ret->mToSignal = nullptr;
            return ret;
        }
    };

    struct VA_QueueSubmissionThreadInternal;
    class VA_QueueSubmissionThread : public Task::TaskWorker
    {
    public:
        VA_QueueSubmissionThread(Task::TaskScheduler* scheduler, VA_Device* device, u32 id);
        ~VA_QueueSubmissionThread();

        virtual void RunUnique() override;

    private:
        VA_QueueSubmissionThreadInternal* mData = nullptr;
    };

} // namespace Ifrit::RHI::VulkanRHI2