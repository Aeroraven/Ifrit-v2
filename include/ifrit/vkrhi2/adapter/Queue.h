#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/vkrhi2/adapter/CommandSubmission.h"
#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{

    class VA_CommandListNative;
    class VA_Device;
    class VA_CommandListPool;

    class IFRIT_VKRHI2_API VA_TimelineSemaphore
    {
    public:
        VA_TimelineSemaphore(VA_Device* ctx);
        ~VA_TimelineSemaphore();
        inline VkSemaphore GetSemaphore() const { return mSemaphore; }
        inline u64         FetchAndAdd() { return mRecordedCounter.fetch_add(1u, std::memory_order::seq_cst) + 1; }

    private:
        VA_Device*  mContext;
        VkSemaphore mSemaphore;
        Atomic<u64> mRecordedCounter = 0;
    };

    struct VA_QueueInternal;
    class IFRIT_VKRHI2_API VA_Queue
    {
    public:
        VA_Queue(VA_Device* device, ERhiCommandListPipelineType pipeType, u32 familyIndex);
        ~VA_Queue();

        u32                 GetFamilyIndex();

        void                EnqueueCommandTask(Ref<VA_CommandTask> task);
        VA_CommandListPool* AcquireCommandPool();
        void                ReleaseCommandPool(VA_CommandListPool* pool);
        void                ProcessQueuedTasks();

    private:
        void                      SubmitCommandNative(VA_CommandListNative* cmd, Vec<Ref<VA_CommandSubmission>> toWait,
                                 VkFence fenceToSignal, VkSemaphore swapchainSemaToSignal, Ref<VA_CommandSubmission> desiredToSignalInfo);
        Ref<VA_CommandSubmission> PrepareSubmissionInfo();

    private:
        VA_QueueInternal* mInternal;
    };

} // namespace Ifrit::RHI::VulkanRHI2