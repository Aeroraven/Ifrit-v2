#pragma once

#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/adapter/Device.h"

namespace Ifrit::RHI::VulkanRHI2
{

    class VA_CommandListNative;

    class VA_CommandSubmission : public RHI::RhiTaskSubmission
    {
    public:
        VkSemaphore mSemaphore;
        VkFence     mFence;
        u64         mValue;
        VkFlags     mWaitStage            = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        bool        mIsSwapchainSemaphore = false;
    };

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

    class IFRIT_VKRHI2_API VA_Queue
    {
    public:
        VA_Queue(VA_Device* device, u32 familyIndex);
        ~VA_Queue();

        u32                       GetFamilyIndex();
        Ref<VA_CommandSubmission> SubmitCommandNative(VA_CommandListNative* cmd, Vec<Ref<VA_CommandSubmission>> toWait,
            VkFence fenceToSignal, VkSemaphore swapchainSemaToSignal);

    private:
        VkQueue                     mQueue;
        Owner<VA_TimelineSemaphore> mSemaphore;
        u32                         mFamilyIndex;
    };

} // namespace Ifrit::RHI::VulkanRHI2