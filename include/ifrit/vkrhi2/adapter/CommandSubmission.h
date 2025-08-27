#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"

#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{

    class VA_CommandListNative;

    enum class EVA_CommandTaskState
    {
        Invalid,
        Wait,
        Execute,
        Recorded,
        Submitted,
    };

    class VA_CommandSubmission : public RHI::RhiTaskSubmission
    {
    public:
        VkSemaphore mSemaphore;
        VkFence     mFence;
        u64         mValue;
        VkFlags     mWaitStage            = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        bool        mIsSwapchainSemaphore = false;
    };

    struct VA_CommandTask
    {
        EVA_CommandTaskState           mState = EVA_CommandTaskState::Invalid;
        VA_CommandListNative*          mCmd   = nullptr;
        Vec<Ref<VA_CommandSubmission>> mToWait;
        Ref<VA_CommandSubmission>      mToSignal;

        VkFence                        mExternalFence;
        VkSemaphore                    mExternalSemaphore;
    };

} // namespace Ifrit::RHI::VulkanRHI2