#include "ifrit/vkrhi2/adapter/Queue.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/vkrhi2/adapter/CommandList.h"

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

    IFRIT_VKRHI2_API VA_Queue::VA_Queue(VA_Device* device, u32 familyIndex)
    {
        vkGetDeviceQueue(device->GetVulkanDevice(), familyIndex, 0, &mQueue);
        mSemaphore   = MakeOwner<VA_TimelineSemaphore>(device);
        mFamilyIndex = familyIndex;
    }

    IFRIT_VKRHI2_API     VA_Queue::~VA_Queue() {}

    IFRIT_VKRHI2_API u32 VA_Queue::GetFamilyIndex() { return mFamilyIndex; }

    IFRIT_VKRHI2_API Ref<VA_CommandSubmission> VA_Queue::SubmitCommandNative(VA_CommandListNative* cmd,
        Vec<Ref<VA_CommandSubmission>> toWait, VkFence fenceToSignal, VkSemaphore swapchainSemaToSignal)
    {
        IF_LOG_ASSERTION("VA_Queue", cmd->GetState() == EVA_CommandListNativeState::ReadyToSubmit,
            "command buffer recording is not finished");

        auto                      desiredToSignalVal = mSemaphore->FetchAndAdd();

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
        signalSemaphores.push_back(mSemaphore->GetSemaphore());

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
        }
        VA_AssertResult(vkQueueSubmit(mQueue, 1, &submitInfo, vfence), "Failed to submit command buffer");

        Ref<VA_CommandSubmission> ret = MakeRef<VA_CommandSubmission>();
        ret->mSemaphore               = mSemaphore->GetSemaphore();
        ret->mValue                   = desiredToSignalVal;
        ret->mWaitStage               = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        return ret;
    }
} // namespace Ifrit::RHI::VulkanRHI2