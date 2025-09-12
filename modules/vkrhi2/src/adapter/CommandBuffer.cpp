#include "ifrit/vkrhi2/adapter/CommandBuffer.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/core/console/ConsoleObject.h"

namespace Ifrit::RHI::VulkanRHI2
{
    static TConsoleVariable<u32> cvVulkanCommandListTTL(
        "cv.VulkanRHI2.CommandListTTL", 3, "Vulkan Command List TTL", CVF_ReadOnly);

    // ===== Command Buffer =====
    struct VA_CommandListNativeInternal
    {
        VkCommandBuffer            mCmd             = VK_NULL_HANDLE;
        EVA_CommandListNativeState mState           = EVA_CommandListNativeState::Undefined;
        u64                        mSubmitTimestamp = 0;
        VA_Device*                 mDevice          = nullptr;
    };

    IFRIT_APIDECL VA_CommandListNative::VA_CommandListNative(VkCommandBuffer cmd, VA_Device* device)
    {

        mInternal          = new VA_CommandListNativeInternal();
        mInternal->mCmd    = cmd;
        mInternal->mState  = EVA_CommandListNativeState::ReadyToBegin;
        mInternal->mDevice = device;
    }

    IFRIT_APIDECL                            VA_CommandListNative::~VA_CommandListNative() { delete mInternal; }

    IFRIT_APIDECL VkCommandBuffer            VA_CommandListNative::GetCmd() { return mInternal->mCmd; }

    IFRIT_APIDECL EVA_CommandListNativeState VA_CommandListNative::GetState() { return mInternal->mState; }

    IFRIT_APIDECL void                       VA_CommandListNative::Begin()
    {
        if (!(mInternal->mState == EVA_CommandListNativeState::ReadyToBegin))
        {
            IF_LOG_WARNING("VA_CommandListNative", "Command buffer not in ReadyToBegin state, forcing to ReadyToBegin");
            IF_LOG_WARNING("VA_CommandListNative", "Current state: {}", static_cast<u32>(mInternal->mState));
        }
        IF_LOG_ASSERTION(
            "VA_CommandListNative", mInternal->mState == EVA_CommandListNativeState::ReadyToBegin, "State corrupted");
        mInternal->mState = EVA_CommandListNativeState::Recording;

        VkCommandBufferBeginInfo CmdBufBeginInfo{};
        CmdBufBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        CmdBufBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VA_AssertResult(vkBeginCommandBuffer(mInternal->mCmd, &CmdBufBeginInfo), "Failed to begin command buffer");

        // todo: attach bindless descriptors
    }

    IFRIT_APIDECL u64  VA_CommandListNative::GetSubmitTimestamp() const { return mInternal->mSubmitTimestamp; }

    IFRIT_APIDECL void VA_CommandListNative::ForceSetToReadyState()
    {
        mInternal->mState = EVA_CommandListNativeState::ReadyToBegin;
    }

    IFRIT_APIDECL void VA_CommandListNative::SetSubmitState()
    {
        IF_LOG_ASSERTION("VA_CommandListNative", mInternal->mState == EVA_CommandListNativeState::ReadyToSubmit,
            "State corrupted when setting to submit state");
        mInternal->mState = EVA_CommandListNativeState::Submitted;
    }

    IFRIT_APIDECL void VA_CommandListNative::End()
    {
        IF_LOG_ASSERTION(
            "VA_CommandListNative", mInternal->mState == EVA_CommandListNativeState::Recording, "State corrupted");
        mInternal->mState = EVA_CommandListNativeState::ReadyToSubmit;

        VA_AssertResult(vkEndCommandBuffer(mInternal->mCmd), "Failed to end command buffer");
        mInternal->mSubmitTimestamp = mInternal->mDevice->GetFrameId();
    }

    // ===== Command Pool =====
    struct VA_CommandListPoolInternal : public NonCopyable
    {
        VA_Device*                       mDevice  = nullptr;
        ERhiCommandListPipelineType      mType    = ERhiCommandListPipelineType::Invalid;
        VkCommandPool                    mCmdPool = VK_NULL_HANDLE;
        VkQueue                          mQueue   = VK_NULL_HANDLE;

        Vec<Owner<VA_CommandListNative>> mAllocatedCommandBuffers;
        Vec<Owner<VA_CommandListNative>> mFreeCommandBuffers;

        Mutex                            mMutex;
    };

    IFRIT_APIDECL VA_CommandListPool::VA_CommandListPool(
        VA_Device* device, ERhiCommandListPipelineType type, VkQueue queue, u32 familyIndex)
    {
        mInternal          = new VA_CommandListPoolInternal();
        mInternal->mDevice = device;
        mInternal->mType   = type;
        mInternal->mQueue  = queue;

        VkCommandPoolCreateInfo CmdPoolCreateInfo{};
        CmdPoolCreateInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        CmdPoolCreateInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        CmdPoolCreateInfo.queueFamilyIndex = familyIndex;
        VA_AssertResult(
            vkCreateCommandPool(device->GetVulkanDevice(), &CmdPoolCreateInfo, nullptr, &mInternal->mCmdPool),
            "Failed to create command pool");

        IF_LOG_DEBUG("VA_CommandListPool", "Created command pool for family index {}", familyIndex);
    }

    IFRIT_APIDECL void VA_CommandListPool::RecycleCommandBuffers()
    {
        ScopedLock lock(mInternal->mMutex);
        auto       numFreeCmds = 0;

        for (i32 i = SizeCast<i32>(mInternal->mAllocatedCommandBuffers.size()) - 1; i >= 0; i--)
        {
            auto& cmd             = mInternal->mAllocatedCommandBuffers[i];
            auto  state           = cmd->GetState();
            auto  submitTimestamp = cmd->GetSubmitTimestamp();

            if ((state == EVA_CommandListNativeState::Submitted)
                && mInternal->mDevice->GetFrameId() - submitTimestamp > cvVulkanCommandListTTL.GetValue())
            {
                cmd->ForceSetToReadyState();
                auto lastNonFreeId = mInternal->mAllocatedCommandBuffers.size() - 1 - numFreeCmds;
                if (i != lastNonFreeId)
                {
                    std::swap(
                        mInternal->mAllocatedCommandBuffers[i], mInternal->mAllocatedCommandBuffers[lastNonFreeId]);
                }
                numFreeCmds++;
            }
        }

        if (numFreeCmds > 0)
        {
            // IF_LOG_DEBUG("VA_CommandListPool", "Recycling {} command buffers", numFreeCmds);
            auto startIt = mInternal->mAllocatedCommandBuffers.end() - numFreeCmds;

            // Revalidate states for cmds being recycled
            for (auto it = startIt; it != mInternal->mAllocatedCommandBuffers.end(); it++)
            {
                auto state = (*it)->GetState();
                IF_LOG_ASSERTION("VA_CommandListPool", state == EVA_CommandListNativeState::ReadyToBegin,
                    "State corrupted when recycling command buffer, state: {}", static_cast<u32>(state));
                mInternal->mFreeCommandBuffers.push_back(std::move(*it));
            }
            mInternal->mAllocatedCommandBuffers.erase(startIt, mInternal->mAllocatedCommandBuffers.end());
        }
    }

    IFRIT_APIDECL VA_CommandListPool::~VA_CommandListPool()
    {
        // delete command pool
        vkDestroyCommandPool(mInternal->mDevice->GetVulkanDevice(), mInternal->mCmdPool, nullptr);
        delete mInternal;
    }

    IFRIT_APIDECL VA_CommandListNative* VA_CommandListPool::AllocateCommandBuffer()
    {
        ScopedLock                  lock(mInternal->mMutex);

        Owner<VA_CommandListNative> cmd;
        if (!mInternal->mFreeCommandBuffers.empty())
        {
            cmd = std::move(mInternal->mFreeCommandBuffers.back());
            mInternal->mFreeCommandBuffers.pop_back();
        }
        else
        {
            VkCommandBufferAllocateInfo CmdBufAllocInfo{};
            CmdBufAllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            CmdBufAllocInfo.commandPool        = mInternal->mCmdPool;
            CmdBufAllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            CmdBufAllocInfo.commandBufferCount = 1;

            VkCommandBuffer cmdVk = VK_NULL_HANDLE;
            VA_AssertResult(vkAllocateCommandBuffers(mInternal->mDevice->GetVulkanDevice(), &CmdBufAllocInfo, &cmdVk),
                "Failed to allocate command buffer");

            cmd = MakeOwner<VA_CommandListNative>(cmdVk, mInternal->mDevice);
            IF_LOG_ASSERTION("VA_CommandListPool", cmd != nullptr, "Failed to create command list wrapper");
        }

        cmd->GetState(); // just to make sure the state is valid
        mInternal->mAllocatedCommandBuffers.push_back(std::move(cmd));
        auto retCmd = mInternal->mAllocatedCommandBuffers.back().get();
        return retCmd;
    }

} // namespace Ifrit::RHI::VulkanRHI2