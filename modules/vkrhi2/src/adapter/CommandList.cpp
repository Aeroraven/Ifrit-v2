#include "ifrit/vkrhi2/adapter/CommandList.h"
#include "ifrit.internal/vkrhi2/adapter/CommandListUtils.h"
#include "ifrit/vkrhi2/adapter/Device.h"

namespace Ifrit::RHI::VulkanRHI2
{
    // ===== Command Buffer =====
    struct VA_CommandListNativeInternal
    {
        VkCommandBuffer            mCmd   = VK_NULL_HANDLE;
        EVA_CommandListNativeState mState = EVA_CommandListNativeState::Undefined;
    };

    IFRIT_APIDECL VA_CommandListNative::VA_CommandListNative(VkCommandBuffer cmd)
    {
        mInternal         = new VA_CommandListNativeInternal();
        mInternal->mCmd   = cmd;
        mInternal->mState = EVA_CommandListNativeState::ReadyToBegin;
    }

    IFRIT_APIDECL                            VA_CommandListNative::~VA_CommandListNative() { delete mInternal; }

    IFRIT_APIDECL VkCommandBuffer            VA_CommandListNative::GetCmd() { return mInternal->mCmd; }

    IFRIT_APIDECL EVA_CommandListNativeState VA_CommandListNative::GetState() { return mInternal->mState; }

    IFRIT_APIDECL void                       VA_CommandListNative::Begin()
    {
        IF_LOG_ASSERTION(
            "VA_CommandListNative", mInternal->mState == EVA_CommandListNativeState::ReadyToBegin, "State corrupted");
        mInternal->mState = EVA_CommandListNativeState::Recording;

        VkCommandBufferBeginInfo CmdBufBeginInfo{};
        CmdBufBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        CmdBufBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VA_AssertResult(vkBeginCommandBuffer(mInternal->mCmd, &CmdBufBeginInfo), "Failed to begin command buffer");

        // todo: attach bindless descriptors
    }

    IFRIT_APIDECL void VA_CommandListNative::End()
    {
        IF_LOG_ASSERTION(
            "VA_CommandListNative", mInternal->mState == EVA_CommandListNativeState::Recording, "State corrupted");
        mInternal->mState = EVA_CommandListNativeState::ReadyToSubmit;

        VA_AssertResult(vkEndCommandBuffer(mInternal->mCmd), "Failed to end command buffer");
    }

    // ===== Command Pool =====
    class VA_CommandPool
    {
    public:
        VA_CommandPool(VA_Device* device, VA_Queue* queue)
        {
            VkCommandPoolCreateInfo poolCI{};
            poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolCI.queueFamilyIndex = queue->GetFamilyIndex();
            poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            VA_AssertResult(vkCreateCommandPool(device->GetVulkanDevice(), &poolCI, nullptr, &mCmdPool),
                "Failed to create command pool");

            mContext = device;
        }
        ~VA_CommandPool() { vkDestroyCommandPool(mContext->GetVulkanDevice(), mCmdPool, nullptr); }
        void EnqueueInFlightCommandBuffer(Owner<VA_CommandListNative> cmdBuf) { mInFlightCmdList.push_back(cmdBuf); }
        void ResetCommandPool()
        {
            // Make all in-flight command buffers available again
            for (auto& cmdBuf : mInFlightCmdList)
            {
                VA_AssertResult(vkResetCommandBuffer(cmdBuf->GetCmd(), VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT),
                    "Failed to reset command buffer");
                mAvailableCmdList.push_back(cmdBuf);
            }
            mInFlightCmdList.clear();
        }
        Owner<VA_CommandListNative> AllocateCommandList()
        {
            if (mAvailableCmdList.size())
            {
                auto p = std::move(mAvailableCmdList.back());
                mAvailableCmdList.pop_back();
                return p;
            }

            VkCommandBufferAllocateInfo bufferAI{};
            bufferAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            bufferAI.commandPool        = mCmdPool;
            bufferAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            bufferAI.commandBufferCount = 1;

            VkCommandBuffer buffer;
            VA_AssertResult(vkAllocateCommandBuffers(mContext->GetVulkanDevice(), &bufferAI, &buffer),
                "Failed to allocate command buffer");

            return MakeOwner<VA_CommandListNative>(buffer);
        }

    public:
        VA_Device*                       mContext;
        VkCommandPool                    mCmdPool;
        Vec<Owner<VA_CommandListNative>> mAvailableCmdList;
        Vec<Owner<VA_CommandListNative>> mInFlightCmdList;
    };

    struct VA_CommandBufferManagerInternal
    {
        ERhiPipelineType            mType;
        VA_Queue*                   mQueue;
        Vec<Owner<VA_CommandPool>>  mCommandPools;

        Owner<VA_CommandListNative> mActiveCmdBuffer;
        Owner<VA_CommandListNative> mUploadCmdBuffer;

        Owner<RhiTaskSubmission>    mRenderingCompleteSemaphore;
        Owner<RhiTaskSubmission>    mUploadingCompleteSemaphore;

        Vec<RhiTaskSubmission*>     mRenderingCmdToWaitOn;
        Vec<RhiTaskSubmission*>     mUploadingCmdToWaitOn;
    };

    struct VA_CommandListContextInternal
    {
        VA_CommandListContext*         mPrimaryContext = nullptr;
        ERhiPipelineType               mType;
        VA_Queue*                      mQueue;
        Owner<VA_CommandBufferManager> mManager;
    };

    // ===== Command Buffer Manager =====
    IFRIT_VKRHI2_API VA_CommandBufferManager::VA_CommandBufferManager(
        VA_Device* device, ERhiPipelineType type, VA_Queue* queue)
    {
        mInternal         = new VA_CommandBufferManagerInternal();
        mInternal->mType  = type;
        mInternal->mQueue = queue;
        for (int i = 0; i < 3; i++)
        {
            mInternal->mCommandPools.push_back(MakeOwner<VA_CommandPool>(device, queue));
        }
    }
    IFRIT_VKRHI2_API VA_CommandListNative* VA_CommandBufferManager::GetActiveCommandList()
    {
        return mInternal->mActiveCmdBuffer.get();
    }
    IFRIT_VKRHI2_API VA_CommandListNative* VA_CommandBufferManager::GetUploadCommandList()
    {
        return mInternal->mUploadCmdBuffer.get();
    }

    IFRIT_VKRHI2_API void VA_CommandBufferManager::SubmitActiveCommandList()
    {
        IF_LOG_ASSERTION(
            "VA_CommandBufferManager", mInternal->mUploadCmdBuffer == nullptr, "Upload cmd buffer is busy");
        IF_LOG_ASSERTION("VA_CommandBufferManager", mInternal->mActiveCmdBuffer != nullptr, "No active cmd buffer");
        IF_LOG_ASSERTION("VA_CommandBufferManager",
            mInternal->mActiveCmdBuffer->GetState() == EVA_CommandListNativeState::Recording,
            "Active cmd buffer not ready to submit");
    }
    IFRIT_VKRHI2_API void VA_CommandBufferManager::SubmitUploadCommandList() {}

    // ===== Command List Context =====
    IFRIT_VKRHI2_API      VA_CommandListContext::VA_CommandListContext(
        VA_Device* device, ERhiPipelineType type, VA_Queue* queue, VA_CommandListContext* primaryCmd)
        : RhiCommandListContext()
    {
        mInternal                  = new VA_CommandListContextInternal();
        mInternal->mType           = type;
        mInternal->mPrimaryContext = primaryCmd;
        mContext                   = device;

        mInternal->mManager = MakeOwner<VA_CommandBufferManager>(device, type, mInternal->mQueue);
    }

} // namespace Ifrit::RHI::VulkanRHI2
