#include "ifrit/vkrhi2/adapter/DisplayViewport.h"
#include "ifrit.internal/vkrhi2/platform/PlatformSurface.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/adapter/CommandBuffer.h"
#include "ifrit/vkrhi2/adapter/CommandContext.h"
#include "ifrit.internal/vkrhi2/adapter/CmdHelpersBarrier.h"

namespace Ifrit::RHI::VulkanRHI2
{
    struct VA_SwapchainInternal
    {
        VkSurfaceKHR       mSurface = VK_NULL_HANDLE;
        VA_Device*         mDevice  = nullptr;
        VkSurfaceFormatKHR mPreferredFormat;
        VkPresentModeKHR   mPresentMode = VK_PRESENT_MODE_FIFO_KHR;
        VkSwapchainKHR     mHandle      = VK_NULL_HANDLE;

        Vec<VkImage>       mImages;

        Vec<VkSemaphore>   mImageAcquiredSema;
        Vec<VkFence>       mImageAcquiredFence;
        Vec<VkSemaphore>   mRenderCompleteSema;

        u32                mPresentId = 0;
        u32                mImageIdx  = 0;
    };

    IFRIT_VKRHI2_API VA_Swapchain::VA_Swapchain(VA_Device* device, const VA_SwapchainDesc& desc)
    {
        mData          = new VA_SwapchainInternal();
        mData->mDevice = device;
        CreatePlatformSurface(device, &mData->mSurface);

        VkSurfaceCapabilitiesKHR surfaceCapabilities;
        VA_AssertResult(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                            device->GetVulkanPhysicalDevice(), mData->mSurface, &surfaceCapabilities),
            "Failed to get surface capabilities");

        device->SetupPresentQueue(mData->mSurface);

        // Surface formats
        Vec<VkSurfaceFormatKHR> formats;
        bool                    fmtFound = false;
        {
            u32 formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                device->GetVulkanPhysicalDevice(), mData->mSurface, &formatCount, nullptr);
            formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                device->GetVulkanPhysicalDevice(), mData->mSurface, &formatCount, formats.data());

            for (auto& format : formats)
            {
                if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                {
                    mData->mPreferredFormat = format;
                    fmtFound                = true;
                    break;
                }
            }
            if (!fmtFound)
            {
                IF_LOG_WARNING("VA_Swapchain", "Preferred format not found, using first available");
                mData->mPreferredFormat = formats[0];
            }
        }

        // Present modes
        Vec<VkPresentModeKHR> presentModes;
        {
            u32 presentModeCount = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                device->GetVulkanPhysicalDevice(), mData->mSurface, &presentModeCount, nullptr);
            presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                device->GetVulkanPhysicalDevice(), mData->mSurface, &presentModeCount, presentModes.data());

            bool hasImmediate = false;
            bool hasMailbox   = false;
            bool hasFifo      = false;
            for (auto& mode : presentModes)
            {
                if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
                    hasImmediate = true;
                if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                    hasMailbox = true;
                if (mode == VK_PRESENT_MODE_FIFO_KHR)
                    hasFifo = true;
            }
            if (hasImmediate && !desc.mEnableVSync)
            {
                mData->mPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
            }
            else if (hasMailbox)
            {
                mData->mPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
            }
            else if (hasFifo)
            {
                mData->mPresentMode = VK_PRESENT_MODE_FIFO_KHR;
            }
            else
            {
                IF_LOG_WARNING("VA_Swapchain", "No supported present mode found, using the first available");
                mData->mPresentMode = presentModes[0];
            }
        }

        // Set extent
        VkExtent2D extent;
        {
            if (surfaceCapabilities.currentExtent.width != UINT32_MAX)
            {
                extent = surfaceCapabilities.currentExtent;
            }
            else
            {
                extent.width  = desc.mWidth;
                extent.height = desc.mHeight;

                if (extent.width < surfaceCapabilities.minImageExtent.width)
                    extent.width = surfaceCapabilities.minImageExtent.width;
                else if (extent.width > surfaceCapabilities.maxImageExtent.width)
                    extent.width = surfaceCapabilities.maxImageExtent.width;

                if (extent.height < surfaceCapabilities.minImageExtent.height)
                    extent.height = surfaceCapabilities.minImageExtent.height;
                else if (extent.height > surfaceCapabilities.maxImageExtent.height)
                    extent.height = surfaceCapabilities.maxImageExtent.height;
            }
        }

        // Set num back buffers
        u32 numBackBuffers = desc.mDesiredNumBackBuffers;
        {
            if (numBackBuffers < surfaceCapabilities.minImageCount)
                numBackBuffers = surfaceCapabilities.minImageCount;
            else if (surfaceCapabilities.maxImageCount > 0 && numBackBuffers > surfaceCapabilities.maxImageCount)
                numBackBuffers = surfaceCapabilities.maxImageCount;
        }

        // Create swapchain
        VkSwapchainCreateInfoKHR swapchainCI{};
        swapchainCI.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swapchainCI.surface          = mData->mSurface;
        swapchainCI.minImageCount    = numBackBuffers;
        swapchainCI.imageFormat      = mData->mPreferredFormat.format;
        swapchainCI.imageColorSpace  = mData->mPreferredFormat.colorSpace;
        swapchainCI.imageExtent      = extent;
        swapchainCI.imageArrayLayers = 1;
        swapchainCI.imageUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        swapchainCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchainCI.preTransform     = surfaceCapabilities.currentTransform;
        swapchainCI.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swapchainCI.presentMode      = mData->mPresentMode;
        swapchainCI.clipped          = VK_TRUE;
        swapchainCI.oldSwapchain     = VK_NULL_HANDLE;
        VA_AssertResult(vkCreateSwapchainKHR(device->GetVulkanDevice(), &swapchainCI, nullptr, &mData->mHandle),
            "Failed to create swapchain");

        // Get images
        {
            u32 imageCount = 0;
            VA_AssertResult(vkGetSwapchainImagesKHR(device->GetVulkanDevice(), mData->mHandle, &imageCount, nullptr),
                "Failed to get swapchain images");
            mData->mImages.resize(imageCount);
            VA_AssertResult(
                vkGetSwapchainImagesKHR(device->GetVulkanDevice(), mData->mHandle, &imageCount, mData->mImages.data()),
                "Failed to get swapchain images");
            IF_LOG_DEBUG("VA_Swapchain", "Swapchain created with {} images", imageCount);
        }

        // Create sync objects
        {
            mData->mImageAcquiredSema.resize(numBackBuffers);
            mData->mImageAcquiredFence.resize(numBackBuffers);
            mData->mRenderCompleteSema.resize(numBackBuffers);

            VkSemaphoreCreateInfo semaphoreCI{};
            semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkFenceCreateInfo fenceCI{};
            fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            for (u32 i = 0; i < numBackBuffers; i++)
            {
                VA_AssertResult(
                    vkCreateSemaphore(device->GetVulkanDevice(), &semaphoreCI, nullptr, &mData->mImageAcquiredSema[i]),
                    "Failed to create image acquired semaphore");
                VA_AssertResult(
                    vkCreateFence(device->GetVulkanDevice(), &fenceCI, nullptr, &mData->mImageAcquiredFence[i]),
                    "Failed to create image acquired fence");
                VA_AssertResult(
                    vkCreateSemaphore(device->GetVulkanDevice(), &semaphoreCI, nullptr, &mData->mRenderCompleteSema[i]),
                    "Failed to create render complete semaphore");

                // reset fence to signaled state
                // VA_AssertResult(vkResetFences(device->GetVulkanDevice(), 1, &mData->mImageAcquiredFence[i]),
                //    "Failed to reset image acquired fence");
            }

            IF_LOG_DEBUG("VA_Swapchain", "Created {} sync objects for swapchain", numBackBuffers);
        }
    }

    IFRIT_VKRHI2_API VA_Swapchain::~VA_Swapchain()
    {
        IF_LOG_INFO("VA_Swapchain", "Destroying swapchain");
        auto device = mData->mDevice;
        device->WaitIdle();

        for (u32 i = 0; i < mData->mImageAcquiredSema.size(); i++)
        {
            vkDestroySemaphore(device->GetVulkanDevice(), mData->mImageAcquiredSema[i], nullptr);
            vkDestroyFence(device->GetVulkanDevice(), mData->mImageAcquiredFence[i], nullptr);
            vkDestroySemaphore(device->GetVulkanDevice(), mData->mRenderCompleteSema[i], nullptr);
        }

        if (mData->mHandle != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(device->GetVulkanDevice(), mData->mHandle, nullptr);
            mData->mHandle = VK_NULL_HANDLE;
        }

        if (mData->mSurface != VK_NULL_HANDLE)
        {
            vkDestroySurfaceKHR(device->GetVulkanInstance(), mData->mSurface, nullptr);
            mData->mSurface = VK_NULL_HANDLE;
        }

        delete mData;
        mData = nullptr;
    }

    IFRIT_VKRHI2_API void VA_Swapchain::Present(VA_CommandListContext* cmdListCtx)
    {

        VA_PipelineBarriers barriers;
        barriers.AddImageTransition(mData->mImages[mData->mImageIdx], VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
        barriers.ExecuteNative(cmdListCtx->GetCommandBuffer()->GetCmd());
        cmdListCtx->RegisterExternalDependencies(
            mData->mImageAcquiredFence[mData->mPresentId], mData->mRenderCompleteSema[mData->mPresentId]);
        auto waitInfo = cmdListCtx->FlushCommands(ERhiCommandSubmissionAction::CPUWaitForSubmission);
        PresentInternal();
    }

    IFRIT_VKRHI2_API void VA_Swapchain::PresentInternal()
    {
        auto             device       = mData->mDevice;
        auto             presentQueue = device->GetPresentQueue()->GetNativeQueue();

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = &mData->mRenderCompleteSema[mData->mPresentId];
        presentInfo.swapchainCount     = 1;
        presentInfo.pSwapchains        = &mData->mHandle;
        presentInfo.pImageIndices      = &mData->mImageIdx;

        vkQueuePresentKHR(presentQueue, &presentInfo);
        mData->mPresentId = (mData->mPresentId + 1) % mData->mImageAcquiredSema.size();
    }

    IFRIT_VKRHI2_API u32 VA_Swapchain::AcquireNextImage()
    {
        // IF_LOG_INFO("VA_Swapchain", "Acquiring next image from swapchain {}", mData->mPresentId);

        auto device = mData->mDevice;
        device->WaitIdle();

        VA_AssertResult(vkWaitForFences(device->GetVulkanDevice(), 1, &mData->mImageAcquiredFence[mData->mPresentId],
                            VK_TRUE, UINT64_MAX),
            "Failed to wait for image acquired fence");
        VA_AssertResult(vkResetFences(device->GetVulkanDevice(), 1, &mData->mImageAcquiredFence[mData->mPresentId]),
            "Failed to reset image acquired fence");

        u32 imageIndex = 0;
        VA_AssertResult(vkAcquireNextImageKHR(device->GetVulkanDevice(), mData->mHandle, UINT64_MAX,
                            mData->mImageAcquiredSema[mData->mPresentId], VK_NULL_HANDLE, &imageIndex),
            "Failed to acquire next image from swapchain");
        mData->mImageIdx = imageIndex;

        return imageIndex;
    }

    IFRIT_VKRHI2_API Ref<VA_CommandSubmission> VA_Swapchain::GetCurrentImageAcquiredSemaphore()
    {
        auto sema        = MakeRef<VA_CommandSubmission>();
        sema->mSemaphore = mData->mImageAcquiredSema[mData->mPresentId];
        return sema;
    }
} // namespace Ifrit::RHI::VulkanRHI2