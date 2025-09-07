#pragma once

#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/adapter/DisplayViewport.h"
#include "ifrit/vkrhi2/adapter/CommandSubmission.h"
#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{
    class VA_Device;
    class VA_CommandListContext;

    struct VA_SwapchainDesc
    {
        ERhiImageFormat mFormat                = ERhiImageFormat::Undefined;
        u32             mWidth                 = 0;
        u32             mHeight                = 0;
        u32             mDesiredNumBackBuffers = 3;
        bool            mEnableVSync           = true;
    };

    struct VA_SwapchainInternal;
    class IFRIT_VKRHI2_API VA_Swapchain
    {
    public:
        VA_Swapchain(VA_Device* device, const VA_SwapchainDesc& desc);
        ~VA_Swapchain();

        u32                       AcquireNextImage();
        Ref<VA_CommandSubmission> GetCurrentImageAcquiredSemaphore();

        void                      Present(VA_CommandListContext* cmdListCtx);

    private:
        void PresentInternal();

    private:
        VA_SwapchainInternal* mData;
    };
} // namespace Ifrit::RHI::VulkanRHI2