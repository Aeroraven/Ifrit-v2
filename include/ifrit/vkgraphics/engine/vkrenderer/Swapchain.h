
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include <vulkan/vulkan.h>
#ifdef _WIN32
    #include <vulkan/vulkan_win32.h>
#else
    #include <vulkan/vulkan_xlib.h>
#endif
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include <vector>

namespace Ifrit::Graphics::VulkanGraphics
{
    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities;
        Vec<VkSurfaceFormatKHR>  formats;
        Vec<VkPresentModeKHR>    presentModes;
    };

    class IFRIT_APIDECL Swapchain : public Rhi::RhiSwapchain
    {
    private:
        EngineContext*          m_context = nullptr;
        void*                   m_hInstance;
        void*                   m_hWnd;
        VkSurfaceKHR            m_surface;
        VkQueue                 m_presentQueue;
        u32                     m_presentQueueFamilyIndex;
        SwapChainSupportDetails m_supportDetails;
        VkSurfaceFormatKHR      m_preferredSurfaceFormat;
        VkPresentModeKHR        m_preferredPresentMode;
        VkExtent2D              m_extent;
        u32                     m_backbufferCount = 0;

        VkSwapchainKHR          m_swapchain;

        Vec<VkImage>            m_images;
        Vec<VkImageView>        m_imageViews;

        u32                     m_currentFrame = 0;
        u32                     m_imageIndex   = 0;
        Vec<VkSemaphore>        m_imageAvailableSemaphores;
        Vec<VkSemaphore>        m_renderingFinishSemaphores;
        Vec<VkFence>            m_inFlightFences;

    protected:
        void Init();
        void Destructor();

    public:
        Swapchain(Rhi::RhiDevice* context);
        ~Swapchain();
        u32                AcquireNextImage() override;
        void               Present() override;

        inline VkQueue     GetPresentQueue() const { return m_presentQueue; }
        inline VkFence     GetCurrentFrameFence() const { return m_inFlightFences[m_currentFrame]; }
        inline VkSemaphore GetImageAvailableSemaphoreCurrentFrame() const
        {
            return m_imageAvailableSemaphores[m_currentFrame];
        }
        inline VkSemaphore GetRenderingFinishSemaphoreCurrentFrame() const
        {
            return m_renderingFinishSemaphores[m_currentFrame];
        }
        inline VkSemaphore GetImageAvailableSemaphore(u32 index) const { return m_imageAvailableSemaphores[index]; }

        inline VkImage     GetCurrentImage() const { return m_images[m_imageIndex]; }
        inline VkImageView GetCurrentImageView() const { return m_imageViews[m_imageIndex]; }
        inline VkFormat    GetPreferredFormat() const { return m_preferredSurfaceFormat.format; }

        inline u32         GetQueueFamily() const { return m_presentQueueFamilyIndex; }
        inline u32         GetNumBackbuffers() const override { return m_backbufferCount; }
        inline u32         GetCurrentFrameIndex() const override { return m_currentFrame; }
        inline u32         GetCurrentImageIndex() const override { return m_currentFrame; }
    };
} // namespace Ifrit::Graphics::VulkanGraphics