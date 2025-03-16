
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

namespace Ifrit::GraphicsBackend::VulkanGraphics {
struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

class IFRIT_APIDECL Swapchain : public Rhi::RhiSwapchain {
private:
  EngineContext *m_context = nullptr;
  void *m_hInstance;
  void *m_hWnd;
  VkSurfaceKHR m_surface;
  VkQueue m_presentQueue;
  u32 m_presentQueueFamilyIndex;
  SwapChainSupportDetails m_supportDetails;
  VkSurfaceFormatKHR m_preferredSurfaceFormat;
  VkPresentModeKHR m_preferredPresentMode;
  VkExtent2D m_extent;
  u32 m_backbufferCount = 0;

  VkSwapchainKHR m_swapchain;

  std::vector<VkImage> m_images;
  std::vector<VkImageView> m_imageViews;

  u32 m_currentFrame = 0;
  u32 m_imageIndex = 0;
  std::vector<VkSemaphore> m_imageAvailableSemaphores;
  std::vector<VkSemaphore> m_renderingFinishSemaphores;
  std::vector<VkFence> m_inFlightFences;

protected:
  void init();
  void destructor();

public:
  Swapchain(Rhi::RhiDevice *context);
  ~Swapchain();
  u32 acquireNextImage() override;
  void present() override;

  inline VkQueue getPresentQueue() const { return m_presentQueue; }
  inline VkFence getCurrentFrameFence() const { return m_inFlightFences[m_currentFrame]; }
  inline VkSemaphore getImageAvailableSemaphoreCurrentFrame() const {
    return m_imageAvailableSemaphores[m_currentFrame];
  }
  inline VkSemaphore getRenderingFinishSemaphoreCurrentFrame() const {
    return m_renderingFinishSemaphores[m_currentFrame];
  }
  inline VkSemaphore getImageAvailableSemaphore(u32 index) const { return m_imageAvailableSemaphores[index]; }

  inline VkImage getCurrentImage() const { return m_images[m_imageIndex]; }
  inline VkImageView getCurrentImageView() const { return m_imageViews[m_imageIndex]; }
  inline VkFormat getPreferredFormat() const { return m_preferredSurfaceFormat.format; }

  inline u32 getQueueFamily() const { return m_presentQueueFamilyIndex; }
  inline u32 getNumBackbuffers() const override { return m_backbufferCount; }
  inline u32 getCurrentFrameIndex() const override { return m_currentFrame; }
  inline u32 getCurrentImageIndex() const override { return m_currentFrame; }
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics