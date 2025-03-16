
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

#ifdef _WIN32
#include <windows.h>
#endif
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Swapchain.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include <algorithm>
#include <array>
#include <vector>

using namespace Ifrit::Common::Utility;

namespace Ifrit::GraphicsBackend::VulkanGraphics {
IFRIT_APIDECL void Swapchain::init() {
  // Window surface
#ifdef _WIN32
  m_hInstance = m_context->getArgs().m_win32.m_hInstance;
  m_hWnd = m_context->getArgs().m_win32.m_hWnd;
  VkWin32SurfaceCreateInfoKHR surfaceCI{};
  surfaceCI.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
  surfaceCI.hinstance = (HINSTANCE)m_hInstance;
  surfaceCI.hwnd = (HWND)m_hWnd;
  vkrVulkanAssert(vkCreateWin32SurfaceKHR(m_context->getInstance(), &surfaceCI, nullptr, &m_surface),
                  "Failed to create window surface");
#else
  // TODO: Implement for linux
  vkrError("Unsupported platform");
#endif

  // Queue specify
  auto &queueData = m_context->getQueueInfo();
  auto &deviceExtensions = m_context->getDeviceExtensions();
  for (int i = 0; i < queueData.m_queueFamilies.size(); i++) {
    VkBool32 presentSupport = VK_FALSE;
    VkBool32 swapchainSupport = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(m_context->getPhysicalDevice(), i, m_surface, &presentSupport);
    {
      uint32_t extensionCount = 0;
      vkEnumerateDeviceExtensionProperties(m_context->getPhysicalDevice(), nullptr, &extensionCount, nullptr);
      std::vector<VkExtensionProperties> availableExtensions(extensionCount);
      vkEnumerateDeviceExtensionProperties(m_context->getPhysicalDevice(), nullptr, &extensionCount,
                                           availableExtensions.data());
      bool allSupported = true;
      for (auto extension : deviceExtensions) {
        bool supported = false;
        for (auto availableExtension : availableExtensions) {
          if (strcmp(extension, availableExtension.extensionName) == 0) {
            supported = true;
            break;
          }
        }
        if (!supported) {
          allSupported = false;
          break;
        }
      }
      swapchainSupport = allSupported;
    }

    if (presentSupport && swapchainSupport) {
      for (auto queue : queueData.m_allQueues) {
        if (queue.m_familyIndex == i) {
          m_presentQueue = queue.m_queue;
          m_presentQueueFamilyIndex = i;
          break;
        }
      }
      break;
    }
  }
  vkrDebug("Queue specified");

  // Swapchain support details
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_context->getPhysicalDevice(), m_surface, &m_supportDetails.capabilities);
  {
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_context->getPhysicalDevice(), m_surface, &formatCount, nullptr);
    vkrAssert(formatCount != 0, "No surface formats found");
    m_supportDetails.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_context->getPhysicalDevice(), m_surface, &formatCount,
                                         m_supportDetails.formats.data());
  }
  {
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_context->getPhysicalDevice(), m_surface, &presentModeCount, nullptr);
    vkrAssert(presentModeCount != 0, "No present modes found");
    m_supportDetails.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_context->getPhysicalDevice(), m_surface, &presentModeCount,
                                              m_supportDetails.presentModes.data());
  }

  // Choose format
  // TODO: HDR support
  bool found = false;
  for (auto &format : m_supportDetails.formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      m_preferredSurfaceFormat = format;
      found = true;
      break;
    }
  }
  if (!found) {
    vkrLog("Preferred format not found, using first available");
    m_preferredSurfaceFormat = m_supportDetails.formats[0];
  }

  // Choose present mode
  found = false;
  for (auto &mode : m_supportDetails.presentModes) {
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      m_preferredPresentMode = mode;
      found = true;
      break;
    }
  }
  if (!found) {
    vkrLog("Preferred present mode not found, using FIFO");
    m_preferredPresentMode = VK_PRESENT_MODE_FIFO_KHR;
  }

  // Choose extent
  if (m_supportDetails.capabilities.currentExtent.width != UINT32_MAX) {
    m_extent = m_supportDetails.capabilities.currentExtent;
  } else {
    auto width = m_context->getArgs().m_surfaceWidth;
    auto height = m_context->getArgs().m_surfaceHeight;
    m_extent = {width, height};
    m_extent.width = std::clamp(m_extent.width, m_supportDetails.capabilities.minImageExtent.width,
                                m_supportDetails.capabilities.maxImageExtent.width);
    m_extent.height = std::clamp(m_extent.height, m_supportDetails.capabilities.minImageExtent.height,
                                 m_supportDetails.capabilities.maxImageExtent.height);
  }

  // Backbuffer count
  m_backbufferCount = m_context->getArgs().m_expectedSwapchainImageCount;
  m_backbufferCount = std::clamp(m_backbufferCount, m_supportDetails.capabilities.minImageCount,
                                 m_supportDetails.capabilities.maxImageCount);

  // Create swapchain
  VkSwapchainCreateInfoKHR swapchainCI{};
  swapchainCI.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapchainCI.surface = m_surface;
  swapchainCI.minImageCount = m_backbufferCount;
  swapchainCI.imageFormat = m_preferredSurfaceFormat.format;
  swapchainCI.imageColorSpace = m_preferredSurfaceFormat.colorSpace;
  swapchainCI.imageExtent = m_extent;
  swapchainCI.imageArrayLayers = 1;
  swapchainCI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  // Note: Beware of queue family ownership
  swapchainCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapchainCI.queueFamilyIndexCount = 0;
  swapchainCI.pQueueFamilyIndices = nullptr;

  swapchainCI.preTransform = m_supportDetails.capabilities.currentTransform;
  swapchainCI.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchainCI.presentMode = m_preferredPresentMode;
  swapchainCI.clipped = VK_TRUE;
  swapchainCI.oldSwapchain = VK_NULL_HANDLE;

  vkrVulkanAssert(vkCreateSwapchainKHR(m_context->getDevice(), &swapchainCI, nullptr, &m_swapchain),
                  "Failed to create swapchain");

  // Retrieve images
  uint32_t imageCount;
  vkGetSwapchainImagesKHR(m_context->getDevice(), m_swapchain, &imageCount, nullptr);
  m_images.resize(imageCount);
  vkGetSwapchainImagesKHR(m_context->getDevice(), m_swapchain, &imageCount, m_images.data());

  // Create image views
  m_imageViews.resize(imageCount);
  for (int i = 0; i < static_cast<int>(imageCount); i++) {
    VkImageViewCreateInfo imageViewCI{};
    imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCI.image = m_images[i];
    imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCI.format = m_preferredSurfaceFormat.format;
    imageViewCI.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCI.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCI.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCI.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCI.subresourceRange.baseMipLevel = 0;
    imageViewCI.subresourceRange.levelCount = 1;
    imageViewCI.subresourceRange.baseArrayLayer = 0;
    imageViewCI.subresourceRange.layerCount = 1;
    vkrVulkanAssert(vkCreateImageView(m_context->getDevice(), &imageViewCI, nullptr, &m_imageViews[i]),
                    "Failed to create image view");
  }

  // Create semaphores and fences
  m_imageAvailableSemaphores.resize(m_backbufferCount);
  m_inFlightFences.resize(m_backbufferCount);
  m_renderingFinishSemaphores.resize(m_backbufferCount);

  VkSemaphoreCreateInfo semaphoreCI{};
  semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fenceCI{};
  fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (uint32_t i = 0; i < m_backbufferCount; i++) {
    vkrVulkanAssert(vkCreateSemaphore(m_context->getDevice(), &semaphoreCI, nullptr, &m_imageAvailableSemaphores[i]),
                    "Failed to create semaphore");
    vkrVulkanAssert(vkCreateFence(m_context->getDevice(), &fenceCI, nullptr, &m_inFlightFences[i]),
                    "Failed to create fence");
    vkrVulkanAssert(vkCreateSemaphore(m_context->getDevice(), &semaphoreCI, nullptr, &m_renderingFinishSemaphores[i]),
                    "Failed to create semaphore");
  }

  vkrLog("Swapchain created");
}

IFRIT_APIDECL void Swapchain::destructor() {
  for (auto imageView : m_imageViews) {
    vkDestroyImageView(m_context->getDevice(), imageView, nullptr);
  }
  vkDestroySwapchainKHR(m_context->getDevice(), m_swapchain, nullptr);
  vkDestroySurfaceKHR(m_context->getInstance(), m_surface, nullptr);

  for (uint32_t i = 0; i < m_backbufferCount; i++) {
    vkDestroySemaphore(m_context->getDevice(), m_imageAvailableSemaphores[i], nullptr);
    vkDestroySemaphore(m_context->getDevice(), m_renderingFinishSemaphores[i], nullptr);
    vkDestroyFence(m_context->getDevice(), m_inFlightFences[i], nullptr);
  }
}

IFRIT_APIDECL Swapchain::Swapchain(Rhi::RhiDevice *context) : m_context(checked_cast<EngineContext>(context)) {
  init();
}

IFRIT_APIDECL Swapchain::~Swapchain() { destructor(); }

IFRIT_APIDECL uint32_t Swapchain::acquireNextImage() {
  vkWaitForFences(m_context->getDevice(), 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);
  vkResetFences(m_context->getDevice(), 1, &m_inFlightFences[m_currentFrame]);
  uint32_t imageIndex;
  vkAcquireNextImageKHR(m_context->getDevice(), m_swapchain, UINT64_MAX, m_imageAvailableSemaphores[m_currentFrame],
                        VK_NULL_HANDLE, &imageIndex);
  m_imageIndex = imageIndex;
  return imageIndex;
}

IFRIT_APIDECL void Swapchain::present() {
  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &m_renderingFinishSemaphores[m_currentFrame];
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &m_swapchain;
  presentInfo.pImageIndices = &m_imageIndex;

  vkQueuePresentKHR(m_presentQueue, &presentInfo);
  m_currentFrame = (m_currentFrame + 1) % m_backbufferCount;
}
} // namespace Ifrit::GraphicsBackend::VulkanGraphics