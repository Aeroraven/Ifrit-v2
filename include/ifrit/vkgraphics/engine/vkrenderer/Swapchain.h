#pragma once
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#else
#include <vulkan/vulkan_xlib.h>
#endif
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include <vector>

namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics {
struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

class IFRIT_APIDECL Swapchain {
private:
  EngineContext *m_context;
  void *m_hInstance;
  void *m_hWnd;
  VkSurfaceKHR m_surface;
  VkQueue m_presentQueue;
  uint32_t m_presentQueueFamilyIndex;
  SwapChainSupportDetails m_supportDetails;
  VkSurfaceFormatKHR m_preferredSurfaceFormat;
  VkPresentModeKHR m_preferredPresentMode;
  VkExtent2D m_extent;
  uint32_t m_backbufferCount = 0;

  VkSwapchainKHR m_swapchain;

  std::vector<VkImage> m_images;
  std::vector<VkImageView> m_imageViews;

  uint32_t m_currentFrame = 0;
  uint32_t m_imageIndex = 0;
  std::vector<VkSemaphore> m_imageAvailableSemaphores;
  std::vector<VkSemaphore> m_renderingFinishSemaphores;
  std::vector<VkFence> m_inFlightFences;

protected:
  void init();
  void destructor();

public:
  Swapchain(EngineContext *context);
  ~Swapchain();
  uint32_t acquireNextImage();
  void presentImage();

  inline VkQueue getPresentQueue() const { return m_presentQueue; }
  inline VkFence getCurrentFrameFence() const {
    return m_inFlightFences[m_currentFrame];
  }
  inline VkSemaphore getImageAvailableSemaphoreCurrentFrame() const {
    return m_imageAvailableSemaphores[m_currentFrame];
  }
  inline VkSemaphore getRenderingFinishSemaphoreCurrentFrame() const {
    return m_renderingFinishSemaphores[m_currentFrame];
  }
  inline VkSemaphore getImageAvailableSemaphore(uint32_t index) const {
    return m_imageAvailableSemaphores[index];
  }

  inline VkImage getCurrentImage() const { return m_images[m_imageIndex]; }
  inline VkImageView getCurrentImageView() const {
    return m_imageViews[m_imageIndex];
  }
  inline VkFormat getPreferredFormat() const {
    return m_preferredSurfaceFormat.format;
  }

  inline uint32_t getQueueFamily() const { return m_presentQueueFamilyIndex; }
  inline uint32_t getNumBackbuffers() const { return m_backbufferCount; }
  inline uint32_t getCurrentFrame() const { return m_currentFrame; }
  inline uint32_t getCurrentImageId() const { return m_currentFrame; }
};
} // namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics