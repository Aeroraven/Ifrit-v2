#pragma once
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#else
#include <vulkan/vulkan_xlib.h>
#endif
#include <vector>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
namespace Ifrit::Engine::VkRenderer{
    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    class IFRIT_APIDECL Swapchain{
    private:
        EngineContext* m_context;
        void* m_hInstance;
        void* m_hWnd;
        VkSurfaceKHR m_surface;
        VkQueue m_presentQueue;
        SwapChainSupportDetails m_supportDetails;
        VkSurfaceFormatKHR m_preferredSurfaceFormat;
        VkPresentModeKHR m_preferredPresentMode;
        VkExtent2D m_extent;
        uint32_t m_backbufferCount = 0;

        VkSwapchainKHR m_swapchain;
        
        std::vector<VkImage> m_images;
        std::vector<VkImageView> m_imageViews;

    protected:
        void init();
        void destructor();
    public:
        Swapchain(EngineContext* context);
        ~Swapchain();

    };
}