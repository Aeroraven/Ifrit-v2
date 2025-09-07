#include "ifrit.internal/vkrhi2/platform/PlatformSurface.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/vkrhi2/adapter/Device.h"

#ifdef _WIN32
    #include <Windows.h>
    #include <vulkan/vulkan_win32.h>
#endif

namespace Ifrit::RHI::VulkanRHI2
{

    IFRIT_APIDECL void CreatePlatformSurfaceWin32(
        VkInstance instance, VkSurfaceKHR* outSurface, void* hwnd, void* hinstance)
    {
#ifdef _WIN32
        VkWin32SurfaceCreateInfoKHR surfaceCI{};
        surfaceCI.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        surfaceCI.hinstance = (HINSTANCE)hinstance;
        surfaceCI.hwnd      = (HWND)hwnd;
        VA_AssertResult(
            vkCreateWin32SurfaceKHR(instance, &surfaceCI, nullptr, outSurface), "Failed to create window surface");
#else
        IF_LOG_CRITICAL("CreatePlatformSurfaceWin32", "Unsupported platform");
#endif
    }

    IFRIT_APIDECL void CreatePlatformSurface(VA_Device* device, VkSurfaceKHR* outSurface)
    {
        auto initArgs = device->GetInitializationArgs();
#ifdef _WIN32
        CreatePlatformSurfaceWin32(
            device->GetVulkanInstance(), outSurface, initArgs.mWin32.m_hWnd, initArgs.mWin32.m_hInstance);
#else
        IF_LOG_CRITICAL("CreatePlatformSurface", "Unsupported platform");
#endif
    }

} // namespace Ifrit::RHI::VulkanRHI2