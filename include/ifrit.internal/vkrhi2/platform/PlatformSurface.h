#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/rhi/common/RhiBaseTypes.h"
#include "ifrit/vkrhi2/util/Log.h"
#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{
    class VA_Device;

    IFRIT_VKRHI2_API void CreatePlatformSurfaceWin32(VkInstance instance, VkSurfaceKHR* outSurface, void* hwnd, void* hinstance);
    IFRIT_VKRHI2_API void CreatePlatformSurface(VA_Device* device, VkSurfaceKHR* outSurface);

} // namespace Ifrit::RHI::VulkanRHI2