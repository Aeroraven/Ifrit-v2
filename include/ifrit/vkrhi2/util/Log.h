#pragma once
#include <vulkan/vulkan.h>
#include "ifrit/core/logging/Logging.h"

namespace Ifrit::RHI::VulkanRHI2
{
    IF_FORCEINLINE void VA_AssertResult(VkResult result, const char* message)
    {
        if (result != VK_SUCCESS)
        {
            IF_LOG_CRITICAL("VulkanRHI2", "Vulkan operation failed: {} with result: {}", message, (i64)result);
        }
    }
} // namespace Ifrit::RHI::VulkanRHI2