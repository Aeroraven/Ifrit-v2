
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
#include "ifrit/common/logging/Logging.h"
#include <stdexcept>
namespace Ifrit::Graphics::VulkanGraphics
{
    inline void vkrAssert(bool condition, const char* message)
    {
        if (!condition)
        {
            iError("Error Message:{}", message);
            throw std::runtime_error(message);
        }
    }
    inline void vkrDebug(const char* message)
    {
        iDebug(message);
    }
    inline void vkrVulkanAssert(VkResult result, const char* message)
    {
        if (result != VK_SUCCESS)
        {
            iError("Error code: {}", static_cast<int32_t>(result));
            if (result == VK_ERROR_OUT_OF_DATE_KHR)
            {
                iError("Error details: VK_ERROR_OUT_OF_DATE_KHR");
            }
            else if (result == VK_ERROR_DEVICE_LOST)
            {
                iError("Error details: VK_ERROR_DEVICE_LOST");
            }
            else if (result == VK_ERROR_SURFACE_LOST_KHR)
            {
                iError("Error details: VK_ERROR_SURFACE_LOST_KHR");
            }
            else if (result == VK_ERROR_OUT_OF_HOST_MEMORY)
            {
                iError("Error details: VK_ERROR_OUT_OF_HOST_MEMORY");
            }
            else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
            {
                iError("Error details: VK_ERROR_OUT_OF_DEVICE_MEMORY");
            }
            else if (result == VK_ERROR_INITIALIZATION_FAILED)
            {
                iError("Error details: VK_ERROR_INITIALIZATION_FAILED");
            }
            else if (result == VK_ERROR_EXTENSION_NOT_PRESENT)
            {
                iError("Error details: VK_ERROR_EXTENSION_NOT_PRESENT");
            }
            else if (result == VK_ERROR_FEATURE_NOT_PRESENT)
            {
                iError("Error details: VK_ERROR_FEATURE_NOT_PRESENT");
            }
            else if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
            {
                iError("Error details: VK_ERROR_INCOMPATIBLE_DRIVER");
            }
            else if (result == VK_ERROR_TOO_MANY_OBJECTS)
            {
                iError("Error details: VK_ERROR_TOO_MANY_OBJECTS");
            }
            else if (result == VK_ERROR_FORMAT_NOT_SUPPORTED)
            {
                iError("Error details: VK_ERROR_FORMAT_NOT_SUPPORTED");
            }
            else if (result == VK_ERROR_FRAGMENTED_POOL)
            {
                iError("Error details: VK_ERROR_FRAGMENTED_POOL");
            }
            else if (result == VK_ERROR_UNKNOWN)
            {
                iError("Error details: VK_ERROR_UNKNOWN");
            }

            throw std::runtime_error(message);
        }
    }
    inline void vkrLog(const char* message)
    {
        iInfo(message);
    }
    inline void vkrError(const char* message)
    {
        iError("Error:{}", message);
        throw std::runtime_error(message);
    }
} // namespace Ifrit::Graphics::VulkanGraphics