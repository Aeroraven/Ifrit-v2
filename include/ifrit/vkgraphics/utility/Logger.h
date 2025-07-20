
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
#include "ifrit/vkgraphics/common/Pch.h"
#include <stdexcept>

namespace Ifrit::RHI::VulkanAdapter
{
    inline void vkrAssert(bool condition, const char* message)
    {
        if (!condition)
        {
            IF_LOG_ERROR("General", "Error Message:{}", message);
            throw std::runtime_error(message);
        }
    }
    inline void vkrDebug(const char* message) { IF_LOG_DEBUG("General", "{}", message); }
    inline void vkrVulkanAssert(VkResult result, const char* message)
    {
        if (result != VK_SUCCESS)
        {
            IF_LOG_ERROR("General", "Error code: {}", static_cast<int32_t>(result));
            if (result == VK_ERROR_OUT_OF_DATE_KHR)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_OUT_OF_DATE_KHR");
            }
            else if (result == VK_ERROR_DEVICE_LOST)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_DEVICE_LOST");
            }
            else if (result == VK_ERROR_SURFACE_LOST_KHR)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_SURFACE_LOST_KHR");
            }
            else if (result == VK_ERROR_OUT_OF_HOST_MEMORY)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_OUT_OF_HOST_MEMORY");
            }
            else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_OUT_OF_DEVICE_MEMORY");
            }
            else if (result == VK_ERROR_INITIALIZATION_FAILED)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_INITIALIZATION_FAILED");
            }
            else if (result == VK_ERROR_EXTENSION_NOT_PRESENT)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_EXTENSION_NOT_PRESENT");
            }
            else if (result == VK_ERROR_FEATURE_NOT_PRESENT)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_FEATURE_NOT_PRESENT");
            }
            else if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_INCOMPATIBLE_DRIVER");
            }
            else if (result == VK_ERROR_TOO_MANY_OBJECTS)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_TOO_MANY_OBJECTS");
            }
            else if (result == VK_ERROR_FORMAT_NOT_SUPPORTED)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_FORMAT_NOT_SUPPORTED");
            }
            else if (result == VK_ERROR_FRAGMENTED_POOL)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_FRAGMENTED_POOL");
            }
            else if (result == VK_ERROR_UNKNOWN)
            {
                IF_LOG_ERROR("General", "Error details: VK_ERROR_UNKNOWN");
            }

            throw std::runtime_error(message);
        }
    }
    inline void vkrLog(const char* message) { IF_LOG_INFO("General", "{}", message); }
    inline void vkrError(const char* message)
    {
        IF_LOG_ERROR("General", "Error:{}", message);
        throw std::runtime_error(message);
    }
} // namespace Ifrit::RHI::VulkanAdapter