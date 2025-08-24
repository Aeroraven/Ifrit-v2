#pragma once
#include <vulkan/vulkan.h>
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/util/Helpers.h"

namespace Ifrit::RHI::VulkanRHI2
{
    inline VkImageUsageFlags TranslateCreateFlags(VA_Device* device, ERhiImageUsage usage)
    {
        VkImageUsageFlags baseFlags =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        if (HasFlagBit(usage, ERhiImageUsageFlag::RenderTarget))
        {
            baseFlags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
        if (HasFlagBit(usage, ERhiImageUsageFlag::Depth))
        {
            baseFlags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        }
        if (HasFlagBit(usage, ERhiImageUsageFlag::UnorderedAccess))
        {
            baseFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
        if (HasFlagBit(usage, ERhiImageUsageFlag::InputAttachment))
        {
            baseFlags |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
        }
        if (HasFlagBit(usage, ERhiImageUsageFlag::Presentable))
        {
            baseFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
        return baseFlags;
    }
} // namespace Ifrit::RHI::VulkanRHI2