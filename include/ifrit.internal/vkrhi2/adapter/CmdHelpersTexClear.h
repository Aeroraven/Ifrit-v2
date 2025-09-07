#pragma once

#include "ifrit/rhi/common/RhiBaseTypes.h"
#include "ifrit/vkrhi2/util/Log.h"
#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{

    void CmdClearColorTexture(VkCommandBuffer cmd, VkImage image, VkImageLayout layout, RhiClearColorValue clearValue,
        const VkImageSubresourceRange& range)
    {
        IF_LOG_ASSERTION("VAHelper_CmdClearTexture", range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT,
            "CmdClearTexture only supports color aspect");

        VkClearColorValue vkClearValue{};
        switch (clearValue.m_Type)
        {
            case ERhiTypeFlags::Float32:
                vkClearValue.float32[0] = clearValue.m_ValueF32[0];
                vkClearValue.float32[1] = clearValue.m_ValueF32[1];
                vkClearValue.float32[2] = clearValue.m_ValueF32[2];
                vkClearValue.float32[3] = clearValue.m_ValueF32[3];
                break;
            case ERhiTypeFlags::UInt32:
                vkClearValue.uint32[0] = clearValue.m_ValueU32[0];
                vkClearValue.uint32[1] = clearValue.m_ValueU32[1];
                vkClearValue.uint32[2] = clearValue.m_ValueU32[2];
                vkClearValue.uint32[3] = clearValue.m_ValueU32[3];
                break;
            case ERhiTypeFlags::Int32:
                vkClearValue.int32[0] = clearValue.m_ValueI32[0];
                vkClearValue.int32[1] = clearValue.m_ValueI32[1];
                vkClearValue.int32[2] = clearValue.m_ValueI32[2];
                vkClearValue.int32[3] = clearValue.m_ValueI32[3];
                break;
            default:
                IF_LOG_CRITICAL("CmdClearTexture", "Unsupported clear color value type");
                return;
        }

        vkCmdClearColorImage(cmd, image, layout, &vkClearValue, 1, &range);
    }

    void CmdClearDepthStencilTexture(VkCommandBuffer cmd, VkImage image, VkImageLayout layout, f32 depth, u32 stencil,
        const VkImageSubresourceRange& range)
    {
        IF_LOG_ASSERTION("VAHelper_CmdClearTexture",
            range.aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT),
            "CmdClearTexture only supports depth/stencil aspect");

        VkClearDepthStencilValue vkClearValue{};
        vkClearValue.depth   = depth;
        vkClearValue.stencil = stencil;

        vkCmdClearDepthStencilImage(cmd, image, layout, &vkClearValue, 1, &range);
    }

} // namespace Ifrit::RHI::VulkanRHI2