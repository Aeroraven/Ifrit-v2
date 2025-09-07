#pragma once
#include <vulkan/vulkan.h>
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/vkrhi2/adapter/Device.h"

namespace Ifrit::RHI::VulkanRHI2
{
    IF_FORCEINLINE u32 GetQueueFamilyIndex(VA_Device* device, ERhiPipelineType type)
    {
        auto info = device->GetActiveQueueFamilies();
        switch (type)
        {
            case ERhiPipelineType::Graphics:
                return info.mGraphics;
            case ERhiPipelineType::Compute:
                return info.mAsyncCompute;
            case ERhiPipelineType::Transfer:
                return info.mTransfer;
            default:
                IF_LOG_CRITICAL("VA_Device", "Unknown pipeline type");
                std::abort();
        }
    }
} // namespace Ifrit::RHI::VulkanRHI2