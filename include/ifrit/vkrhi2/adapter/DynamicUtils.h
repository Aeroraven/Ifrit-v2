
#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/rhi/common/RhiDynamicUtils.h"

#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{

    class IFRIT_VKRHI2_API VA_DynamicUtils : public RHI::RhiDynamicUtils
    {
    public:
        VA_DynamicUtils(VA_Device* device);
        virtual RhiShaderRef CreateShader_RhiInternal(const RhiShaderCreateDesc& desc) override;
        virtual RhiSampler*  GetDefaultSampler_RhiInternal() override;
        virtual SizedBuffer  GetRootConstantData_RhiInternal(
             RhiShaderVariant* variant, const RhiShaderParameter& params) override;
    };

} // namespace Ifrit::RHI::VulkanRHI2