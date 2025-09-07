#include "ifrit/vkrhi2/adapter/DynamicUtils.h"
#include "ifrit/vkrhi2/adapter/MemoryResource.h"

namespace Ifrit::RHI::VulkanRHI2
{
    // ===== Dynamic Utils =====
    IFRIT_VKRHI2_API RhiSampler* VA_DynamicUtils::GetDefaultSampler_RhiInternal()
    {
        auto device   = static_cast<VA_Device*>(mContext);
        auto registry = device->GetSamplerRegistry()->GetGlobalSampler(ERhiGlobalSamplerType::PointWrap);
        return registry;
    }
} // namespace Ifrit::RHI::VulkanRHI2