#include "ifrit/vkrhi2/adapter/DynamicUtils.h"
#include "ifrit/vkrhi2/adapter/Shader.h"

namespace Ifrit::RHI::VulkanRHI2
{
    // ===== Dynamic Utils =====
    IFRIT_APIDECL                 VA_DynamicUtils::VA_DynamicUtils(VA_Device* device) { mContext = device; }
    IFRIT_VKRHI2_API RhiShaderRef VA_DynamicUtils::CreateShader_RhiInternal(const RhiShaderCreateDesc& desc)
    {
        VA_Shader* shader = new VA_Shader(static_cast<VA_Device*>(mContext), desc);
        return MakeCountRef<RhiShader>(shader);
    }
    IFRIT_VKRHI2_API SizedBuffer VA_DynamicUtils::GetRootConstantData_RhiInternal(
        RhiShaderVariant* variant, const RhiShaderParameter& params)
    {
        auto vaVariant = static_cast<VA_ShaderVariant*>(variant);
        return vaVariant->GetRootConstantData(params);
    }
} // namespace Ifrit::RHI::VulkanRHI2