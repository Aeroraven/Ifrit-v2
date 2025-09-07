#pragma once
#include "RhiApi.h"
#include "RhiBaseTypes.h"
#include "ifrit/rhi/common/RhiCommandList.h"
#include "ifrit/rhi/common/RhiDevice.h"
#include "ifrit/rhi/common/RhiShaderResource.h"

namespace Ifrit::RHI
{

    class IFRIT_RHI_API RhiDynamicUtils : public RhiDeviceChild
    {
    public:
        virtual ~RhiDynamicUtils()                                                     = default;
        virtual RhiShaderRef CreateShader_RhiInternal(const RhiShaderCreateDesc& desc) = 0;
        virtual RhiSampler*  GetDefaultSampler_RhiInternal()                           = 0;
    };
} // namespace Ifrit::RHI