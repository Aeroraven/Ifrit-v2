#pragma once

#include "ifrit/core/base/IfritBase.h"

namespace Ifrit::RHI::VulkanRHI2
{
    IF_FORCEINLINE bool HasFlagBit(u32 value, u32 bit) { return (value & bit) != 0; }
} // namespace Ifrit::RHI::VulkanRHI2