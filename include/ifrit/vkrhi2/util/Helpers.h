#pragma once

#include "ifrit/core/base/IfritBase.h"

namespace Ifrit::RHI::VulkanRHI2
{
    template <typename T,typename U> IF_FORCEINLINE bool HasFlagBit(T value, U bit) { return (value & bit) != 0; }
} // namespace Ifrit::RHI::VulkanRHI2