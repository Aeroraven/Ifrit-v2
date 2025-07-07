#pragma once

#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"

namespace Ifrit::HAL
{
    IFRIT_CORE_API void* MemAllocAligned(usize size, u32 alignment);
    IFRIT_CORE_API void* MemAlloc(usize size);
    IFRIT_CORE_API void  MemFree(void* ptr);

} // namespace Ifrit::HAL