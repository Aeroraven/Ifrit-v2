#pragma once

#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"

namespace Ifrit::HAL
{
    IFRIT_CORE_API u32              GetCurrentThreadId() IF_NOEXCEPT;
    IFRIT_CORE_API u32              GetMaxHardwareConcurrency() IF_NOEXCEPT;

    IFRIT_CORE_API void             SetCurrentThreadId(u32 threadId) IF_NOEXCEPT;

    IF_FORCEINLINE IF_CONSTEVAL u32 GetMaxThreadLimit() IF_NOEXCEPT { return 32; }

} // namespace Ifrit::HAL