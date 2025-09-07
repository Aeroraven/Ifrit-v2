#include "ifrit/core/hal/HalHostConcurrency.h"
#include "ifrit/core/logging/Logging.h"
#include <thread>

namespace Ifrit::HAL
{
    thread_local static u32 sCurrentThreadId = ~0u;
    IFRIT_APIDECL u32       GetCurrentThreadId() IF_NOEXCEPT
    {
        if (sCurrentThreadId == ~0u)
        {
            IF_LOG_WARNING("HalHostConcurrency", "Current thread ID is not set, using default value.");
        }
        return sCurrentThreadId;
    }
    IFRIT_APIDECL u32  GetMaxHardwareConcurrency() IF_NOEXCEPT { return std::thread::hardware_concurrency(); }
    IFRIT_APIDECL void SetCurrentThreadId(u32 threadId) IF_NOEXCEPT { sCurrentThreadId = threadId; }

} // namespace Ifrit::HAL