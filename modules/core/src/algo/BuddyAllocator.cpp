#include "ifrit/core/algo/BuddyAllocator.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit
{
    IFRIT_APIDECL void BuddyAllocatorReportCritical(const String& msg)
    { 
        IF_LOG_CRITICAL("BuddyAllocator", "{}",msg);
    }
} // namespace Ifrit