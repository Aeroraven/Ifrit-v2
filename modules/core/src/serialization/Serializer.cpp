#include "ifrit/core/serialization/Serializer.h"
#include "ifrit/core/logging/Logging.h"

namespace Ifrit::Serialization
{
    IFRIT_APIDECL void SerializationErrorReport(const String& str)
    {
        IF_LOG_ERROR("Serialization", "Serialization error: {}", str);
    }
} // namespace Ifrit::Serialization