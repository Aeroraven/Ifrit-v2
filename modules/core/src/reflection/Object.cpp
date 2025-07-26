#include "ifrit/core/reflection/Object.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/reflection/Serializer.h"

namespace Ifrit::Reflection
{
    IFRIT_APIDECL void Object::Serialize(Archive* archive) const
    {
        IF_LOG_ASSERTION("Reflector", archive != nullptr, "Archive must not be null");
        IF_LOG_ASSERTION(
            "Reflector", SerializeInterface != nullptr, "SerializeInterface must not be null for object serialization");

        SerializeInterface(archive, Ptr);
    }
} // namespace Ifrit::Reflection