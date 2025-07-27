#include "ifrit/core/reflection/Object.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/reflection/Serializer.h"
#include "ifrit/core/reflection/Reflection.h"
#include "ifrit/core/reflection/RttiIdentifier.h"

namespace Ifrit::Reflection
{
    IFRIT_APIDECL void Object::Serialize(Archive* archive) const
    {
        IF_LOG_ASSERTION("Reflector", archive != nullptr, "Archive must not be null");
        IF_LOG_ASSERTION(
            "Reflector", SerializeInterface != nullptr, "SerializeInterface must not be null for object serialization");

        SerializeInterface(archive, Ptr);
    }

    IFRIT_APIDECL void Object::Deserialize(Archive* archive) const
    {
        IF_LOG_ASSERTION("Reflector", archive != nullptr, "Archive must not be null");
        IF_LOG_ASSERTION("Reflector", DeserializeInterface != nullptr,
            "DeserializeInterface must not be null for object deserialization");
        DeserializeInterface(archive, Ptr);
    }

    IFRIT_APIDECL void ObjectBadCastReport(const String& expected, const String& requested)
    {
        IF_LOG_CRITICAL("Reflector", "Bad cast from {} to {}", expected, requested);
        throw std::bad_cast();
    }

    IFRIT_CORE_API bool IsValidObjectCast(const std::type_info& fromType, const std::type_info& toType)
    {
        if (fromType == toType)
        {
            return true;
        }
        auto internalFromHash = Internal_GetTypeHashFromTypeInfoHash(GetTypeIDHash(fromType));
        auto internalToHash   = Internal_GetTypeHashFromTypeInfoHash(GetTypeIDHash(toType));
        if (Internal_TypeOnInheritanceChain(internalToHash, internalFromHash))
        {
            return true;
        }
        return false;
    }
} // namespace Ifrit::Reflection