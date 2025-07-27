#include "ifrit/core/reflection/Serializer.h"
#include "ifrit/core/reflection/Reflection.h"
namespace Ifrit::Reflection
{
    IFRIT_APIDECL void InvokeSerializeDynamicImpl(void* ptr, const std::type_info& typeInfo, Archive* archive)
    {
        auto reflObj    = Internal_Reference(ptr, typeInfo);
        auto properties = GetPropertyList(reflObj);
        archive->BeginObject("__ifrit_reflection_type");
        archive->Serialize(String(typeInfo.name()));
        archive->EndObject();
        archive->BeginObject("__ifrit_reflection_id");
        archive->Serialize(typeInfo.hash_code());
        archive->EndObject();
        for (const auto& [k, v] : properties)
        {
            archive->BeginObject(k);
            v.Value().Serialize(archive);
            archive->EndObject();
        }
    }

    IFRIT_APIDECL void InvokeDeserializeDynamicImpl(void* ptr, const std::type_info& typeInfo, Archive* archive)
    {
        auto reflObj    = Internal_Reference(ptr, typeInfo);
        auto properties = GetPropertyList(reflObj);
        for (const auto& [k, v] : properties)
        {
            if (archive->HasObject(k))
            {
                archive->BeginObject(k);
                v.Value().Deserialize(archive);
                archive->EndObject();
            }
            else
            {
                IF_LOG_WARNING("Reflector",
                    "During deserialization, key {} does not present in object with type {}. The serialized archive might be corrupted",
                    k, typeInfo.name());
            }
        }
    }
} // namespace Ifrit::Reflection
