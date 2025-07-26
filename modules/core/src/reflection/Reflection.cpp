#include "ifrit/core/reflection/Reflection.h"
#include <stdexcept>
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::Reflection
{
    struct DynamicReflectionManager
    {
        HashMap<u64, FReflTypeMetaInfo> TypeRegistry;
        HashMap<u64, u64>               TypeIDHashToInternalHash;
    };
    IFRIT_APIDECL DynamicReflectionManager& GetDynamicReflectionManager()
    {
        static DynamicReflectionManager instance;
        return instance;
    }

    IFRIT_APIDECL void Internal_RegisterType(const FReflTypeMetaInfo& typeInfo, std::type_info const& typeInfoStd)
    {

        auto& manager                                             = GetDynamicReflectionManager();
        manager.TypeRegistry[typeInfo.Hash]                       = typeInfo;
        manager.TypeIDHashToInternalHash[typeInfoStd.hash_code()] = typeInfo.Hash;
    }

    IFRIT_APIDECL TReflObject<ObjectImpl> Internal_Construct(u64 typeHash)
    {
        auto& manager = GetDynamicReflectionManager();

        u64   hash = typeHash;
        auto  it   = manager.TypeRegistry.find(hash);
        if (it != manager.TypeRegistry.end())
        {

            const FReflTypeMetaInfo& typeInfo = it->second;
            if (typeInfo.Constructor)
            {

                return { typeInfo.Constructor(), hash };
            }
        }

        IF_LOG_CRITICAL("Reflector", "Type not registered for construction: {}", typeHash);
    }

    IFRIT_CORE_API void Internal_RegisterPropertyField(const FReflTypeMetaInfo& typeInfo,
        const FReflPropertyMetaInfo& propInfo, const String& propertyName, Fn<ObjectImpl(ObjectImpl&)> accessor)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = typeInfo.Hash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {

            manager.TypeRegistry[typeHash].PropertyFields[propInfo.Hash] = { propertyName, accessor };
        }
        else
        {

            IF_LOG_CRITICAL("Reflector", "Type not registered for property field: {}", typeHash);
        }
    }

    IFRIT_CORE_API ObjectImpl Internal_GetProperty(TReflObject<ObjectImpl>& obj, u64 propertyHash)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = obj.TypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            if (it->second.PropertyFields.count(propertyHash) > 0)
            {
                auto& propertyField = it->second.PropertyFields[propertyHash];
                return propertyField.Accessor(obj.ObjectValue);
            }
            else
            {
                IF_LOG_CRITICAL("Reflector", "Property not found: {}", propertyHash);
            }
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for property access: {}", typeHash);
        }
    }
    IFRIT_CORE_API HashMap<u64, FPropertyField>& Internal_GetPropertyList(TReflObject<ObjectImpl>& obj)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = obj.TypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            return it->second.PropertyFields;
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for property list access: {}", typeHash);
            throw std::runtime_error("Type not registered for property list access");
        }
    }
    IFRIT_CORE_API TReflObject<ObjectImpl> Internal_Reference(void* target, std::type_info const& typeInfo)
    {
        if (!target)
        {
            IF_LOG_CRITICAL("Reflector", "Cannot create reference to null pointer");
        }

        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = typeInfo.hash_code();
        auto  it       = manager.TypeIDHashToInternalHash.find(typeHash);
        if (it != manager.TypeIDHashToInternalHash.end())
        {
            u64        internalHash = it->second;
            auto&      metaInfo     = manager.TypeRegistry[internalHash];
            ObjectImpl obj          = ObjectImpl::CreateProxy(target, metaInfo.MetaInfo);
            return { std::move(obj), internalHash };
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for reference: {}", typeHash);
        }
    }

} // namespace Ifrit::Reflection