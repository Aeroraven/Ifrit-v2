#include "ifrit/core/reflection/Reflection.h"
#include <stdexcept>
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/reflection/RttiIdentifier.h"
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

        auto& manager = GetDynamicReflectionManager();
        if (manager.TypeRegistry.count(typeInfo.Hash) > 0)
        {
            IF_LOG_WARNING("Reflector", "Type already registered: {}", typeInfo.Hash);
            return;
        }
        manager.TypeRegistry[typeInfo.Hash]                          = typeInfo;
        manager.TypeIDHashToInternalHash[GetTypeIDHash(typeInfoStd)] = typeInfo.Hash;
    }

    IFRIT_CORE_API void Internal_RegisterPolymorphic(u64 baseTypeHash, u64 derivedTypeHash)
    {
        auto& manager = GetDynamicReflectionManager();
        if (manager.TypeRegistry.count(baseTypeHash) > 0 && manager.TypeRegistry.count(derivedTypeHash) > 0)
        {
            manager.TypeRegistry[derivedTypeHash].BaseTypes.push_back(baseTypeHash);
            manager.TypeRegistry[derivedTypeHash].Polymorphic = true;
            manager.TypeRegistry[baseTypeHash].Polymorphic    = true;
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Base or derived type not registered for polymorphic relation: {} -> {}",
                baseTypeHash, derivedTypeHash);
        }
    }

    IFRIT_CORE_API bool Internal_TypeOnInheritanceChain(u64 baseTypeHashToSearch, u64 derivedTypeHash)
    {
        auto& manager = GetDynamicReflectionManager();
        if (manager.TypeRegistry.count(derivedTypeHash) > 0)
        {
            const auto& derivedTypeInfo = manager.TypeRegistry[derivedTypeHash];
            for (auto& p : derivedTypeInfo.BaseTypes)
            {
                if (p == baseTypeHashToSearch)
                    return true;
            }
            for (u64 baseTypeHash : derivedTypeInfo.BaseTypes)
            {
                if (Internal_TypeOnInheritanceChain(baseTypeHashToSearch, baseTypeHash))
                {
                    return true;
                }
            }
        }
        return false;
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

    HashMap<u64, FPropertyField> GetPropertyListRecursive(u64 typeHash)
    {

        auto&                        manager = GetDynamicReflectionManager();
        HashMap<u64, FPropertyField> properties;

        auto                         it = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            const FReflTypeMetaInfo& typeInfo = it->second;
            properties.insert(typeInfo.PropertyFields.begin(), typeInfo.PropertyFields.end());

            for (u64 baseTypeHash : typeInfo.BaseTypes)
            {
                // IF_LOG_DEBUG("Reflector", "Adding properties from base type: {}", baseTypeHash);
                auto baseProperties = GetPropertyListRecursive(baseTypeHash);
                properties.insert(baseProperties.begin(), baseProperties.end());
            }
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for property list access: {}", typeHash);
        }

        return properties;
    }

    IFRIT_CORE_API HashMap<u64, FPropertyField> Internal_GetPropertyList(TReflObject<ObjectImpl>& obj)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = obj.TypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            return GetPropertyListRecursive(typeHash);
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for property list access: {}", typeHash);
        }
    }
    IFRIT_CORE_API TReflObject<ObjectImpl> Internal_Reference(void* target, std::type_info const& typeInfo)
    {
        if (!target)
        {
            IF_LOG_CRITICAL("Reflector", "Cannot create reference to null pointer");
        }

        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = GetTypeIDHash(typeInfo);
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

    IFRIT_CORE_API u64 Internal_GetTypeHashFromTypeInfoHash(u64 typeInfoHash)
    {
        auto& manager = GetDynamicReflectionManager();
        auto  it      = manager.TypeIDHashToInternalHash.find(typeInfoHash);
        if (it != manager.TypeIDHashToInternalHash.end())
        {
            return it->second;
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for hash lookup: {}", typeInfoHash);
            throw std::runtime_error("Type not registered for hash lookup");
        }
    }
    IFRIT_CORE_API void Internal_IgnoreNonVirtualInhertance()
    {
        // This function is a placeholder for future implementation
        // It can be used to ignore non-virtual inheritance in the reflection system
        IF_LOG_WARNING("Reflector", "Ignoring non-virtual inheritance, which is not implemented yet");
    }

} // namespace Ifrit::Reflection