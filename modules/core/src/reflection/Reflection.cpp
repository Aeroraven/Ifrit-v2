#include "ifrit/core/reflection/Reflection.h"
#include <stdexcept>
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/reflection/RttiIdentifier.h"
namespace Ifrit::Reflection
{
    struct DynamicReflectionManager
    {
        THashMap<u64, FReflTypeMetaInfo> TypeRegistry;
        THashMap<u64, u64>               TypeIDHashToInternalHash;
    };
    IFRIT_APIDECL DynamicReflectionManager& GetDynamicReflectionManager()
    {
        static DynamicReflectionManager instance;
        return instance;
    }

    IFRIT_APIDECL void Internal_RegisterType(const FReflTypeMetaInfo& typeInfo, std::type_info const& typeInfoStd)
    {

        auto& manager = GetDynamicReflectionManager();
        if (manager.TypeRegistry.count(typeInfo.Hash) > 0) IF_UNLIKELY
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
            manager.TypeRegistry[baseTypeHash].DerivedTypes.push_back(derivedTypeHash);
            manager.TypeRegistry[derivedTypeHash].Polymorphic = true;
            manager.TypeRegistry[baseTypeHash].Polymorphic    = true;
        }
        else IF_UNLIKELY
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
        const FReflPropertyMetaInfo& propInfo, const String& propertyName, Fn<ObjectImpl(ObjectImpl&)> accessor,
        Fn<void(ObjectImpl&)> uihandle)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = typeInfo.Hash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {

            manager.TypeRegistry[typeHash].PropertyFields[propInfo.Hash].Name          = propertyName;
            manager.TypeRegistry[typeHash].PropertyFields[propInfo.Hash].Accessor      = std::move(accessor);
            manager.TypeRegistry[typeHash].PropertyFields[propInfo.Hash].UIHandle      = std::move(uihandle);
            manager.TypeRegistry[typeHash].PropertyFields[propInfo.Hash].Metadata.Name = propertyName;
        }
        else IF_UNLIKELY
        {

            IF_LOG_CRITICAL("Reflector", "Type not registered for property field: {}", typeHash);
        }
    }

    IFRIT_CORE_API void Internal_RegisterMethodField(const FReflTypeMetaInfo& typeInfo,
        const FReflMethodMetaInfo& methodInfo, const String& methodName,
        Fn<ObjectImpl(ObjectImpl&, const Vec<ObjectImpl>&)> invoker, Fn<void(ObjectImpl&)> uihandle)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = typeInfo.Hash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            manager.TypeRegistry[typeHash].MethodFields[methodInfo.Hash].Name     = methodName;
            manager.TypeRegistry[typeHash].MethodFields[methodInfo.Hash].Invoker  = std::move(invoker);
            manager.TypeRegistry[typeHash].MethodFields[methodInfo.Hash].UIHandle = std::move(uihandle);
        }
        else IF_UNLIKELY
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for method field: {}", typeHash);
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
            else IF_UNLIKELY
            {
                IF_LOG_CRITICAL("Reflector", "Property not found: {}", propertyHash);
            }
        }
        else IF_UNLIKELY
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for property access: {}", typeHash);
        }
    }

    IFRIT_CORE_API ObjectImpl Internal_InvokeMethod(
        TReflObject<ObjectImpl>& obj, u64 methodHash, const Vec<ObjectImpl>& args)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = obj.TypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            if (it->second.MethodFields.count(methodHash) > 0)
            {
                auto& methodField = it->second.MethodFields[methodHash];
                return methodField.Invoke(obj.ObjectValue, args);
            }
            else IF_UNLIKELY
            {
                IF_LOG_CRITICAL("Reflector", "Method not found: {}", methodHash);
            }
        }
        else IF_UNLIKELY
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for method access: {}", typeHash);
        }
    }

    IFRIT_CORE_API const char* Internal_GetFunctionAlias(u64 typeHash, u64 methodHash)
    {
        auto& manager = GetDynamicReflectionManager();
        auto  it      = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            if (it->second.MethodFields.count(methodHash) > 0)
            {
                auto& methodField = it->second.MethodFields[methodHash];
                return methodField.Name.c_str();
            }
            else IF_UNLIKELY
            {
                IF_LOG_CRITICAL("Reflector", "Method not found for alias: {}", methodHash);
            }
        }
        else IF_UNLIKELY
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for method alias retrieval: {}", typeHash);
        }
    }

    IFRIT_CORE_API Vec<Reference<const FReflTypeMetaInfo>> Internal_GetAllDerivedTypes(
        u64 baseTypeHash, bool includeBase)
    {
        auto&                                   manager = GetDynamicReflectionManager();
        Vec<Reference<const FReflTypeMetaInfo>> result;
        auto                                    it = manager.TypeRegistry.find(baseTypeHash);
        if (it != manager.TypeRegistry.end())
        {
            if (includeBase)
            {
                result.push_back(std::cref(it->second));
            }
            for (u64 derivedTypeHash : it->second.DerivedTypes)
            {
                auto derivedIt = manager.TypeRegistry.find(derivedTypeHash);
                if (derivedIt != manager.TypeRegistry.end())
                {
                    result.push_back(std::cref(derivedIt->second));
                }
                else IF_UNLIKELY
                {
                    IF_LOG_CRITICAL("Reflector", "Derived type not found: {}", derivedTypeHash);
                }
                auto subDerivedTypes = Internal_GetAllDerivedTypes(derivedTypeHash, false);
                result.insert(result.end(), subDerivedTypes.begin(), subDerivedTypes.end());
            }
        }
        else IF_UNLIKELY
        {
            IF_LOG_CRITICAL("Reflector", "Base type not registered for derived types retrieval: {}", baseTypeHash);
        }
        return result;
    }

    THashMap<u64, FPropertyField> GetPropertyListRecursive(u64 typeHash)
    {

        auto&                         manager = GetDynamicReflectionManager();
        THashMap<u64, FPropertyField> properties;

        auto                          it = manager.TypeRegistry.find(typeHash);
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

    IFRIT_CORE_API THashMap<u64, FPropertyField> Internal_GetPropertyList(TReflObject<ObjectImpl>& obj)
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
    IFRIT_CORE_API void Internal_ReportWrongFunctionCall()
    {
        IF_LOG_CRITICAL(
            "Reflector", "Wrong function call detected. Please check the function signature and arguments.");
    }

    IFRIT_CORE_API void Internal_PropertyAddHint(
        u64 baseTypeHash, u64 propertyHash, const String& hintName, PropertyHintValueType value)
    {
        auto& manager            = GetDynamicReflectionManager();
        auto& baseType           = manager.TypeRegistry[baseTypeHash];
        auto& property           = baseType.PropertyFields[propertyHash];
        property.Hints[hintName] = value;

        if (hintName == "Editable")
        {
            property.Metadata.Editable = EPropertyEditable::Editable;
        }
        else if (hintName == "Visible")
        {
            property.Metadata.Editable = EPropertyEditable::ReadOnly;
        }
        else if (hintName == "AssetCategory")
        {
            if (std::holds_alternative<std::string>(value))
            {
                auto categoryStr = std::get<std::string>(value);
                if (categoryStr == "Texture")
                {
                    property.Metadata.AssetCategory = EPropertyAssetCategory::Texture;
                }
                else if (categoryStr == "Mesh")
                {
                    property.Metadata.AssetCategory = EPropertyAssetCategory::Mesh;
                }
            }
        }
        else if (hintName == "UISlider.min")
        {
            property.Metadata.UIControl = EPropertyUIControl::UISlider;
            if (std::holds_alternative<f64>(value))
            {
                property.Metadata.UIClampMin = std::get<f64>(value);
            }
            else if (std::holds_alternative<i32>(value))
            {
                property.Metadata.UIClampMin = static_cast<f64>(std::get<i32>(value));
            }
            else
            {
                IF_LOG_WARNING("Reflector", "UISlider.min hint value is not a double: {}", value.index());
            }
        }
        else if (hintName == "UISlider.max")
        {
            property.Metadata.UIControl = EPropertyUIControl::UISlider;
            if (std::holds_alternative<f64>(value))
            {
                property.Metadata.UIClampMax = std::get<f64>(value);
            }
            else if (std::holds_alternative<i32>(value))
            {
                property.Metadata.UIClampMax = static_cast<f64>(std::get<i32>(value));
            }
            else
            {
                IF_LOG_WARNING("Reflector", "UISlider.max hint value is not a double: {}", value.index());
            }
        }
        else if (hintName == "UISelect")
        {
            property.Metadata.UIControl = EPropertyUIControl::UISelect;
        }
        else if (hintName == "UIText")
        {
            property.Metadata.UIControl = EPropertyUIControl::UIText;
        }
        else if (hintName == "UIColor")
        {
            property.Metadata.UIControl = EPropertyUIControl::UIColor;
        }
        else
        {
            IF_LOG_WARNING("Reflector", "Unknown property hint: {}", hintName);
        }
    }

    IFRIT_CORE_API const PropertyMetadata& Internal_GetPropertyMetadata(u64 baseTypeHash, u64 propertyHash)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = baseTypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            const auto& property = it->second.PropertyFields.at(propertyHash);
            return property.Metadata;
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for property metadata access: {}", typeHash);
            throw std::runtime_error("Type not registered for property metadata access");
        }
    }

    Vec<Fn<void()>> Internal_GetPropertyEditorHandlesRecursive(u64 typeHash, TReflObject<ObjectImpl>& obj)
    {
        auto&           manager = GetDynamicReflectionManager();
        Vec<Fn<void()>> handles;
        auto            it = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            const FReflTypeMetaInfo& typeInfo = it->second;
            for (const auto& [_, property] : typeInfo.PropertyFields)
            {
                if (property.UIHandle)
                {
                    handles.push_back([property, &obj]() {
                        auto propEntry = property.Accessor(obj.ObjectValue);
                        property.UIHandle(propEntry);
                    });
                }
            }

            for (u64 baseTypeHash : typeInfo.BaseTypes)
            {
                auto baseHandles = Internal_GetPropertyEditorHandlesRecursive(baseTypeHash, obj);
                handles.insert(handles.end(), baseHandles.begin(), baseHandles.end());
            }
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for property editor handles: {}", typeHash);
        }
        return handles;
    }

    IFRIT_CORE_API Vec<Fn<void()>> Internal_GetPropertyEditorHandles(TReflObject<ObjectImpl>& obj)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = obj.TypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            const FReflTypeMetaInfo& typeInfo = it->second;
            Vec<Fn<void()>>          handles;
            for (const auto& [_, property] : typeInfo.PropertyFields)
            {
                if (property.UIHandle)
                {
                    handles.push_back([property, &obj]() {
                        auto propEntry = property.Accessor(obj.ObjectValue);
                        property.UIHandle(propEntry);
                    });
                }
            }
            for (u64 baseTypeHash : typeInfo.BaseTypes)
            {
                auto baseHandles = Internal_GetPropertyEditorHandlesRecursive(baseTypeHash, obj);
                handles.insert(handles.end(), baseHandles.begin(), baseHandles.end());
            }
            return handles;
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for property editor handles: {}", typeHash);
        }
    }

    Vec<Fn<void()>> Internal_GetMethodEditorHandlesRecursive(u64 typeHash, TReflObject<ObjectImpl>& obj)
    {
        auto&           manager = GetDynamicReflectionManager();
        Vec<Fn<void()>> handles;
        auto            it = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            const FReflTypeMetaInfo& typeInfo = it->second;
            for (const auto& [_, method] : typeInfo.MethodFields)
            {
                if (method.UIHandle)
                {
                    handles.push_back([method, &obj]() { method.UIHandle(obj.ObjectValue); });
                }
            }

            for (u64 baseTypeHash : typeInfo.BaseTypes)
            {
                auto baseHandles = Internal_GetMethodEditorHandlesRecursive(baseTypeHash, obj);
                handles.insert(handles.end(), baseHandles.begin(), baseHandles.end());
            }
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for method editor handles: {}", typeHash);
        }
        return handles;
    }

    IFRIT_CORE_API Vec<Fn<void()>> Internal_GetMethodEditorHandles(TReflObject<ObjectImpl>& obj)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = obj.TypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            const FReflTypeMetaInfo& typeInfo = it->second;
            Vec<Fn<void()>>          handles;
            for (const auto& [_, method] : typeInfo.MethodFields)
            {
                if (method.UIHandle)
                {
                    handles.push_back([method, &obj]() { method.UIHandle(obj.ObjectValue); });
                }
            }
            for (u64 baseTypeHash : typeInfo.BaseTypes)
            {
                auto baseHandles = Internal_GetMethodEditorHandlesRecursive(baseTypeHash, obj);
                handles.insert(handles.end(), baseHandles.begin(), baseHandles.end());
            }
            return handles;
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for method editor handles: {}", typeHash);
        }
    }

    IFRIT_CORE_API Vec<Fn<ObjectImpl(const Vec<ObjectImpl>&)>> Internal_GetRegisteredFuncs(TReflObject<ObjectImpl>& obj)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = obj.TypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            const FReflTypeMetaInfo&                    typeInfo = it->second;
            Vec<Fn<ObjectImpl(const Vec<ObjectImpl>&)>> funcs;
            for (const auto& [_, method] : typeInfo.MethodFields)
            {
                funcs.push_back(
                    [method, &obj](const Vec<ObjectImpl>& args) { return method.Invoke(obj.ObjectValue, args); });
            }
            return funcs;
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for function retrieval: {}", typeHash);
        }
    }

    IFRIT_CORE_API u32 Internal_GetNumRegisteredFuncs(TReflObject<ObjectImpl>& obj)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = obj.TypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            const FReflTypeMetaInfo& typeInfo = it->second;
            return static_cast<u32>(typeInfo.MethodFields.size());
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for function count: {}", typeHash);
        }
        return 0;
    }

    IFRIT_CORE_API u32 Internal_GetNumVisibleProperties(TReflObject<ObjectImpl>& obj)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = obj.TypeHash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            const FReflTypeMetaInfo& typeInfo = it->second;
            u32                      count    = 0;
            for (const auto& [_, property] : typeInfo.PropertyFields)
            {
                if (property.Metadata.Editable != EPropertyEditable::None)
                {
                    count++;
                }
            }
        }
        else
        {
            IF_LOG_CRITICAL("Reflector", "Type not registered for property count: {}", typeHash);
        }
        return 0;
    }
} // namespace Ifrit::Reflection