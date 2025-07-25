#include "ifrit/core/reflection/Reflection.h"
#include <stdexcept>
namespace Ifrit::Reflection
{
    struct DynamicReflectionManager
    {
        HashMap<u64, FTypeMetaInfo>                     TypeRegistry;
        HashMap<u64, HashMap<String, Fn<TAny(TAny&)>>> PropertyAccessors;
        HashMap<String, u64>                            TypeNameToHash;
    };

    DynamicReflectionManager& GetDynamicReflectionManager()
    {
        static DynamicReflectionManager instance;
        return instance;
    }

    IFRIT_APIDECL void Internal_RegisterType(const FTypeMetaInfo& typeInfo)
    {
        auto& manager                            = GetDynamicReflectionManager();
        manager.TypeRegistry[typeInfo.Hash]      = typeInfo;
        manager.TypeNameToHash[typeInfo.Name]    = typeInfo.Hash;
        manager.PropertyAccessors[typeInfo.Hash] = HashMap<String, Fn<TAny(TAny&)>>();
    }

    String GetActualFuncSigName(String typeName)
    {
        return "const char *__cdecl Ifrit::GetFuncName<class " + typeName + ">(void)";
    }

    IFRIT_APIDECL TAny Internal_Construct(String typeName)
    {
        auto& manager = GetDynamicReflectionManager();
        u64   hash    = manager.TypeNameToHash[GetActualFuncSigName(typeName)];
        auto  it      = manager.TypeRegistry.find(hash);
        if (it != manager.TypeRegistry.end())
        {
            const FTypeMetaInfo& typeInfo = it->second;
            if (typeInfo.Constructor)
            {
                return typeInfo.Constructor();
            }
        }
        throw std::runtime_error("Type not registered: " + typeName);
    }

    IFRIT_CORE_API void Internal_RegisterPropertyField(
        const FTypeMetaInfo& typeInfo, const String& propertyName, Fn<TAny(TAny&)> accessor)
    {
        auto& manager  = GetDynamicReflectionManager();
        u64   typeHash = typeInfo.Hash;
        auto  it       = manager.TypeRegistry.find(typeHash);
        if (it != manager.TypeRegistry.end())
        {
            auto& propertyAccessors         = manager.PropertyAccessors[typeHash];
            propertyAccessors[propertyName] = accessor;
        }
        else
        {
            throw std::runtime_error("Type not registered: " + String(typeInfo.Name));
        }
    }

} // namespace Ifrit::Reflection