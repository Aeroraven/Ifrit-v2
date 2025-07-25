#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/typing/Util.h"
#include <any>

namespace Ifrit::Reflection
{
    using TAny = std::any;

    struct FTypeMetaInfo
    {
        const char* Name;
        u64         Hash;
        Fn<TAny()>  Constructor;

        FTypeMetaInfo() = default;
        FTypeMetaInfo(const char* name, u64 hash, Fn<TAny()>&& constructor = nullptr)
            : Name(name), Hash(hash), Constructor(std::move(constructor))
        {
        }
        bool operator==(const FTypeMetaInfo& other) const { return Hash == other.Hash; }
        bool operator!=(const FTypeMetaInfo& other) const { return !(*this == other); }

        template <typename T> static FTypeMetaInfo Create()
        {
            auto constructor = []() -> TAny { return std::make_any<T>(); };
            return FTypeMetaInfo(TTypeInfo<T>::Name, TTypeInfo<T>::Hash, std::move(constructor));
        }
    };

    template <typename T> struct TTypeMetaInfo
    {
        static constexpr const char* Name = TTypeInfo<T>::Name;
        static constexpr u64         Hash = TTypeInfo<T>::Hash;
    };

    template <typename T> struct TMemberType
    {
        using Type = T;
    };
    template <typename U, typename T> struct TMemberType<U T::*>
    {
        using Type      = U;
        using ClassType = T;
    };
    template <typename T>
    concept IConceptIsMemberPointer = requires(T t) {
        typename TMemberType<T>::Type;
        typename TMemberType<T>::ClassType;
    };

    template <auto Member>
        requires IConceptIsMemberPointer<decltype(Member)>
    class TAutoMemberAccessor
    {
    public:
        using ClassType  = TMemberType<decltype(Member)>::ClassType;
        using MemberType = TMemberType<decltype(Member)>::Type;

        static MemberType&     Get(ClassType& obj) { return obj.*Member; }

        static Fn<TAny(TAny&)> GetMemberAccessor()
        {
            return [](TAny& classObj) -> TAny {
                auto& obj = std::any_cast<ClassType&>(classObj);
                return std::ref(Get(obj));
            };
        }
    };

    IFRIT_CORE_API void Internal_RegisterType(const FTypeMetaInfo& typeInfo);
    IFRIT_CORE_API TAny Internal_Construct(String typeName);
    IFRIT_CORE_API void Internal_RegisterPropertyField(
        const FTypeMetaInfo& typeInfo, const String& propertyName, Fn<TAny(TAny&)> accessor);

    template <typename T> inline void RegisterType() { Internal_RegisterType(FTypeMetaInfo::Create<T>()); }
    template <auto Member>
        requires IConceptIsMemberPointer<decltype(Member)>
    inline void RegisterPropertyField(const String& propertyName)
    {
        using Accessor = TAutoMemberAccessor<Member>;
        auto typeInfo  = FTypeMetaInfo::Create<typename Accessor::ClassType>();
        Internal_RegisterPropertyField(typeInfo, propertyName, Accessor::GetMemberAccessor());
    }
    inline TAny ConstructObject(String typeName) { return Internal_Construct(typeName); }

} // namespace Ifrit::Reflection