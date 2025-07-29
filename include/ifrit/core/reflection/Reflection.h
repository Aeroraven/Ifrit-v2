#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/typing/TypeMetaInfo.h"
#include "ifrit/core/reflection/Object.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/core/reflection/TypeMetaExtended.h"
#include <ranges>
#include <variant>
#include "ifrit/core/reflection/RttiIdentifier.h"
#include "ifrit/core/reflection/PropertyMeta.h"
#include "ifrit/core/reflection/PropertyUIControl.h"

namespace Ifrit::Reflection
{
    using ObjectImpl = Object;

    using PropertyHintValueType = std::variant<i32, f64, String>;

    struct FPropertyWrapper
    {
        Object                   Prop;

        template <typename T> T& As() const { return Prop.As<T>(); }
        Object&                  Value() { return Prop; }
        const Object&            Value() const { return Prop; }
    };
    using PropertyImpl = FPropertyWrapper;

    struct FHashedString
    {
        String Value;
        u64    Hash;
        FHashedString(const String& value) : Value(value), Hash(std::hash<String>()(value)) {}
        bool operator==(const FHashedString& other) const { return Hash == other.Hash; }
    };

} // namespace Ifrit::Reflection

namespace std
{
    template <> struct hash<Ifrit::Reflection::FHashedString>
    {
        std::size_t operator()(const Ifrit::Reflection::FHashedString& hs) const noexcept { return hs.Hash; }
    };
} // namespace std

namespace Ifrit::Reflection
{

    struct FPropertyField
    {
        String                                 Name;
        Fn<ObjectImpl(ObjectImpl&)>            Accessor;
        Fn<void(ObjectImpl&)>                  UIHandle;
        HashMap<String, PropertyHintValueType> Hints;
        PropertyMetadata                       Metadata;

        FPropertyWrapper GetProperty(ObjectImpl& obj) const { return FPropertyWrapper{ Accessor(obj) }; }
    };

    struct FPropertyKVPair
    {
        String           Name;
        FPropertyWrapper Value;
    };

    template <typename T> struct TReflObject
    {
        T   ObjectValue;
        u64 TypeHash;
    };

    template <typename T> struct TReflTypeMetaInfo
    {
        static constexpr StringView Name                 = TMetaTypeInfo<T>::Name;
        static constexpr u64        Hash                 = TMetaTypeInfo<T>::Hash;
        static constexpr bool       DefaultConstructible = std::is_default_constructible_v<T>;
    };

    template <auto T> struct TReflPropMetaInfo
    {
        static constexpr StringView Name = TMetaPropertyInfo<T>::Name;
        static constexpr u64        Hash = TMetaPropertyInfo<T>::Hash;
    };

    struct FReflTypeMetaInfo
    {
        StringView                   Name;
        u64                          Hash;
        Fn<ObjectImpl()>             Constructor = nullptr;
        HashMap<u64, FPropertyField> PropertyFields;
        FMetaTypeExtendedInfo        MetaInfo;
        bool                         Polymorphic = false;
        Vec<u64>                     BaseTypes;

        FReflTypeMetaInfo() = default;
        bool operator==(const FReflTypeMetaInfo& other) const { return Hash == other.Hash; }
        bool operator!=(const FReflTypeMetaInfo& other) const { return !(*this == other); }

        template <typename T> static FReflTypeMetaInfo Create()
        {
            FReflTypeMetaInfo metaInfo;
            metaInfo.Name = TReflTypeMetaInfo<T>::Name;
            metaInfo.Hash = TReflTypeMetaInfo<T>::Hash;
            if constexpr (TReflTypeMetaInfo<T>::DefaultConstructible)
            {
                auto constructor     = []() -> ObjectImpl { return ObjectImpl::Create<T>(); };
                metaInfo.Constructor = std::move(constructor);
            }
            metaInfo.MetaInfo = FMetaTypeExtendedInfo::Create<T>();
            return metaInfo;
        }
    };

    struct FReflPropertyMetaInfo
    {
        FReflTypeMetaInfo ClassTypeInfo;
        FReflTypeMetaInfo MemberTypeInfo;
        StringView        Name;
        u64               Hash;

        FReflPropertyMetaInfo() = default;

        template <auto Member>
            requires IConceptIsMemberPointer<decltype(Member)>
        static FReflPropertyMetaInfo Create()
        {
            using Accessor = TMemberType<decltype(Member)>;
            using PropMeta = TReflPropMetaInfo<Member>;

            FReflPropertyMetaInfo meta;
            meta.ClassTypeInfo  = FReflTypeMetaInfo::Create<typename Accessor::ClassType>();
            meta.MemberTypeInfo = FReflTypeMetaInfo::Create<typename Accessor::Type>();
            meta.Name           = PropMeta::Name;
            meta.Hash           = PropMeta::Hash;
            return meta;
        }
    };

    // Internal API
    IFRIT_CORE_API void Internal_RegisterType(const FReflTypeMetaInfo& typeInfo, std::type_info const& typeInfoStd);
    IFRIT_CORE_API TReflObject<ObjectImpl> Internal_Construct(u64 typeHash);
    IFRIT_CORE_API void                    Internal_RegisterPropertyField(const FReflTypeMetaInfo& typeInfo,
                           const FReflPropertyMetaInfo& propInfo, const String& propertyName, Fn<ObjectImpl(ObjectImpl&)> accessor,
                           Fn<void(ObjectImpl&)> uihandle);
    IFRIT_CORE_API ObjectImpl              Internal_GetProperty(TReflObject<ObjectImpl>& obj, u64 propertyHash);
    IFRIT_CORE_API HashMap<u64, FPropertyField> Internal_GetPropertyList(TReflObject<ObjectImpl>& obj);
    IFRIT_CORE_API TReflObject<ObjectImpl> Internal_Reference(void* target, std::type_info const& typeInfo);
    IFRIT_CORE_API u64                     Internal_GetTypeHashFromTypeInfoHash(u64 typeInfoHash);
    IFRIT_CORE_API void                    Internal_RegisterPolymorphic(u64 baseTypeHash, u64 derivedTypeHash);
    IFRIT_CORE_API bool Internal_TypeOnInheritanceChain(u64 baseTypeHashToSearch, u64 derivedTypeHash);
    IFRIT_CORE_API void Internal_IgnoreNonVirtualInhertance();
    IFRIT_CORE_API void Internal_PropertyAddHint(
        u64 baseTypeHash, u64 propertyHash, const String& hintName, PropertyHintValueType value);
    IFRIT_CORE_API const PropertyMetadata& Internal_GetPropertyMetadata(u64 baseTypeHash, u64 propertyHash);
    IFRIT_CORE_API Vec<Fn<void()>> Internal_GetPropertyEditorHandles(TReflObject<ObjectImpl>& obj);
    IFRIT_CORE_API u32             Internal_GetNumVisibleProperties(TReflObject<ObjectImpl>& obj);

    template <auto Member>
        requires IConceptIsMemberPointer<decltype(Member)>
    class TAutoMemberAccessor
    {
    public:
        using ClassType  = TMemberType<decltype(Member)>::ClassType;
        using MemberType = TMemberType<decltype(Member)>::Type;

        static MemberType&                 Get(ClassType& obj) { return obj.*Member; }

        static Fn<ObjectImpl(ObjectImpl&)> GetMemberAccessor()
        {
            return [](ObjectImpl& classObj) -> ObjectImpl {
                auto& obj = classObj.As<ClassType>();
                return ObjectImpl::Create(std::ref(Get(obj)));
            };
        }
        static Fn<void(ObjectImpl&)> GetUIHandle()
        {
            return [](ObjectImpl& prop) {
                if constexpr (IConceptEditableType<MemberType> || std::is_enum_v<MemberType>)
                {
                    auto                    typeInfo = FReflTypeMetaInfo::Create<ClassType>();
                    auto                    propInfo = FReflPropertyMetaInfo::Create<Member>();
                    const PropertyMetadata& metadata = Internal_GetPropertyMetadata(typeInfo.Hash, propInfo.Hash);
                    ProcessPropertyMetadata<MemberType>(metadata, prop);
                }
                else
                {
                    // IF_LOG_WARNING("Reflector", "UI Handle not available for non-editable type: {}",
                    //     String(typeid(MemberType).name()));
                }
            };
        }
    };

    // Templates
    template <typename T> inline void RegisterType()
    {
        Internal_RegisterType(FReflTypeMetaInfo::Create<T>(), typeid(T));
    }

    template <typename Derived, typename Base>
        requires(!std::is_base_of_v<Base, Derived> || !std::is_polymorphic_v<Base>)
    inline void RegisterPolymorphicRelation()
    {
        Internal_IgnoreNonVirtualInhertance();
    }

    template <typename Derived, typename Base>
        requires std::is_base_of_v<Base, Derived> && std::is_polymorphic_v<Base>
    inline void RegisterPolymorphicRelation()
    {
        u64 baseTypeHash    = Internal_GetTypeHashFromTypeInfoHash(GetTypeIDHash(typeid(Base)));
        u64 derivedTypeHash = Internal_GetTypeHashFromTypeInfoHash(GetTypeIDHash(typeid(Derived)));
        Internal_RegisterPolymorphic(baseTypeHash, derivedTypeHash);
    }

    template <auto Member>
        requires IConceptIsMemberPointer<decltype(Member)>
    inline void RegisterPropertyField(const String& propertyName)
    {
        using Accessor = TAutoMemberAccessor<Member>;

        auto typeInfo = FReflTypeMetaInfo::Create<typename Accessor::ClassType>();
        auto propInfo = FReflPropertyMetaInfo::Create<Member>();
        Internal_RegisterPropertyField(
            typeInfo, propInfo, propertyName, Accessor::GetMemberAccessor(), Accessor::GetUIHandle());
    }
    inline TReflObject<ObjectImpl> ConstructObject(FMetaTypeInfo typeMeta) { return Internal_Construct(typeMeta.Hash); }
    inline FPropertyWrapper        GetProperty(TReflObject<ObjectImpl>& obj, FMetaPropertyInfo propMeta)
    {
        return FPropertyWrapper(Internal_GetProperty(obj, propMeta.Hash));
    }

    inline auto GetPropertyList(TReflObject<ObjectImpl>& obj)
    {
        return Internal_GetPropertyList(obj) | std::views::transform([&](const auto& pair) {
            return FPropertyKVPair{ pair.second.Name, FPropertyWrapper(pair.second.GetProperty(obj.ObjectValue)) };
        });
    }
    template <typename T> inline TReflObject<ObjectImpl> ReferenceObject(T* target)
    {
        return Internal_Reference(target, typeid(*target));
    }

    template <auto Member> inline void RegisterPropertyHint(const String& hintName, PropertyHintValueType value)
    {
        using Accessor   = TAutoMemberAccessor<Member>;
        u64 baseTypeHash = FReflTypeMetaInfo::Create<typename Accessor::ClassType>().Hash;
        u64 propertyHash = FReflPropertyMetaInfo::Create<Member>().Hash;
        Internal_PropertyAddHint(baseTypeHash, propertyHash, hintName, value);
    }
    inline Vec<Fn<void()>> GetPropertyEditorHandles(TReflObject<ObjectImpl>& obj)
    {
        return Internal_GetPropertyEditorHandles(obj);
    }
    inline u32 GetNumVisibleProperties(TReflObject<ObjectImpl>& obj) { return Internal_GetNumVisibleProperties(obj); }

} // namespace Ifrit::Reflection
