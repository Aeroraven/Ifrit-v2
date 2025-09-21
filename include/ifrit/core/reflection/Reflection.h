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
#include "ifrit/core/base/containers/Maps.h"

namespace Ifrit::Reflection
{
    using ObjectImpl                      = Object;
    using PropertyHintValueType           = std::variant<i32, f64, String>;
    template <typename T> using Reference = std::reference_wrapper<T>;

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
        String                                  Name;
        Fn<ObjectImpl(ObjectImpl&)>             Accessor;
        Fn<void(ObjectImpl&)>                   UIHandle;
        THashMap<String, PropertyHintValueType> Hints;
        PropertyMetadata                        Metadata;

        FPropertyWrapper GetProperty(ObjectImpl& obj) const { return FPropertyWrapper{ Accessor(obj) }; }
    };

    struct FMethodField
    {
        String                                              Name;
        Fn<ObjectImpl(ObjectImpl&, const Vec<ObjectImpl>&)> Invoker;
        THashMap<String, PropertyHintValueType>             Hints;
        Fn<void(ObjectImpl&)>                               UIHandle;

        ObjectImpl Invoke(ObjectImpl& obj, const Vec<ObjectImpl>& args) const { return Invoker(obj, args); }
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

    struct FReflTypeMetaInfo
    {
        StringView                    Name;
        u64                           Hash;
        Fn<ObjectImpl()>              Constructor = nullptr;
        THashMap<u64, FPropertyField> PropertyFields;
        THashMap<u64, FMethodField>   MethodFields;
        FMetaTypeExtendedInfo         MetaInfo;
        bool                          Polymorphic = false;
        Vec<u64>                      BaseTypes;
        Vec<u64>                      DerivedTypes;

        FReflTypeMetaInfo() = default;
        bool operator==(const FReflTypeMetaInfo& other) const { return Hash == other.Hash; }
        bool operator!=(const FReflTypeMetaInfo& other) const { return !(*this == other); }

        template <typename T> static FReflTypeMetaInfo Create()
        {
            FReflTypeMetaInfo metaInfo;
            metaInfo.Name = TMetaTypeInfo<T>::Name;
            metaInfo.Hash = TMetaTypeInfo<T>::Hash;
            if constexpr (std::is_default_constructible_v<T>)
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

            FReflPropertyMetaInfo meta;
            meta.ClassTypeInfo  = FReflTypeMetaInfo::Create<typename Accessor::ClassType>();
            meta.MemberTypeInfo = FReflTypeMetaInfo::Create<typename Accessor::Type>();
            meta.Name           = TMetaPropertyInfo<Member>::Name;
            meta.Hash           = TMetaPropertyInfo<Member>::Hash;
            return meta;
        }
    };

    struct FReflMethodMetaInfo
    {
        FReflTypeMetaInfo ClassTypeInfo;
        FReflTypeMetaInfo ReturnTypeInfo;
        StringView        Name;
        u64               Hash;

        FReflMethodMetaInfo() = default;

        template <auto Member>
            requires IConceptIsMemberPointer<decltype(Member)>
        static FReflMethodMetaInfo Create()
        {
            using Accessor = TMemberFunctionTrait<decltype(Member)>;

            FReflMethodMetaInfo meta;
            meta.ClassTypeInfo  = FReflTypeMetaInfo::Create<typename Accessor::ClassType>();
            meta.ReturnTypeInfo = FReflTypeMetaInfo::Create<typename Accessor::ReturnType>();
            meta.Name           = TMetaMemberFunctionInfo<Member>::Name;
            meta.Hash           = TMetaMemberFunctionInfo<Member>::Hash;
            return meta;
        }
    };

    // Internal API
    IFRIT_CORE_API void Internal_RegisterType(const FReflTypeMetaInfo& typeInfo, std::type_info const& typeInfoStd);
    IFRIT_CORE_API TReflObject<ObjectImpl> Internal_Construct(u64 typeHash);
    IFRIT_CORE_API void                    Internal_RegisterPropertyField(const FReflTypeMetaInfo& typeInfo,
                           const FReflPropertyMetaInfo& propInfo, const String& propertyName, Fn<ObjectImpl(ObjectImpl&)> accessor,
                           Fn<void(ObjectImpl&)> uihandle);
    IFRIT_CORE_API void                    Internal_RegisterMethodField(const FReflTypeMetaInfo& typeInfo,
                           const FReflMethodMetaInfo& methodInfo, const String& methodName,
                           Fn<ObjectImpl(ObjectImpl&, const Vec<ObjectImpl>&)> invoker, Fn<void(ObjectImpl&)> uihandle);

    IFRIT_CORE_API ObjectImpl              Internal_GetProperty(TReflObject<ObjectImpl>& obj, u64 propertyHash);
    IFRIT_CORE_API THashMap<u64, FPropertyField> Internal_GetPropertyList(TReflObject<ObjectImpl>& obj);
    IFRIT_CORE_API TReflObject<ObjectImpl> Internal_Reference(void* target, std::type_info const& typeInfo);
    IFRIT_CORE_API u64                     Internal_GetTypeHashFromTypeInfoHash(u64 typeInfoHash);
    IFRIT_CORE_API void                    Internal_RegisterPolymorphic(u64 baseTypeHash, u64 derivedTypeHash);
    IFRIT_CORE_API bool Internal_TypeOnInheritanceChain(u64 baseTypeHashToSearch, u64 derivedTypeHash);
    IFRIT_CORE_API void Internal_PropertyAddHint(
        u64 baseTypeHash, u64 propertyHash, const String& hintName, PropertyHintValueType value);
    IFRIT_CORE_API const PropertyMetadata& Internal_GetPropertyMetadata(u64 baseTypeHash, u64 propertyHash);
    IFRIT_CORE_API Vec<Fn<ObjectImpl(const Vec<ObjectImpl>&)>> Internal_GetRegisteredFuncs(
        TReflObject<ObjectImpl>& obj);
    IFRIT_CORE_API Vec<Fn<void()>> Internal_GetPropertyEditorHandles(TReflObject<ObjectImpl>& obj);
    IFRIT_CORE_API Vec<Fn<void()>> Internal_GetMethodEditorHandles(TReflObject<ObjectImpl>& obj);

    IFRIT_CORE_API u32             Internal_GetNumVisibleProperties(TReflObject<ObjectImpl>& obj);
    IFRIT_CORE_API u32             Internal_GetNumRegisteredFuncs(TReflObject<ObjectImpl>& obj);
    IFRIT_CORE_API ObjectImpl      Internal_InvokeMethod(
             TReflObject<ObjectImpl>& obj, u64 methodHash, const Vec<ObjectImpl>& args);
    IFRIT_CORE_API const char* Internal_GetFunctionAlias(u64 typeHash, u64 methodHash);
    IFRIT_CORE_API Vec<Reference<const FReflTypeMetaInfo>> Internal_GetAllDerivedTypes(
        u64 baseTypeHash, bool includeBase);

    IFRIT_CORE_API void Internal_IgnoreNonVirtualInhertance();
    IFRIT_CORE_API void Internal_ReportWrongFunctionCall();

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
                else if constexpr (IConceptHasCustomEditingHandle<MemberType>)
                {
                    MemberType&             member   = prop.As<MemberType>();
                    auto                    typeInfo = FReflTypeMetaInfo::Create<ClassType>();
                    auto                    propInfo = FReflPropertyMetaInfo::Create<Member>();
                    const PropertyMetadata& metadata = Internal_GetPropertyMetadata(typeInfo.Hash, propInfo.Hash);
                    member.GetUIEditingHandle(metadata);
                }
            };
        }
    };

    template <typename T, typename R, typename... Args, usize... I>
    R MemberFunctorInvokerWrapperImpl(
        R (T::*f)(Args...), T& obj, const Vec<ObjectImpl>& vec, TSinkArg<std::index_sequence<I...>>)
    {
        return (obj.*f)((vec[I]).As<Args>()...);
    }

    template <typename T, typename R, typename... Args>
    R MemberFunctorInvokerWrapper(R (T::*f)(Args...), T& obj, const Vec<ObjectImpl>& vec)
    {
        if (vec.size() != sizeof...(Args)) IF_UNLIKELY
            Internal_ReportWrongFunctionCall();

        return MemberFunctorInvokerWrapperImpl(f, obj, vec, std::index_sequence_for<Args...>{});
    }

    template <auto Member>
        requires IConceptIsMemberPointer<decltype(Member)>
    class TAutoMemberFunctionAccessor
    {
    public:
        using ClassType  = TMemberFunctionTrait<decltype(Member)>::ClassType;
        using ReturnType = TMemberFunctionTrait<decltype(Member)>::ReturnType;
        using ArgTypes   = TMemberFunctionTrait<decltype(Member)>::ArgsTuple;

        static Fn<ObjectImpl(ObjectImpl&, const Vec<ObjectImpl>&)> GetMemberInvoker()
        {
            return [](ObjectImpl& classObj, const Vec<ObjectImpl>& args) -> ObjectImpl {
                auto& obj = classObj.As<ClassType>();
                if constexpr (std::is_void_v<ReturnType>)
                {
                    MemberFunctorInvokerWrapper(Member, obj, args);
                    return ObjectImpl::CreateVoid();
                }
                else
                {
                    return ObjectImpl::CreateClone(MemberFunctorInvokerWrapper(Member, obj, args));
                }
            };
        }
        static Fn<void(ObjectImpl&)> GetUIHandle()
        {

            if constexpr (std::tuple_size<ArgTypes>::value == 0)
            {
                return [](ObjectImpl& obj) {
                    auto funcHandle    = GetFunctionUIHandle();
                    auto invoker       = GetMemberInvoker();
                    auto propInfo      = FReflMethodMetaInfo::Create<Member>();
                    auto typeHash      = FReflTypeMetaInfo::Create<ClassType>().Hash;
                    auto invokeWrapped = [invoker, &obj]() -> void {
                        Vec<ObjectImpl> args;
                        invoker(obj, args);
                    };
                    const char* dispName = Internal_GetFunctionAlias(typeHash, propInfo.Hash);
                    if (funcHandle.mFunctionCallback)
                    {
                        funcHandle.mFunctionCallback(dispName, invokeWrapped);
                    }
                };
            }
            return [](ObjectImpl& obj) {};
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
    template <auto Member>
        requires IConceptIsMemberPointer<decltype(Member)>
    inline void RegisterMethodField(const String& methodName)
    {
        using Accessor = TAutoMemberFunctionAccessor<Member>;

        auto typeInfo   = FReflTypeMetaInfo::Create<typename Accessor::ClassType>();
        auto methodInfo = FReflMethodMetaInfo::Create<Member>();
        Internal_RegisterMethodField(
            typeInfo, methodInfo, methodName, Accessor::GetMemberInvoker(), Accessor::GetUIHandle());
    }

    inline TReflObject<ObjectImpl> ConstructObject(const FMetaTypeInfo& typeMeta)
    {
        return Internal_Construct(typeMeta.Hash);
    }
    inline TReflObject<ObjectImpl> ConstructObjectFromHash(u64 typeHash) { return Internal_Construct(typeHash); }
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

    inline ObjectImpl InvokeMethod(TReflObject<ObjectImpl>& obj, FMetaMethodInfo funcMeta, const Vec<ObjectImpl>& args)
    {
        u64 methodHash = funcMeta.Hash;
        return Internal_InvokeMethod(obj, methodHash, args);
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
    inline Vec<Fn<void()>> GetMethodEditorHandles(TReflObject<ObjectImpl>& obj)
    {
        return Internal_GetMethodEditorHandles(obj);
    }
    inline u32 GetNumVisibleProperties(TReflObject<ObjectImpl>& obj) { return Internal_GetNumVisibleProperties(obj); }

    template <typename T> inline Vec<Reference<const FReflTypeMetaInfo>> GetAllDerivedTypes(bool includeBase)
    {
        u64 baseTypeHash = FReflTypeMetaInfo::Create<T>().Hash;
        return Internal_GetAllDerivedTypes(baseTypeHash, includeBase);
    }

} // namespace Ifrit::Reflection
