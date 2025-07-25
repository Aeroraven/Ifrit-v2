#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/typing/Traits.h"

namespace Ifrit
{
    namespace Private::TypeMeta
    {
        template <class T> consteval inline static const char* GetFuncNameRaw()
        {
#ifdef _MSC_VER
            return __FUNCSIG__;
#else
    #ifdef __PRETTY_FUNCTION__
            return __PRETTY_FUNCTION__;
    #else
            static_assert(false, "Unsupported compiler");
    #endif
#endif
        }

        template <auto T> consteval inline static const char* GetVarNameRaw()
        {
#ifdef _MSC_VER
            return __FUNCSIG__;
#else
    #ifdef __PRETTY_FUNCTION__
            return __PRETTY_FUNCTION__;
    #else
            static_assert(false, "Unsupported compiler");
    #endif
#endif
        }

        template <class T> consteval inline static u64 GetFuncNameLength()
        {
            constexpr auto raw = GetFuncNameRaw<T>();
            u64            len = 0;
            while (raw[len] != '\0')
                ++len;
            return len;
        }

        template <auto T> consteval inline static u64 GetVarNameLength()
        {
            constexpr auto raw = GetVarNameRaw<T>();
            u64            len = 0;
            while (raw[len] != '\0')
                ++len;
            return len;
        }

        template <class T> consteval inline static Array<char, GetFuncNameLength<T>() + 1> GetFuncNameToArray()
        {
            constexpr auto                          raw = GetFuncNameRaw<T>();
            constexpr u64                           len = GetFuncNameLength<T>();

            Array<char, GetFuncNameLength<T>() + 1> arr = {};
            for (u64 i = 0; i <= len; ++i)
            {
                arr[i] = raw[i];
            }
            return arr;
        }

        template <auto T> consteval inline static std::array<char, GetVarNameLength<T>() + 1> GetVarNameToArray()
        {
            constexpr auto                              raw = GetVarNameRaw<T>();
            constexpr u64                               len = GetVarNameLength<T>();

            std::array<char, GetVarNameLength<T>() + 1> arr = {};
            for (u64 i = 0; i <= len; ++i)
            {
                arr[i] = raw[i];
            }
            return arr;
        }

        template <unsigned E, unsigned N> consteval u64 GetFuncNameHash(Array<char, N> const& str)
        {
            if constexpr (N == E)
                return 0;
            else
            {
                return (str[E] + 1) + 257 * GetFuncNameHash<E + 1, N>(str);
            }
        }

        template <unsigned E, char C, unsigned O, unsigned N>
        consteval u64 GetOccurrencePosition(Array<char, N> const& str)
        {
            if constexpr (N == E)
                return 0;
            else
            {
                if (str[E] == C && O == 0)
                    return E;
                else if (str[E] == C)
                    return GetOccurrencePosition<E + 1, C, O - 1, N>(str);
                else
                    return GetOccurrencePosition<E + 1, C, O, N>(str);
            }
        }

        template <unsigned E, char C, unsigned N> consteval u64 GetLastOccurrencePosition(Array<char, N> const& str)
        {
            if constexpr (N == E)
                return 0;
            else
            {
                if (str[E] == C)
                    return std::max(static_cast<u64>(E), GetLastOccurrencePosition<E + 1, C, N>(str));
                else
                    return GetLastOccurrencePosition<E + 1, C, N>(str);
            }
        }

        template <unsigned S, unsigned E, unsigned N>
        consteval inline static Array<char, E - S> GetSubArray(Array<char, N> const& str)
        {
            Array<char, E - S> subArray{};
            for (unsigned i = S; i < E; ++i)
            {
                subArray[i - S] = str[i];
            }
            return subArray;
        }
        template <unsigned S, unsigned N> consteval inline static bool HasStructPrefix(Array<char, N> const& str)
        {
            constexpr Array<char, 7> prefix = { 's', 't', 'r', 'u', 'c', 't', ' ' };
            for (unsigned i = 0; i < 7; ++i)
            {
                if (str[i + S] != prefix[i])
                    return false;
            }
            return true;
        }

        template <unsigned S, unsigned N> consteval inline static bool HasClassPrefix(Array<char, N> const& str)
        {
            constexpr Array<char, 6> prefix = { 'c', 'l', 'a', 's', 's', ' ' };
            for (unsigned i = 0; i < 6; ++i)
            {
                if (str[i + S] != prefix[i])
                    return false;
            }
            return true;
        }

        template <typename T> consteval inline static auto GetActualClassNameArray()
        {
            constexpr auto funcionSignature = GetFuncNameToArray<T>();
            constexpr u64  firstPos         = GetOccurrencePosition<0, '<', 0>(funcionSignature) + 1;
            constexpr u64  lastPos          = GetLastOccurrencePosition<0, '>'>(funcionSignature);
            constexpr u64  length           = lastPos - firstPos;
            constexpr bool hasStructPrefix  = HasStructPrefix<firstPos>(funcionSignature);
            constexpr bool hasClassPrefix   = HasClassPrefix<firstPos>(funcionSignature);
            constexpr int  finalOffset      = firstPos + 0 + 0;
            constexpr auto subArray         = GetSubArray<finalOffset, lastPos>(funcionSignature);
            return subArray;
        }

        template <auto T> consteval inline static auto GetActualVarNameArray()
        {
            constexpr auto funcionSignature = GetVarNameToArray<T>();
            constexpr u64  firstPos         = GetOccurrencePosition<0, '<', 0>(funcionSignature) + 1;
            constexpr u64  lastPos          = GetLastOccurrencePosition<0, '>'>(funcionSignature) + 1;
            constexpr bool hasStructPrefix  = HasStructPrefix<firstPos>(funcionSignature);
            constexpr bool hasClassPrefix   = HasClassPrefix<firstPos>(funcionSignature);
            constexpr int  finalOffset      = firstPos + 0 + 0;
            constexpr auto subArray         = GetSubArray<finalOffset, lastPos>(funcionSignature);
            return subArray;
        }

        template <typename T> consteval inline static u64 GetFuncNameHashId()
        {
            constexpr auto arr = GetActualClassNameArray<T>();
            return GetFuncNameHash<0>(arr);
        }

        template <auto T> consteval inline static u64 GetVarNameHashId()
        {
            constexpr auto arr = GetActualVarNameArray<T>();
            return GetFuncNameHash<0>(arr);
        }
    } // namespace Private::TypeMeta

    template <class T> struct TMetaTypeInfo
    {
        static constexpr u64         Hash                 = Private::TypeMeta::GetFuncNameHashId<T>();
        static constexpr auto        NameArray            = Private::TypeMeta::GetActualClassNameArray<T>();
        static constexpr StringView  Name                 = StringView(NameArray.data(), NameArray.size() - 1);
        static constexpr bool        DefaultConstructible = std::is_default_constructible_v<T>;

        static std::type_info const& GetTypeInfo()
        {
            static std::type_info const* typeInfo = &typeid(T);
            return *typeInfo;
        }
    };

    template <auto T>
        requires IConceptIsMemberPointer<decltype(T)>
    struct TMetaPropertyInfo
    {
        static constexpr u64        Hash      = Private::TypeMeta::GetVarNameHashId<T>();
        static constexpr auto       NameArray = Private::TypeMeta::GetActualVarNameArray<T>();
        static constexpr StringView Name      = StringView(NameArray.data(), NameArray.size() - 1);

        using MemberType = typename TMemberType<decltype(T)>::Type;
        using ClassType  = typename TMemberType<decltype(T)>::ClassType;
    };

    struct FMetaPropertyInfo;
    struct FMetaTypeInfo
    {
        const u64    Hash;
        const String Name;
        std::type_info const& (*GetTypeInfo)();

    private:
        FMetaTypeInfo(FMetaTypeInfo&&) = default;
        FMetaTypeInfo(u64 hash, String name, std::type_info const& (*getTypeInfo)())
            : Hash(hash), Name(std::move(name)), GetTypeInfo(getTypeInfo)
        {
        }

    public:
        template <class T> static FMetaTypeInfo Create()
        {
            return FMetaTypeInfo(
                TMetaTypeInfo<T>::Hash, String{ TMetaTypeInfo<T>::Name }, &TMetaTypeInfo<T>::GetTypeInfo);
        }
        friend struct FMetaPropertyInfo;
    };

    struct FMetaPropertyInfo
    {
        FMetaTypeInfo ClassTypeInfo;
        FMetaTypeInfo MemberTypeInfo;
        const u64     Hash;
        const String  Name;

    private:
        FMetaPropertyInfo(FMetaTypeInfo classTypeInfo, FMetaTypeInfo memberTypeInfo, u64 hash, String name)
            : ClassTypeInfo(std::move(classTypeInfo))
            , MemberTypeInfo(std::move(memberTypeInfo))
            , Hash(hash)
            , Name(std::move(name))
        {
        }

    public:
        template <auto Member>
            requires IConceptIsMemberPointer<decltype(Member)>
        static FMetaPropertyInfo Create()
        {
            using PropertyType = typename TMetaPropertyInfo<Member>::MemberType;
            using ClassType    = typename TMetaPropertyInfo<Member>::ClassType;
            return FMetaPropertyInfo(FMetaTypeInfo::Create<ClassType>(), FMetaTypeInfo::Create<PropertyType>(),
                TMetaPropertyInfo<Member>::Hash, String{ TMetaPropertyInfo<Member>::Name });
        }
    };

} // namespace Ifrit