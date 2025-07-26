#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/core/typing/TypeMetaInfo.h"
#include "ifrit/core/reflection/Archive.h"

namespace Ifrit::Reflection
{
    // Type Traits
    template <typename T> struct TTraitIsVector : std::false_type
    {
        using ElementType = void;
    };

    template <typename T, typename Alloc> struct TTraitIsVector<std::vector<T, Alloc>> : std::true_type
    {
        using ElementType = T;
    };

    template <typename T> struct TTraitIsMap : std::false_type
    {
        using KeyType   = void;
        using ValueType = void;
    };
    template <typename K, typename V, typename C, typename A> struct TTraitIsMap<std::map<K, V, C, A>> : std::true_type
    {
        using KeyType   = K;
        using ValueType = V;
    };
    template <typename K, typename V, typename H, typename E, typename A>
    struct TTraitIsMap<std::unordered_map<K, V, H, E, A>> : std::true_type
    {
        using KeyType   = K;
        using ValueType = V;
    };

    template <typename T>
    concept IConceptIsTriviallySerializable =
        requires(T t) { requires(std::is_integral_v<T> || std::is_floating_point_v<T>); };

    template <typename T>
    concept IConceptIsStaticSerializable = requires(T t) {
        {
            t.Serialize(std::declval<Archive*>())
        } -> std::same_as<void>;
        {
            t.Deserialize(std::declval<Archive*>())
        } -> std::same_as<void>;
    };

    template <typename T>
    concept IConceptIsVectorSerializable = requires(T t) {
        requires TTraitIsVector<T>::value;
        typename TTraitIsVector<T>::ElementType;
        requires IConceptIsTriviallySerializable<typename TTraitIsVector<T>::ElementType>;
    };

    template <typename T>
    concept IConceptIsMapSerializable = requires(T t) {
        requires TTraitIsMap<T>::value;
        typename TTraitIsMap<T>::KeyType;
        typename TTraitIsMap<T>::ValueType;
        requires IConceptIsTriviallySerializable<typename TTraitIsMap<T>::KeyType>;
        requires IConceptIsTriviallySerializable<typename TTraitIsMap<T>::ValueType>;
    };

    template <typename T>
    concept IConceptSerializable = IConceptIsTriviallySerializable<T> || IConceptIsStaticSerializable<T>
        || IConceptIsVectorSerializable<T> || IConceptIsMapSerializable<T>;

    template <typename T>
        requires IConceptSerializable<T>
    void InvokeSerialize(T& obj, Archive* archive)
    {
        archive->BeginObject(typeid(T).name());

        if constexpr (IConceptIsStaticSerializable<T>)
        {
            obj.Serialize(archive);
        }
        else if constexpr (IConceptIsVectorSerializable<T>)
        {
            archive->BeginArray("Items");
            for (auto& item : obj)
            {
                InvokeSerialize(item, archive);
            }
            archive->EndArray();
        }
        else if constexpr (IConceptIsMapSerializable<T>)
        {
            archive->BeginArray("MapItems");
            for (auto& [key, value] : obj)
            {
                InvokeSerialize(key, archive);
                InvokeSerialize(value, archive);
            }
            archive->EndArray();
        }
        else if constexpr (IConceptIsTriviallySerializable<T>)
        {
            archive->Serialize(obj);
        }
        else if constexpr (IConceptCustomSerializable<T>)
        {
            archive->Serialize(obj.Serialize());
        }
        else
        {
            std::abort();
        }
        archive->EndObject();
    }

} // namespace Ifrit::Reflection