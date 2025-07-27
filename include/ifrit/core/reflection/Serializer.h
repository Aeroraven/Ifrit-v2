#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/core/typing/TypeMetaInfo.h"
#include "ifrit/core/reflection/Archive.h"
#include "ifrit/core/reflection/RttiIdentifier.h"
namespace Ifrit::Reflection
{
    IFRIT_CORE_API void InvokeSerializeDynamicImpl(void* ptr, const std::type_info& typeInfo, Archive* archive);
    IFRIT_CORE_API void InvokeDeserializeDynamicImpl(void* ptr, const std::type_info& typeInfo, Archive* archive);
    IFRIT_CORE_API void InvokePolymorphicConstructImpl(void*& ptr, u64 typeInfoHash);

    namespace Internal
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
        template <typename K, typename V, typename C, typename A>
        struct TTraitIsMap<std::map<K, V, C, A>> : std::true_type
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

        template <typename T> struct TTraitIsUniquePtr : std::false_type
        {
            using ElementType = void;
        };

        template <typename T> struct TTraitIsUniquePtr<std::unique_ptr<T>> : std::true_type
        {
            using ElementType = T;
        };

        template <typename T> struct TTraitIsStlArray : std::false_type
        {
            using ElementType = void;
        };

        template <typename T, usize N> struct TTraitIsStlArray<std::array<T, N>> : std::true_type
        {
            using ElementType = T;
        };

        template <typename T>
        concept IConceptIsTriviallySerializable = requires(
            T t) { requires(std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, String>); };

        template <typename T, typename = void> struct TTraitIsSerializable : std::false_type
        {
        };

        template <IConceptIsTriviallySerializable T>
        struct TTraitIsSerializable<T, std::void_t<decltype(std::declval<T&>())>> : std::true_type
        {
        };

        template <typename T>
        struct TTraitIsSerializable<T,
            std::enable_if_t<TTraitIsVector<T>::value
                && TTraitIsSerializable<typename TTraitIsVector<T>::ElementType>::value>> : std::true_type
        {
        };

        template <typename T>
        struct TTraitIsSerializable<T,
            std::enable_if_t<TTraitIsMap<T>::value && TTraitIsSerializable<typename TTraitIsMap<T>::KeyType>::value
                && TTraitIsSerializable<typename TTraitIsMap<T>::ValueType>::value>> : std::true_type
        {
        };

        template <typename T>
        struct TTraitIsSerializable<T,
            std::enable_if_t<TTraitIsUniquePtr<T>::value
                && TTraitIsSerializable<typename TTraitIsUniquePtr<T>::ElementType>::value>> : std::true_type
        {
        };

        template <typename T>
        concept IConceptSerializable = TTraitIsSerializable<T>::value;

        template <typename T>
        concept IConceptVectorSerializable =
            TTraitIsVector<T>::value && IConceptSerializable<typename TTraitIsVector<T>::ElementType>;

        template <typename T>
        concept IConceptMapSerializable =
            TTraitIsMap<T>::value && IConceptSerializable<typename TTraitIsMap<T>::KeyType>
            && IConceptSerializable<typename TTraitIsMap<T>::ValueType>;

        template <typename T>
        concept IConceptIsVector = TTraitIsVector<T>::value;

        template <typename T>
        concept IConceptIsMap = TTraitIsMap<T>::value;

        template <typename T>
        concept IConceptIsUniquePtr = TTraitIsUniquePtr<T>::value;

        template <typename T>
        concept IConceptIsStlArray = TTraitIsStlArray<T>::value;

        template <typename T>
        concept IConceptIsPolymorphic = std::is_polymorphic_v<T>;

        // Tag types for dispatch
        enum class ESpecializationTag : u8
        {
            Vector,
            Map,
            UniquePtr,
            Trivial,
            Dynamic
        };

        template <ESpecializationTag Tag> struct TSpecializationTag
        {
            static constexpr ESpecializationTag Value = Tag;
        };

        // Tag selection trait
        template <typename T> consteval auto SelectSerializationTag()
        {
            if constexpr (IConceptIsVector<T>)
                return TSpecializationTag<ESpecializationTag::Vector>{};
            else if constexpr (IConceptIsMap<T>)
                return TSpecializationTag<ESpecializationTag::Map>{};
            else if constexpr (IConceptIsUniquePtr<T>)
                return TSpecializationTag<ESpecializationTag::UniquePtr>{};
            else if constexpr (IConceptIsTriviallySerializable<T>)
                return TSpecializationTag<ESpecializationTag::Trivial>{};
            else
                return TSpecializationTag<ESpecializationTag::Dynamic>{};
        }

        // Serializer common keys
        struct FSerializerReservedKeys
        {
            constexpr static const char* kVectorContainer   = "__ifrit_vector";
            constexpr static const char* kVectorItem        = "__ifrit_vector_item";
            constexpr static const char* kMapContainer      = "__ifrit_map";
            constexpr static const char* kMapItem           = "__ifrit_map_item";
            constexpr static const char* kMapKey            = "__ifrit_map_key";
            constexpr static const char* kMapValue          = "__ifrit_map_value";
            constexpr static const char* kUniquePtrValue    = "__ifrit_unique_ptr_value";
            constexpr static const char* kUniquePtrValid    = "__ifrit_unique_ptr_valid";
            constexpr static const char* kUniquePtrTypePoly = "__ifrit_unique_ptr_polymorphic";
            constexpr static const char* kUniquePtrTypeHash = "__ifrit_unique_ptr_type_hash";
        };

        // Serializer
        template <typename T>
        void SerializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::Vector>)
        {
            archive->BeginArray(FSerializerReservedKeys::kVectorContainer);
            for (auto& item : obj)
            {
                archive->BeginObject(FSerializerReservedKeys::kVectorItem);
                InvokeSerialize(item, archive);
                archive->EndObject();
            }
            archive->EndArray();
        }

        template <typename T> void SerializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::Map>)
        {
            archive->BeginArray(FSerializerReservedKeys::kMapContainer);
            for (auto& [k, v] : obj)
            {
                archive->BeginObject(FSerializerReservedKeys::kMapItem);
                archive->BeginObject(FSerializerReservedKeys::kMapKey);
                InvokeSerialize(k, archive);
                archive->EndObject();
                archive->BeginObject(FSerializerReservedKeys::kMapValue);
                InvokeSerialize(v, archive);
                archive->EndObject();
                archive->EndObject();
            }
            archive->EndArray();
        }

        template <typename T>
        void SerializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::UniquePtr>)
        {
            if (obj.get())
            {
                archive->BeginObject(FSerializerReservedKeys::kUniquePtrValid);
                archive->Serialize(1);
                archive->EndObject();

                archive->BeginObject(FSerializerReservedKeys::kUniquePtrTypePoly);
                using U = TTraitIsUniquePtr<T>::ElementType;
                if constexpr (IConceptIsPolymorphic<U>)
                    archive->Serialize(1);
                else
                    archive->Serialize(0);
                archive->EndObject();

                archive->BeginObject(FSerializerReservedKeys::kUniquePtrTypeHash);
                archive->Serialize(GetTypeIDHash(typeid(*obj)));
                archive->EndObject();

                archive->BeginObject(FSerializerReservedKeys::kUniquePtrValue);
                InvokeSerialize(*obj, archive);
                archive->EndObject();
            }
            else
            {
                archive->BeginObject(FSerializerReservedKeys::kUniquePtrValid);
                archive->Serialize(0);
                archive->EndObject();
            }
        }

        template <typename T>
        void SerializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::Trivial>)
        {
            archive->Serialize(obj);
        }

        template <typename T>
        void SerializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::Dynamic>)
        {
            InvokeSerializeDynamicImpl(&obj, typeid(*(&obj)), archive);
        }

        // Deserializer
        template <typename T>
        void DeserializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::Vector>)
        {
            using ElementType = typename TTraitIsVector<T>::ElementType;

            if (archive->HasArray(FSerializerReservedKeys::kVectorContainer))
            {
                archive->BeginArray(FSerializerReservedKeys::kVectorContainer);
                size_t size = archive->GetArraySize();
                obj.clear();
                obj.reserve(size);

                int cnt = 0;
                while (archive->HasNextArrayElement())
                {
                    ElementType element;
                    archive->BeginObject(FSerializerReservedKeys::kVectorItem);
                    InvokeDeserialize(element, archive);
                    archive->EndObject();
                    obj.push_back(std::move(element));
                    archive->NextArrayElement();
                }
                archive->EndArray();
            }
        }

        template <typename T>
        void DeserializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::Map>)
        {
            using KeyType   = typename TTraitIsMap<T>::KeyType;
            using ValueType = typename TTraitIsMap<T>::ValueType;
            if (archive->HasArray(FSerializerReservedKeys::kMapContainer))
            {
                archive->BeginArray(FSerializerReservedKeys::kMapContainer);
                obj.clear();

                size_t size = archive->GetArraySize();

                while (archive->HasNextArrayElement())
                {
                    archive->BeginObject(FSerializerReservedKeys::kMapItem);

                    KeyType key;
                    archive->BeginObject(FSerializerReservedKeys::kMapKey);
                    InvokeDeserialize(key, archive);
                    archive->EndObject();

                    ValueType value;
                    archive->BeginObject(FSerializerReservedKeys::kMapValue);
                    InvokeDeserialize(value, archive);
                    archive->EndObject();

                    obj[std::move(key)] = std::move(value);
                    archive->EndObject();
                    archive->NextArrayElement();
                }
                archive->EndArray();
            }
        }

        template <typename T>
        void DeserializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::UniquePtr>)
        {
            using ElementType    = typename TTraitIsUniquePtr<T>::ElementType;
            int valid            = 0;
            int poly             = 0;
            u64 cachedTypeIdHash = 0;
            if (archive->HasObject(FSerializerReservedKeys::kUniquePtrValid))
            {
                archive->BeginObject(FSerializerReservedKeys::kUniquePtrValid);
                archive->Serialize(valid);
                archive->EndObject();
            }
            if (valid)
            {
                if (archive->HasObject(FSerializerReservedKeys::kUniquePtrTypePoly))
                {
                    archive->BeginObject(FSerializerReservedKeys::kUniquePtrTypePoly);
                    archive->Serialize(poly);
                    archive->EndObject();
                }
                if (archive->HasObject(FSerializerReservedKeys::kUniquePtrTypeHash))
                {
                    archive->BeginObject(FSerializerReservedKeys::kUniquePtrTypeHash);
                    archive->Serialize(cachedTypeIdHash);
                    archive->EndObject();
                }

                if (archive->HasObject(FSerializerReservedKeys::kUniquePtrValue))
                {
                    archive->BeginObject(FSerializerReservedKeys::kUniquePtrValue);
                    if (!poly)
                    {
                        obj = std::make_unique<ElementType>();
                    }
                    else
                    {
                        void* ptr;
                        InvokePolymorphicConstructImpl(ptr, cachedTypeIdHash);
                        obj = std::unique_ptr<ElementType>(reinterpret_cast<ElementType*>(ptr));
                    }

                    InvokeDeserialize(*obj, archive);
                    archive->EndObject();
                }
                else
                {
                    obj = nullptr;
                }
            }
            else
            {
                obj = nullptr;
            }
        }

        template <typename T>
        void DeserializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::Trivial>)
        {
            archive->Serialize(obj);
        }

        template <typename T>
        void DeserializeImpl(T& obj, Archive* archive, TSpecializationTag<ESpecializationTag::Dynamic>)
        {
            InvokeDeserializeDynamicImpl(&obj, typeid(*(&obj)), archive);
        }

    } // namespace Internal

    template <typename T> inline void InvokeSerialize(T& obj, Archive* archive)
    {
        Internal::SerializeImpl(obj, archive, Internal::SelectSerializationTag<T>());
    }

    template <typename T> inline void InvokeDeserialize(T& obj, Archive* archive)
    {
        Internal::DeserializeImpl(obj, archive, Internal::SelectSerializationTag<T>());
    }

} // namespace Ifrit::Reflection