#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/typing/TypeMetaInfo.h"
#include "ifrit/core/reflection/Serializer.h"

namespace Ifrit::Reflection
{
    class Archive;
    struct FMetaTypeExtendedInfo
    {
        u64    Hash;
        String Name;
        std::type_info const& (*GetTypeInfo)();
        FMetaTypeInfo (*GetMetaInfo)();
        void (*Destructor)(void*);
        void (*SerializeInterface)(Archive*, void*);
        void (*DeserializeInterface)(Archive*, void*);
        void (*OutputStreamFn)(std::ostream&, const void*);
        void (*InputStreamFn)(std::istream&, void*);

        FMetaTypeExtendedInfo() = default;

    private:
        FMetaTypeExtendedInfo(u64 hash, String name, std::type_info const& (*getTypeInfo)(),
            FMetaTypeInfo (*getMetaInfo)(), void (*destructor)(void*), void (*serializeInterface)(Archive*, void*),
            void (*deserializeInterface)(Archive*, void*), void (*outputStreamFn)(std::ostream&, const void*),
            void (*inputStreamFn)(std::istream&, void*))
            : Hash(hash)
            , Name(std::move(name))
            , GetTypeInfo(getTypeInfo)
            , GetMetaInfo(getMetaInfo)
            , Destructor(destructor)
            , SerializeInterface(serializeInterface)
            , DeserializeInterface(deserializeInterface)
            , OutputStreamFn(outputStreamFn)
            , InputStreamFn(inputStreamFn)
        {
        }

    public:
        template <class T> static FMetaTypeExtendedInfo Create()
        {
            u64    Hash                            = TMetaTypeInfo<T>::Hash;
            String Name                            = String{ TMetaTypeInfo<T>::Name };
            std::type_info const& (*GetTypeInfo)() = &TMetaTypeInfo<T>::GetTypeInfo;
            FMetaTypeInfo (*GetMetaInfo)()         = &FMetaTypeInfo::Create<T>;
            void (*Destructor)(void*)              = [](void* ptr) { delete static_cast<T*>(ptr); };
            void (*SerializeInterface)(
                Archive*, void*) = [](Archive* archive, void* obj) { InvokeSerialize(*static_cast<T*>(obj), archive); };
            void (*DeserializeInterface)(Archive*, void*) = [](Archive* archive, void* obj) {
                InvokeDeserialize(*static_cast<T*>(obj), archive);
            };
            void (*OutputStreamFn)(std::ostream&, const void*) = nullptr;
            void (*InputStreamFn)(std::istream&, void*)        = nullptr;
            if constexpr (IConceptIsOutputStreamable<T>)
            {
                OutputStreamFn = [](std::ostream& os, const void* obj) { os << *static_cast<const T*>(obj); };
            }
            if constexpr (IConceptIsInputStreamable<T>)
            {
                InputStreamFn = [](std::istream& is, void* obj) { is >> *static_cast<T*>(obj); };
            }

            FMetaTypeExtendedInfo typeInfo(Hash, std::move(Name), GetTypeInfo, GetMetaInfo, Destructor,
                SerializeInterface, DeserializeInterface, OutputStreamFn, InputStreamFn);
            return typeInfo;
        }
        template <> static FMetaTypeExtendedInfo Create<void>()
        {
            return FMetaTypeExtendedInfo(TMetaTypeInfo<void>::Hash, String{ TMetaTypeInfo<void>::Name },
                &TMetaTypeInfo<void>::GetTypeInfo, &FMetaTypeInfo::Create<void>, nullptr, nullptr, nullptr, nullptr,
                nullptr);
        }
    };
} // namespace Ifrit::Reflection