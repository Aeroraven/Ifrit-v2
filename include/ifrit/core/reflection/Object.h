#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include <memory>
#include <iostream>
#include "ifrit/core/typing/TypeMetaInfo.h"
#include "ifrit/core/reflection/Serializer.h"
#include "ifrit/core/reflection/TypeMetaExtended.h"

namespace Ifrit::Reflection
{
    IFRIT_CORE_API void ObjectBadCastReport(const String& expected, const String& requested);
    IFRIT_CORE_API bool IsValidObjectCast(const std::type_info& fromType, const std::type_info& toType);

    class Archive;
    struct IFRIT_CORE_API Object
    {
        void* Ptr                                          = nullptr;
        std::type_info const& (*TypeInfoGetter)()          = nullptr;
        FMetaTypeInfo (*TypeMetaGetter)()                  = nullptr;
        void (*Destructor)(void*)                          = nullptr;
        void (*SerializeInterface)(Archive*, void*)        = nullptr;
        void (*DeserializeInterface)(Archive*, void*)      = nullptr;
        void (*OutputStreamFn)(std::ostream&, const void*) = nullptr;
        void (*InputStreamFn)(std::istream&, void*)        = nullptr;

        Object(void* ptr, std::type_info const& (*typeInfoGetter)(), FMetaTypeInfo (*typeMetaGetter)(),
            void (*destructor)(void*), void (*serializeInterface)(Archive*, void*),
            void (*deserializeInterface)(Archive*, void*), void (*outputStreamFn)(std::ostream&, const void*) = nullptr,
            void (*inputStreamFn)(std::istream&, void*) = nullptr)
            : Ptr(ptr)
            , TypeInfoGetter(typeInfoGetter)
            , TypeMetaGetter(typeMetaGetter)
            , Destructor(destructor)
            , SerializeInterface(serializeInterface)
            , DeserializeInterface(deserializeInterface)
            , OutputStreamFn(outputStreamFn)
            , InputStreamFn(inputStreamFn)
        {
        }
        Object(Object&& rhs) noexcept
            : Ptr(rhs.Ptr)
            , TypeInfoGetter(rhs.TypeInfoGetter)
            , TypeMetaGetter(rhs.TypeMetaGetter)
            , Destructor(rhs.Destructor)
            , SerializeInterface(rhs.SerializeInterface)
            , DeserializeInterface(rhs.DeserializeInterface)
            , OutputStreamFn(rhs.OutputStreamFn)
            , InputStreamFn(rhs.InputStreamFn)
        {
            rhs.Ptr = nullptr;
        }

        Object(const Object&) = delete;

        ~Object()
        {
            if (Ptr)
            {
                Destructor(Ptr);
            }
        }
        Object& operator=(Object&& rhs)
        {
            if (this != &rhs)
            {
                if (Ptr)
                {
                    Destructor(Ptr);
                }
                Ptr                  = rhs.Ptr;
                TypeInfoGetter       = rhs.TypeInfoGetter;
                TypeMetaGetter       = rhs.TypeMetaGetter;
                Destructor           = rhs.Destructor;
                SerializeInterface   = rhs.SerializeInterface;
                DeserializeInterface = rhs.DeserializeInterface;
                OutputStreamFn       = rhs.OutputStreamFn;
                InputStreamFn        = rhs.InputStreamFn;
                rhs.Ptr              = nullptr;
            }
            return *this;
        }
        Object&              operator=(const Object&) = delete;

        // streaming operators
        friend std::ostream& operator<<(std::ostream& os, const Object& obj)
        {
            if (obj.OutputStreamFn)
            {
                obj.OutputStreamFn(os, obj.Ptr);
            }
            else
            {
                os << "Object at " << obj.Ptr;
            }
            return os;
        }

        friend std::istream& operator>>(std::istream& is, Object& obj)
        {
            if (obj.InputStreamFn)
            {
                obj.InputStreamFn(is, obj.Ptr);
            }
            else
            {
                is.setstate(std::ios::failbit);
            }
            return is;
        }

        template <typename T> T* AsPtr() const
        {
            if (TypeInfoGetter() == typeid(T))
            {
                return static_cast<T*>(Ptr);
            }
            return nullptr;
        }
        template <typename T> T& As() const
        {
            if (IsValidObjectCast(TypeInfoGetter(), typeid(T)))
            {
                return *static_cast<T*>(Ptr);
            }
            ObjectBadCastReport(typeid(T).name(), TypeInfoGetter().name());
            throw std::bad_cast();
        }
        template <typename T> void ForcedReinterpretTransferTo(std::unique_ptr<T>& target) noexcept
        {
            auto casted = reinterpret_cast<T*>(Ptr);
            target      = std::unique_ptr<T>(casted);
            Ptr         = nullptr;
        }

        void ForcedTransferToUnsafe(void*& target) noexcept
        {
            target = Ptr;
            Ptr    = nullptr;
        }

        void Serialize(Archive* archive) const;
        void Deserialize(Archive* archive) const;

    public:
        // Static factory methods
        template <typename T, typename... Args> static Object Create(Args&&... args)
        {
            T*                    ptr    = new T(std::forward<Args>(args)...);
            FMetaTypeExtendedInfo tpInfo = FMetaTypeExtendedInfo::Create<T>();
            return Object(ptr, tpInfo.GetTypeInfo, tpInfo.GetMetaInfo, tpInfo.Destructor, tpInfo.SerializeInterface,
                tpInfo.DeserializeInterface, tpInfo.OutputStreamFn, tpInfo.InputStreamFn);
        }

        static Object CreateProxy(void* ptr, FMetaTypeExtendedInfo& propInfo)
        {
            return Object(
                ptr, propInfo.GetTypeInfo, propInfo.GetMetaInfo, [](void*) {}, propInfo.SerializeInterface,
                propInfo.DeserializeInterface, propInfo.OutputStreamFn, propInfo.InputStreamFn);
        }

        template <typename T> static Object Create(std::reference_wrapper<T> ref)
        {
            FMetaTypeExtendedInfo tpInfo = FMetaTypeExtendedInfo::Create<T>();
            return Object(
                &ref.get(), tpInfo.GetTypeInfo, tpInfo.GetMetaInfo, [](void*) {}, tpInfo.SerializeInterface,
                tpInfo.DeserializeInterface, tpInfo.OutputStreamFn, tpInfo.InputStreamFn);
        }
    };

} // namespace Ifrit::Reflection
