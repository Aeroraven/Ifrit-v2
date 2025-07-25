#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include <memory>
#include <iostream>

namespace Ifrit::Reflection
{
    struct Object
    {
        void* Ptr                                          = nullptr;
        std::type_info const& (*TypeInfoGetter)()          = nullptr;
        void (*Destructor)(void*)                          = nullptr;
        void (*OutputStreamFn)(std::ostream&, const void*) = nullptr;
        void (*InputStreamFn)(std::istream&, void*)        = nullptr;

        Object(void* ptr, std::type_info const& (*typeInfoGetter)(), void (*destructor)(void*),
            void (*outputStreamFn)(std::ostream&, const void*) = nullptr,
            void (*inputStreamFn)(std::istream&, void*)        = nullptr)
            : Ptr(ptr)
            , TypeInfoGetter(typeInfoGetter)
            , Destructor(destructor)
            , OutputStreamFn(outputStreamFn)
            , InputStreamFn(inputStreamFn)
        {
        }
        Object(Object&& rhs) noexcept
            : Ptr(rhs.Ptr)
            , TypeInfoGetter(rhs.TypeInfoGetter)
            , Destructor(rhs.Destructor)
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
                Ptr            = rhs.Ptr;
                TypeInfoGetter = rhs.TypeInfoGetter;
                Destructor     = rhs.Destructor;
                OutputStreamFn = rhs.OutputStreamFn;
                InputStreamFn  = rhs.InputStreamFn;
                rhs.Ptr        = nullptr;
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
            if (TypeInfoGetter() == typeid(T))
            {
                return *static_cast<T*>(Ptr);
            }
            throw std::bad_cast();
        }
        template <typename T> void ForcedReinterpretTransferTo(std::unique_ptr<T>& target) noexcept
        {
            auto casted = reinterpret_cast<T*>(Ptr);
            target      = std::unique_ptr<T>(casted);
            Ptr         = nullptr;
        }

        template <typename T, typename... Args> static Object Create(Args&&... args)
        {
            T* ptr                                             = new T(std::forward<Args>(args)...);
            void (*outputStreamFn)(std::ostream&, const void*) = nullptr;
            void (*inputStreamFn)(std::istream&, void*)        = nullptr;
            if constexpr (IConceptIsOutputStreamable<T>)
            {
                outputStreamFn = [](std::ostream& os, const void* obj) { os << *static_cast<const T*>(obj); };
            }
            if constexpr (IConceptIsInputStreamable<T>)
            {
                inputStreamFn = [](std::istream& is, void* obj) { is >> *static_cast<T*>(obj); };
            }

            return Object(
                ptr, []() -> std::type_info const& { return typeid(T); }, [](void* p) { delete static_cast<T*>(p); },
                outputStreamFn, inputStreamFn);
        }

        template <typename T> static Object Create(std::reference_wrapper<T> ref)
        {
            void (*outputStreamFn)(std::ostream&, const void*) = nullptr;
            void (*inputStreamFn)(std::istream&, void*)        = nullptr;
            if constexpr (IConceptIsOutputStreamable<T>)
            {
                outputStreamFn = [](std::ostream& os, const void* obj) { os << *static_cast<const T*>(obj); };
            }
            if constexpr (IConceptIsInputStreamable<T>)
            {
                inputStreamFn = [](std::istream& is, void* obj) { is >> *static_cast<T*>(obj); };
            }
            return Object(
                &ref.get(), []() -> std::type_info const& { return typeid(T); },
                [](void*) { /* No-op destructor for references */ }, outputStreamFn, inputStreamFn);
        }
    };

} // namespace Ifrit::Reflection