
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/typing/Traits.h"
#ifdef _DEBUG
    #include <stdexcept>
    #define IFRIT_UNCHECKED_NOEXCEPT
#else
    #define IFRIT_UNCHECKED_NOEXCEPT noexcept
#endif

namespace Ifrit
{

    template <typename T> IntPtr ToIntPtr(T* ptr) { return reinterpret_cast<IntPtr>(ptr); }
    template <typename T> T*     FromIntPtr(IntPtr ptr) { return reinterpret_cast<T*>(ptr); }

    // Reference: NVIDIAGameWorks/nvrhi/blob/main/include/nvrhi/nvrhi.h
    template <typename T, typename U>
        requires IConceptIsDynamicallyConvertible<T*, U*>
    IF_NODISCARD constexpr inline T* CheckedCast(U* ptr) IFRIT_UNCHECKED_NOEXCEPT
    {

#ifdef _DEBUG
        if (ptr == nullptr)
            return nullptr;
        auto casted = dynamic_cast<T*>(ptr);
        if (casted == nullptr)
            throw std::runtime_error("Invalid cast");
        return casted;
#else
        return static_cast<T*>(ptr);
#endif
    }

    template <typename T, typename U>
        requires IConceptIsDynamicallyConvertible<T*, U*>
    IF_NODISCARD inline T* ForcedCheckedCast(U* ptr)
    {
        // static cast
        if (ptr == nullptr)
            return nullptr;
        auto casted = dynamic_cast<T*>(ptr);
        if (casted == nullptr)
            std::abort();
        return casted;
    }

    template <typename T, typename U>
        requires IConceptIsDynamicallyConvertible<T*, U*>
    IF_NODISCARD inline Ref<T> CheckedPointerCast(const Ref<U>& ptr)
    {
#ifdef _DEBUG
        // dynamic cast
        if (ptr == nullptr)
            return nullptr;
        auto casted = std::dynamic_pointer_cast<T>(ptr);
        if (casted == nullptr)
            throw std::runtime_error("Invalid cast");
        return casted;
#else
        return std::static_pointer_cast<T>(ptr);
#endif
    }

    template <IIntegral T> T IF_NODISCARD constexpr IF_FORCEINLINE SizeCast(size_t size) noexcept
    {
        return static_cast<T>(size);
    }

    // Non-copyable class:
    // https://www.boost.org/doc/libs/1_41_0/boost/noncopyable.hpp
    class IFRIT_APIDECL NonCopyable
    {
    protected:
        NonCopyable()  = default;
        ~NonCopyable() = default;

    private:
        NonCopyable(const NonCopyable&)            = delete;
        NonCopyable& operator=(const NonCopyable&) = delete;
    };

    struct IFRIT_APIDECL NonCopyableStruct
    {
        NonCopyableStruct()  = default;
        ~NonCopyableStruct() = default;

    private:
        NonCopyableStruct(const NonCopyableStruct&)            = delete;
        NonCopyableStruct& operator=(const NonCopyableStruct&) = delete;
    };

} // namespace Ifrit