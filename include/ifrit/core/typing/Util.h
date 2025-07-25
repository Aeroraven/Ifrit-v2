
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
#ifdef _DEBUG
    #include <stdexcept>
#endif

namespace Ifrit
{
    template <typename T> IntPtr         ToIntPtr(T* ptr) { return reinterpret_cast<IntPtr>(ptr); }
    template <typename T> T*             FromIntPtr(IntPtr ptr) { return reinterpret_cast<T*>(ptr); }

    template <typename T, typename U> T* CheckedCast(U* ptr)
    {
        // Reference: NVIDIAGameWorks/nvrhi/blob/main/include/nvrhi/nvrhi.h
#ifdef _DEBUG
        // dynamic cast
        if (ptr == nullptr)
        {
            return nullptr;
        }
        auto casted = dynamic_cast<T*>(ptr);
        if (casted == nullptr)
        {
            throw std::runtime_error("Invalid cast");
        }
        return casted;
#else
        return static_cast<T*>(ptr);
#endif
    }

    template <typename T, typename U> const T* CheckedCast(const U* ptr)
    {
#ifdef _DEBUG
        // dynamic cast
        if (ptr == nullptr)
        {
            return nullptr;
        }
        auto casted = dynamic_cast<const T*>(ptr);
        if (casted == nullptr)
        {
            throw std::runtime_error("Invalid cast");
        }
        return casted;
#else
        return static_cast<const T*>(ptr);
#endif
    }

    template <typename T, typename U> std::shared_ptr<T> CheckedPointerCast(const std::shared_ptr<U>& ptr)
    {
#ifdef _DEBUG
        // dynamic cast
        if (ptr == nullptr)
        {
            return nullptr;
        }
        auto casted = std::dynamic_pointer_cast<T>(ptr);
        if (casted == nullptr)
        {
            throw std::runtime_error("Invalid cast");
        }
        return casted;
#else
        return std::static_pointer_cast<T>(ptr);
#endif
    }

    template <typename T> T SizeCast(size_t size) { return static_cast<T>(size); }

    // Non-copyable class:
    // https://www.boost.org/doc/libs/1_41_0/boost/noncopyable.hpp
    class IFRIT_APIDECL     NonCopyable
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