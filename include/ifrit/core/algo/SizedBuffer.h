
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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

namespace Ifrit
{

    class SizedBuffer
    {
    private:
        Vec<u8> m_Data;

    public:
        SizedBuffer() = default;
        SizedBuffer(u32 size) : m_Data(size) {}
        SizedBuffer(void* ptr, u32 size) : m_Data(size) { memcpy(m_Data.data(), ptr, size); }

        template <typename T> SizedBuffer(const Vec<T>& data) : m_Data(data.size() * sizeof(T))
        {
            memcpy(m_Data.data(), data.data(), m_Data.size());
        }

        SizedBuffer(const SizedBuffer& other) : m_Data(other.m_Data) {}
        SizedBuffer(SizedBuffer&& other) noexcept : m_Data(std::move(other.m_Data)) {}

        SizedBuffer& operator=(const SizedBuffer& other)
        {
            if (this != &other)
            {
                m_Data = other.m_Data;
            }
            return *this;
        }
        SizedBuffer& operator=(SizedBuffer&& other) noexcept
        {
            if (this != &other)
            {
                m_Data = std::move(other.m_Data);
            }
            return *this;
        }
        SizedBuffer& operator=(const Vec<u8>& other)
        {
            m_Data = other;
            return *this;
        }
        SizedBuffer& operator=(Vec<u8>&& other) noexcept
        {
            m_Data = std::move(other);
            return *this;
        }
        SizedBuffer& operator=(const Vec<u8>&& other)
        {
            m_Data = other;
            return *this;
        }

        ~SizedBuffer() = default;

        void*       GetData() { return m_Data.data(); }
        const void* GetData() const { return m_Data.data(); }
        u32         GetSize() const { return static_cast<u32>(m_Data.size()); }

        u8&         operator[](u32 index) { return m_Data[index]; }
        const u8&   operator[](u32 index) const { return m_Data[index]; }

        void        CopyFromRaw(const void* ptr, u32 size)
        {
            m_Data.resize(size);
            memcpy(m_Data.data(), ptr, size);
        }

        template <typename T> Vec<T> ToByteVector() const
        {
            static_assert(sizeof(T) == 1, "T must be a byte-sized type");
            Vec<T> result(m_Data.size() / sizeof(T));
            memcpy(result.data(), m_Data.data(), m_Data.size());
            return result;
        }

        String ToString() const { return String(reinterpret_cast<const char*>(m_Data.data()), m_Data.size()); }
    };
} // namespace Ifrit