
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

    class RSizedBuffer
    {
    private:
        Vec<u8> m_Data;

    public:
        RSizedBuffer() = default;
        RSizedBuffer(u32 size) : m_Data(size) {}
        RSizedBuffer(void* ptr, u32 size) : m_Data(size) { memcpy(m_Data.data(), ptr, size); }

        template <typename T> RSizedBuffer(const Vec<T>& data) : m_Data(data.size() * sizeof(T))
        {
            memcpy(m_Data.data(), data.data(), m_Data.size());
        }

        RSizedBuffer(const RSizedBuffer& other) : m_Data(other.m_Data) {}
        RSizedBuffer(RSizedBuffer&& other) noexcept : m_Data(std::move(other.m_Data)) {}

        RSizedBuffer& operator=(const RSizedBuffer& other)
        {
            if (this != &other)
            {
                m_Data = other.m_Data;
            }
            return *this;
        }
        RSizedBuffer& operator=(RSizedBuffer&& other) noexcept
        {
            if (this != &other)
            {
                m_Data = std::move(other.m_Data);
            }
            return *this;
        }
        RSizedBuffer& operator=(const Vec<u8>& other)
        {
            m_Data = other;
            return *this;
        }
        RSizedBuffer& operator=(Vec<u8>&& other) noexcept
        {
            m_Data = std::move(other);
            return *this;
        }
        RSizedBuffer& operator=(const Vec<u8>&& other)
        {
            m_Data = other;
            return *this;
        }

        ~RSizedBuffer() = default;

        void*       GetData() { return m_Data.data(); }
        const void* GetData() const { return m_Data.data(); }
        u32         GetSize() const { return static_cast<u32>(m_Data.size()); }

        u8&         operator[](u32 index) { return m_Data[index]; }
        const u8&   operator[](u32 index) const { return m_Data[index]; }

        void        CopyFromRaw(void* ptr, u32 size)
        {
            m_Data.resize(size);
            memcpy(m_Data.data(), ptr, size);
        }
    };
} // namespace Ifrit