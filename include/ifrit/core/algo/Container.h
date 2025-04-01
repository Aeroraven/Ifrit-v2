
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
#include <concepts>

namespace Ifrit
{
    // Reinterpret_cast a pointer to a vector view
    template <typename T> class RVectorView
    {
    private:
        T*     m_data;
        size_t m_size;

    public:
        using iterator       = T*;
        using const_iterator = const T*;

        RVectorView(T* data, size_t size) : m_data(data), m_size(size) {}

        iterator       begin() { return m_data; }
        iterator       end() { return m_data + m_size; }

        const_iterator begin() const { return m_data; }
        const_iterator end() const { return m_data + m_size; }

        size_t         Size() const { return m_size; }
        T&             operator[](size_t index) { return m_data[index]; }
        const T&       operator[](size_t index) const { return m_data[index]; }
        T*             Data() { return m_data; }
        const T*       Data() const { return m_data; }
    };

    // Reinterpret_cast a vector of Ref<Base> to a vector view of Ref<Derived>
    template <typename Derived, typename Base>
        requires std::derived_from<Derived, Base>
    class RDerivedVectorView
    {
    private:
        Vec<Ref<Base>>& m_data;

    public:
        // Ref is wrapper of std::shared_ptr
        struct iterator
        {
            using value_type        = Derived;
            using reference         = Derived&;
            using pointer           = Derived*;
            using difference_type   = std::ptrdiff_t;
            using iterator_category = std::random_access_iterator_tag;

            Vec<Ref<Base>>& m_data;
            size_t          m_index;

            iterator(Vec<Ref<Base>>& data, size_t index) : m_data(data), m_index(index) {}

            iterator& operator++()
            {
                ++m_index;
                return *this;
            }
            iterator operator++(int)
            {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }
            iterator& operator--()
            {
                --m_index;
                return *this;
            }
            iterator operator--(int)
            {
                iterator tmp = *this;
                --(*this);
                return tmp;
            }

            iterator& operator+=(size_t offset)
            {
                m_index += offset;
                return *this;
            }
            iterator& operator-=(size_t offset)
            {
                m_index -= offset;
                return *this;
            }
            iterator operator+(size_t offset) const
            {
                iterator tmp = *this;
                tmp += offset;
                return tmp;
            }
            iterator operator-(size_t offset) const
            {
                iterator tmp = *this;
                tmp -= offset;
                return tmp;
            }
            difference_type operator-(const iterator& other) const { return m_index - other.m_index; }
            bool            operator==(const iterator& other) const { return m_index == other.m_index; }
            bool            operator!=(const iterator& other) const { return m_index != other.m_index; }

            bool            operator<(const iterator& other) const { return m_index < other.m_index; }
            bool            operator>(const iterator& other) const { return m_index > other.m_index; }
            bool            operator<=(const iterator& other) const { return m_index <= other.m_index; }
            bool            operator>=(const iterator& other) const { return m_index >= other.m_index; }

            reference       operator*() { return *std::static_pointer_cast<Derived>(m_data[m_index]); }
            pointer         operator->() { return std::static_pointer_cast<Derived>(m_data[m_index]).get(); }

            reference operator[](size_t index) { return *std::static_pointer_cast<Derived>(m_data[m_index + index]); }
            pointer   operator[](size_t index) const
            {
                return std::static_pointer_cast<Derived>(m_data[m_index + index]).get();
            }
        };

        RDerivedVectorView(Vec<Ref<Base>>& data) : m_data(data) {}

        iterator        begin() { return iterator(m_data, 0); }
        iterator        end() { return iterator(m_data, m_data.size()); }

        size_t          Size() const { return m_data.size(); }
        Ref<Derived>    operator[](size_t index) { return std::static_pointer_cast<Derived>(m_data[index]); }
        Ref<Derived>    operator[](size_t index) const { return std::static_pointer_cast<Derived>(m_data[index]); }

        Vec<Ref<Base>>& Data() { return m_data; }
        const Vec<Ref<Base>>& Data() const { return m_data; }
        Vec<Ref<Base>>&       Get() { return m_data; }
    };

} // namespace Ifrit