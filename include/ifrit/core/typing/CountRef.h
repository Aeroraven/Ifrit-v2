
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

namespace Ifrit
{
    template <class T> class IFRIT_APIDECL CountRef
    {
    public:
        using RefType = T*;
        RefType m_ref;

        CountRef() : m_ref(nullptr) {}
        CountRef(nullptr_t n) : m_ref(nullptr) {}
        CountRef(const CountRef& other)
        {
            m_ref = other.m_ref;
            if (m_ref)
            {
                m_ref->AddRef();
            }
        }

        CountRef(CountRef&& other)
        {
            m_ref       = other.m_ref;
            other.m_ref = nullptr;
        }
        CountRef& operator=(RefType ref)
        {
            if (m_ref != ref)
            {
                auto oldRef = m_ref;
                m_ref       = ref;
                if (m_ref)
                {
                    m_ref->AddRef();
                }
                if (oldRef)
                {
                    oldRef->Release();
                }
            }
            return *this;
        }

        CountRef& operator=(const CountRef& other) { return *this = other.m_ref; }
        CountRef& operator=(CountRef&& other)
        {

            if (this != &other)
            {
                auto oldRef = m_ref;
                m_ref       = other.m_ref;
                other.m_ref = nullptr;
                if (oldRef)
                {
                    oldRef->Release();
                }
            }
            return *this;
        }

        ~CountRef()
        {
            if (m_ref)
            {
                m_ref->Release();
            }
        }

        RefType             operator->() const { return m_ref; }
        RefType             get() const { return m_ref; }
        RefType             get() { return m_ref; }

        IF_FORCEINLINE bool operator==(const CountRef& other) const { return m_ref == other.m_ref; }
        IF_FORCEINLINE bool operator!=(const CountRef& other) const { return m_ref != other.m_ref; }
        IF_FORCEINLINE bool operator==(RefType other) const { return m_ref == other; }
        IF_FORCEINLINE bool operator!=(RefType other) const { return m_ref != other; }
    };

    // TODO: it's a better idea to follow RAII pattern, like make_shared
    template <class T> CountRef<T> MakeCountRef(T* ref)
    {
        CountRef<T> result;
        result.m_ref = ref;
        if (result.m_ref)
        {
            result.m_ref->AddRef();
        }
        return result;
    }

} // namespace Ifrit