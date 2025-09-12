
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
    template <class T> class IFRIT_APIDECL TCountRef
    {
    public:
        using RefType = T*;
        RefType m_ref;

        TCountRef() : m_ref(nullptr) {}
        TCountRef(nullptr_t n) : m_ref(nullptr) {}
        TCountRef(const TCountRef& other)
        {
            m_ref = other.m_ref;
            if (m_ref)
            {
                m_ref->AddRef();
            }
        }

        TCountRef(TCountRef&& other)
        {
            m_ref       = other.m_ref;
            other.m_ref = nullptr;
        }
        TCountRef& operator=(RefType ref)
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

        TCountRef& operator=(const TCountRef& other)
        {
            if (this != &other)
            {
                auto oldRef = m_ref;
                m_ref       = other.m_ref;
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
        TCountRef& operator=(TCountRef&& other)
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

        ~TCountRef()
        {
            if (m_ref)
            {
                m_ref->Release();
            }
        }

        RefType             operator->() const { return m_ref; }
        RefType             Get() const { return m_ref; }
        RefType             Get() { return m_ref; }

        IF_FORCEINLINE bool operator==(const TCountRef& other) const { return m_ref == other.m_ref; }
        IF_FORCEINLINE bool operator!=(const TCountRef& other) const { return m_ref != other.m_ref; }
        IF_FORCEINLINE bool operator==(RefType other) const { return m_ref == other; }
        IF_FORCEINLINE bool operator!=(RefType other) const { return m_ref != other; }

        u32                 GetRefCount() const
        {
            if (m_ref)
            {
                return m_ref->GetRefCount();
            }
            return 0;
        }
    };

    // TODO: it's a better idea to follow RAII pattern, like make_shared
    template <class T> TCountRef<T> MakeCountRef(T* ref)
    {
        TCountRef<T> result;
        result.m_ref = ref;
        if (result.m_ref)
        {
            result.m_ref->AddRef();
        }
        return result;
    }

} // namespace Ifrit