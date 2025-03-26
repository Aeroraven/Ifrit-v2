
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/ApiConv.h"

namespace Ifrit::Graphics::Rhi
{
    template <class T>
    class IFRIT_APIDECL RhiCountRef
    {
    public:
        using RefType = T*;
        RefType m_ref;

        RhiCountRef()
            : m_ref(nullptr) {}
        RhiCountRef(nullptr_t n)
            : m_ref(nullptr) {}
        RhiCountRef(const RhiCountRef& other)
        {
            m_ref = other.m_ref;
            if (m_ref)
            {
                m_ref->AddRef();
            }
        }

        RhiCountRef(RhiCountRef&& other)
        {
            m_ref       = other.m_ref;
            other.m_ref = nullptr;
        }
        RhiCountRef& operator=(RefType ref)
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

        RhiCountRef& operator=(const RhiCountRef& other) { return *this = other.m_ref; }
        RhiCountRef& operator=(RhiCountRef&& other)
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

        ~RhiCountRef()
        {
            if (m_ref)
            {
                m_ref->Release();
            }
        }

        RefType             operator->() const { return m_ref; }
        RefType             get() const { return m_ref; }
        RefType             get() { return m_ref; }

        IF_FORCEINLINE bool operator==(const RhiCountRef& other) const { return m_ref == other.m_ref; }
        IF_FORCEINLINE bool operator!=(const RhiCountRef& other) const { return m_ref != other.m_ref; }
        IF_FORCEINLINE bool operator==(RefType other) const { return m_ref == other; }
        IF_FORCEINLINE bool operator!=(RefType other) const { return m_ref != other; }
    };

    // TODO: it's a better idea to follow RAII pattern, like make_shared
    template <class T>
    RhiCountRef<T> MakeRhiCountRef(T* ref)
    {
        RhiCountRef<T> result;
        result.m_ref = ref;
        if (result.m_ref)
        {
            result.m_ref->AddRef();
        }
        return result;
    }

} // namespace Ifrit::Graphics::Rhi