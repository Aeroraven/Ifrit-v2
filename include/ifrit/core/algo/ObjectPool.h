
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
#include "ifrit/core/typing/Util.h"
#include <memory>

namespace Ifrit
{
    template <typename T, typename Alloc = std::allocator<T>> class RObjectPool : public NonCopyable
    {
    private:
        Vec<T*>  m_Memory;
        Vec<u64> m_MemorySize; // Relative to sizeof(T)
        Vec<T*>  m_VacantList;
        Alloc    m_Allocator;

    public:
        RObjectPool() = default;
        ~RObjectPool()
        {
            for (int i = 0; i < m_Memory.size(); ++i)
            {
                m_Allocator.deallocate(m_Memory[i], m_MemorySize[i]);
            }
            m_Memory.clear();
            m_MemorySize.clear();
            m_VacantList.clear();
        }

        IF_FORCEINLINE u64             GetNextMemAllocSize() { return 128ull << m_Memory.size(); }

        template <typename... Args> T* Allocate(Args&&... args)
        {
            if (m_VacantList.empty())
            {
                auto memSize = GetNextMemAllocSize();
                auto mem     = m_Allocator.allocate(memSize);
                m_Memory.push_back(mem);
                m_MemorySize.push_back(memSize);
                for (u64 i = 0; i < memSize; ++i)
                {
                    m_VacantList.push_back(mem + i);
                }
            }
            auto obj = m_VacantList.back();
            m_VacantList.pop_back();
            new (obj) T(std::forward<Args>(args)...);
            return obj;
        }

        IF_FORCEINLINE void Deallocate(T* obj)
        {
            obj->~T();
            m_VacantList.push_back(obj);
        }
    };
} // namespace Ifrit