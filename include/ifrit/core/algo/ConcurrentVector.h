
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
    // Vector that only grows in the back
    // Reference:
    // At commit: 047c35299a4c8573c41ebc84e90587889cb0e0c6
    // /src/engine/tilerastercuda/TileRasterInvocationCuda.cu
    template <typename T, u32 TPageNums = 4096, u32 TPageSize = 16384> class RConcurrentGrowthVector
    {
    private:
        Atomic<T*> m_Pages[TPageNums];
        Atomic<u32>         m_SpinLock;
        Atomic<u32>         m_CurrentBack = 0;

    private:
        void AutoGrow(u32 pos)
        {
            auto        pageIndex  = pos / TPageSize;
            auto        pageOffset = pos % TPageSize;

            volatile T* pagePtr;
            while ((pagePtr = m_Pages[pageIndex].load(std::memory_order::acquire)) == nullptr)
            {
                // Spin lock, compare and swap
                u32 expected = 0;
                while (!m_SpinLock.compare_exchange_strong(
                    expected, 1, std::memory_order::acq_rel, std::memory_order::acquire))
                {
                    expected = 0;
                }

                auto pagePtr = m_Pages[pageIndex].load(std::memory_order::acquire);
                if (pagePtr == nullptr)
                {
                    // Here, create a new page
                    pagePtr = new T[TPageSize];
                    m_Pages[pageIndex].store(pagePtr, std::memory_order::release);
                }
                m_SpinLock.store(0, std::memory_order::release);
            }
        }

    public:
        RConcurrentGrowthVector()
        {
            for (u32 i = 0; i < TPageNums; ++i)
            {
                m_Pages[i].store(nullptr, std::memory_order::release);
            }
            m_SpinLock.store(0, std::memory_order::release);
            m_CurrentBack.store(0, std::memory_order::release);
        }
        ~RConcurrentGrowthVector()
        {
            for (u32 i = 0; i < TPageNums; ++i)
            {
                auto pagePtr = m_Pages[i].load(std::memory_order::acquire);
                if (pagePtr != nullptr)
                {
                    delete[] pagePtr;
                }
            }
        }
        void PushBack(const T& val)
        {
            auto pos        = m_CurrentBack.fetch_add(1, std::memory_order::acq_rel);
            auto pageIndex  = pos / TPageSize;
            auto pageOffset = pos % TPageSize;
            AutoGrow(pos);
            m_Pages[pageIndex].load(std::memory_order::acquire)[pageOffset] = val;
        }

        void PushBack(T&& val)
        {
            auto pos        = m_CurrentBack.fetch_add(1, std::memory_order::acq_rel);
            auto pageIndex  = pos / TPageSize;
            auto pageOffset = pos % TPageSize;
            AutoGrow(pos);
            m_Pages[pageIndex].load(std::memory_order::acquire)[pageOffset] = std::move(val);
        }

        T operator[](u32 index)
        {
            auto pageIndex  = index / TPageSize;
            auto pageOffset = index % TPageSize;
            return m_Pages[pageIndex].load(std::memory_order::acquire)[pageOffset];
        }
    };
} // namespace Ifrit