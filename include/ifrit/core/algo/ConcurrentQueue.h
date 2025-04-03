
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
#include "ifrit/core/algo/Memory.h"
#include <memory>

namespace Ifrit
{
    // Warning: ABA problem is not resolved yet
    template <typename T> class RPooledConcurrentQueue : public NonCopyable
    {
    public:
        struct RPooledConcurrentQueueElement
        {
            Atomic<IntPtr> m_Next    = 0;
            u8             m_IsDummy = 0;
            union
            {
                Atomic<u64> m_Dummy = 0;
                T           m_Data;
            };
            explicit RPooledConcurrentQueueElement(nullptr_t x) : m_Next(0), m_Dummy(0), m_IsDummy(1) {}
            explicit RPooledConcurrentQueueElement(T&& data) : m_Next(0), m_Data(std::move(data)), m_IsDummy(0) {}
            explicit RPooledConcurrentQueueElement(const T& data) : m_Next(0), m_Data(data), m_IsDummy(0) {}

            ~RPooledConcurrentQueueElement()
            {
                if (m_IsDummy == 0)
                {
                    m_Data.~T();
                }
            }
        };

    private:
        Atomic<IntPtr>                             m_Head     = 0;
        Atomic<IntPtr>                             m_Tail     = 0;
        Atomic<u64>                                m_RefCount = 0;
        RObjectPool<RPooledConcurrentQueueElement> m_Pool;

    private:
        void EnqueueNode(RIndexedPtr nodeIndex)
        {
            auto qId     = nodeIndex.Ptr();
            auto qPtr    = m_Pool.GetPtrFromIndex(nodeIndex);
            qPtr->m_Next = 0;

            bool succ;
            auto pId  = m_Tail.load(std::memory_order::acquire);
            auto pPtr = m_Pool.GetPtrFromIndex(RIndexedPtr(pId));
            do
            {
                pId       = m_Tail.load(std::memory_order::acquire);
                pPtr      = m_Pool.GetPtrFromIndex(RIndexedPtr(pId));
                IntPtr ep = 0ull;
                succ      = pPtr->m_Next.compare_exchange_strong(
                    ep, qId, std::memory_order::acq_rel, std::memory_order::acquire);

                if (!succ)
                {
                    auto expected = pId;
                    auto tailPtr  = m_Pool.GetPtrFromIndex(RIndexedPtr(expected));
                    auto pNext    = pPtr->m_Next.load(std::memory_order::acquire);
                    m_Tail.compare_exchange_strong(
                        expected, pNext, std::memory_order::acq_rel, std::memory_order::acquire);
                }
            }
            while (!succ);
            m_Tail.compare_exchange_strong(pId, qId, std::memory_order::acq_rel, std::memory_order::acquire);
            m_RefCount.fetch_add(1, std::memory_order::acq_rel);
        }

        RIndexedPtr DequeueNode()
        {
            auto pId  = m_Head.load();
            auto pPtr = m_Pool.GetPtrFromIndex(RIndexedPtr(pId));

            auto expected = pId;
            auto pNext    = pPtr->m_Next.load(std::memory_order::acquire);
            do
            {
                pId  = m_Head.load(std::memory_order::acquire);
                pPtr = m_Pool.GetPtrFromIndex(RIndexedPtr(pId));
                if (pPtr->m_Next.load(std::memory_order::acquire) == 0)
                {
                    return RIndexedPtr(0);
                }
                expected = pId;
                pNext    = pPtr->m_Next.load(std::memory_order::acquire);
            }
            while (!m_Head.compare_exchange_strong(
                expected, pNext, std::memory_order::acq_rel, std::memory_order::acquire));

            m_RefCount.fetch_sub(1, std::memory_order::acq_rel);
            return RIndexedPtr(pPtr->m_Next.load());
        }

    public:
        RPooledConcurrentQueue()
        {
            auto m_DummyPtr = m_Pool.AllocateIndexed(nullptr);
            auto m_Dummy    = m_Pool.GetPtrFromIndex({ m_DummyPtr });
            m_Dummy->m_Next = 0;

            m_Head.store(m_DummyPtr, std::memory_order::release);
            m_Tail.store(m_DummyPtr, std::memory_order::release);
        }
        ~RPooledConcurrentQueue()
        {
            while (!Empty())
            {
                DequeueNode();
            }
            // Remaining one dummy node
            auto dummyPtr = m_Head.load(std::memory_order::acquire);
            m_Pool.DeallocateIndexed(RIndexedPtr(dummyPtr));
        }

        void Enqueue(T&& data)
        {
            auto node = m_Pool.AllocateIndexed(std::move(data));
            EnqueueNode(node);
        }
        void Enqueue(const T& data)
        {
            auto node = m_Pool.AllocateIndexed(data);
            EnqueueNode(node);
        }

        T Dequeue()
        {
            auto nodeIdx = DequeueNode();
            if (nodeIdx == nullptr)
            {
                // if default constructor is available,create a default object
                if IF_CONSTEXPR (std::is_default_constructible_v<T>)
                {
                    return T();
                }
                throw std::runtime_error(
                    "RPooledConcurrentQueue: Queue is empty. Add a default constructor to suppress this error.");
            }
            auto node = m_Pool.GetPtrFromIndex(nodeIdx);
            auto data = std::move(node->m_Data);
            m_Pool.DeallocateIndexed(nodeIdx);
            return data;
        }

        T Peek()
        {
            auto headPtr = m_Pool.GetPtrFromIndex(m_Head.load(std::memory_order::acquire));
            auto nextPtr = headPtr->m_Next.load(std::memory_order::acquire);
            if (nextPtr == 0)
            {
                // if default constructor is available,create a default object
                if IF_CONSTEXPR (std::is_default_constructible_v<T>)
                {
                    return T();
                }
                throw std::runtime_error(
                    "RPooledConcurrentQueue: Queue is empty. Add a default constructor to suppress this error.");
            }
            auto node = m_Pool.GetPtrFromIndex(nextPtr);
            return node->m_Data;
        }
        bool Empty()
        {
            auto headPtr = m_Pool.GetPtrFromIndex(RIndexedPtr(m_Head.load(std::memory_order::acquire)));
            auto nextPtr = headPtr->m_Next.load(std::memory_order::acquire);
            return nextPtr == 0;
        }

        u64 Size() { return m_RefCount.load(std::memory_order::acquire); }
    };
} // namespace Ifrit