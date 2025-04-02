
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
        // Head is hold by m_Dummy's m_Next
        RPooledConcurrentQueueElement*             m_Dummy    = nullptr;
        IntPtr                                     m_DummyPtr = IntPtr(0);
        Atomic<IntPtr>                             m_Tail     = 0;
        Atomic<u64>                                m_RefCount = 0;
        RObjectPool<RPooledConcurrentQueueElement> m_Pool;

    private:
        void EnqueueNode(RIndexedPtr nodeIndex)
        {
            auto nodePtr    = m_Pool.GetPtrFromIndex(nodeIndex);
            nodePtr->m_Next = 0;

            IntPtr prev      = m_Tail.exchange(nodeIndex.Ptr(), std::memory_order::acq_rel);
            auto   prevNode  = m_Pool.GetPtrFromIndex(RIndexedPtr(prev));
            prevNode->m_Next = nodeIndex.Ptr();
            m_RefCount.fetch_add(1, std::memory_order::acq_rel);
        }

        RIndexedPtr DequeueNode()
        {
            auto                           head        = m_Dummy->m_Next.load(std::memory_order::acquire);
            IntPtr                         newHead     = 0;
            IntPtr                         nextNodeIdx = 0;
            RPooledConcurrentQueueElement* cHead;
            RIndexedPtr                    cHeadIdxPtr;
            RPooledConcurrentQueueElement* nextNode = nullptr;
            do
            {
                cHeadIdxPtr = RIndexedPtr(head);
                cHead       = m_Pool.GetPtrFromIndex(RIndexedPtr(head));
                if (cHead == nullptr)
                {
                    return RIndexedPtr(0);
                }
                nextNodeIdx = cHead->m_Next.load(std::memory_order::acquire);
                nextNode    = m_Pool.GetPtrFromIndex(RIndexedPtr(nextNodeIdx));
                newHead     = nextNodeIdx;
            }
            while (!m_Dummy->m_Next.compare_exchange_strong(
                head, newHead, std::memory_order::acq_rel, std::memory_order::acquire));
            m_RefCount.fetch_sub(1, std::memory_order::acq_rel);
            return cHeadIdxPtr;
        }

    public:
        RPooledConcurrentQueue()
        {
            m_DummyPtr      = m_Pool.AllocateIndexed(nullptr);
            m_Dummy         = m_Pool.GetPtrFromIndex({ m_DummyPtr });
            m_Dummy->m_Next = 0;
            m_Tail          = m_DummyPtr;
        }
        ~RPooledConcurrentQueue()
        {
            while (!Empty())
            {
                DequeueNode();
            }
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
                throw std::runtime_error("Queue is empty");
            }
            auto node = m_Pool.GetPtrFromIndex(nodeIdx);
            auto data = std::move(node->m_Data);
            m_Pool.DeallocateIndexed(nodeIdx);
            return data;
        }

        T Peek()
        {
            auto node = m_Dummy->m_Next.load(std::memory_order::acquire);
            if (node == 0)
            {
                throw std::runtime_error("Queue is empty");
            }
            auto nodePtr = m_Pool.GetPtrFromIndex(node);
            return nodePtr->m_Data;
        }
        bool Empty() { return m_Dummy->m_Next.load(std::memory_order::acquire) == 0; }
        u64  Size() { return m_RefCount.load(std::memory_order::acquire); }
    };
} // namespace Ifrit