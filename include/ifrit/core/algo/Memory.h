
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
#include "ifrit/core/typing/Util.h"
#include "ifrit/core/logging/Logging.h"
#include <memory>

namespace Ifrit
{

    enum class RThreadSafePagedVectorFillingPolicy
    {
        None,
        IncrementBase1,
    };

    // Paged Queue that only allows push back and pop front. (Maybe circular queue is better?)
    //
    // I won't know whether it's thread safe. But it works when I am writing the CUDA soft renderer.
    // It causes warp divergence, but now it's on CPU.
    // TODO: I want those freed pages to be reused.
    // Reference:
    // At commit: 047c35299a4c8573c41ebc84e90587889cb0e0c6
    // /src/engine/tilerastercuda/TileRasterInvocationCuda.cu
    template <typename T, u32 TPageNums = 4096, u32 TPageSize = 16384>
        requires std::is_integral_v<T> || std::is_floating_point_v<T>
    class RThreadSafePagedVector
    {
    private:
        Atomic<volatile T*> m_Pages[TPageNums];
        Atomic<u32>         m_SpinLock;
        Atomic<u32>         m_CurrentBack  = 0;
        Atomic<u32>         m_CurrentFront = 0;
        Atomic<u32>         m_CurrentSize  = 0;

    public:
        RThreadSafePagedVector()
        {
            for (u32 i = 0; i < TPageNums; ++i)
            {
                m_Pages[i].store(nullptr, std::memory_order::release);
            }
            m_SpinLock.store(0, std::memory_order::release);
            m_CurrentBack.store(0, std::memory_order::release);
            m_CurrentFront.store(0, std::memory_order::release);
            m_CurrentSize.store(0, std::memory_order::release);
        }
        ~RThreadSafePagedVector()
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
        void PushBack(T val)
        {
            auto        pos        = m_CurrentBack.fetch_add(1, std::memory_order::acq_rel);
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
            m_Pages[pageIndex].load(std::memory_order::acquire)[pageOffset] = val;
            m_CurrentSize.fetch_add(1, std::memory_order::acq_rel);
        }

        T PopFront()
        {
            auto curSize = m_CurrentSize.load(std::memory_order::acquire);

            if (curSize == 0)
            {
                return T(0);
            }
            m_CurrentSize.fetch_sub(1, std::memory_order::acq_rel);
            auto pos        = m_CurrentFront.fetch_add(1, std::memory_order::acq_rel);
            auto pageIndex  = pos / TPageSize;
            auto pageOffset = pos % TPageSize;
            // printf("Current size: %u %u\n", curSize,
            // m_Pages[pageIndex].load(std::memory_order::acquire)[pageOffset]);
            return m_Pages[pageIndex].load(std::memory_order::acquire)[pageOffset];
        }

        T operator[](u32 index)
        {
            auto pageIndex  = index / TPageSize;
            auto pageOffset = index % TPageSize;
            return m_Pages[pageIndex].load(std::memory_order::acquire)[pageOffset];
        }
    };

    struct RIndexedPtr
    {
        IntPtr m_Ptr = 0;
        operator IntPtr() const { return m_Ptr; }
        operator u64() const { return static_cast<u64>(m_Ptr); }
        operator u32() const { return static_cast<u32>(m_Ptr); }

        IntPtr& Ptr() { return m_Ptr; }
        bool    operator==(const RIndexedPtr& other) const { return m_Ptr == other.m_Ptr; }
        bool    operator!=(const RIndexedPtr& other) const { return m_Ptr != other.m_Ptr; }

        // Check if is nullptr
        bool    operator==(nullptr_t) const { return m_Ptr == 0; }
        bool    operator!=(nullptr_t) const { return m_Ptr != 0; }
    };

    template <typename T, typename Alloc = std::allocator<T>, typename AtomicAlloc = std::allocator<Atomic<u64>>>
    class RObjectPool : public NonCopyable
    {
    private:
        Vec<T*>                     m_Memory;
        Vec<u64>                    m_MemorySize; // Relative to sizeof(T)
        RThreadSafePagedVector<u64> m_VacantList;

        Alloc                       m_Allocator;
        Vec<T*>                     m_IdToPtr;

        Vec<Atomic<u64>*>           m_AtomMemory;
        Vec<u64>                    m_AtomMemorySize; // Relative to sizeof(Atomic<u64>)
        Vec<Atomic<u64>*>           m_AtomVacantList;
        AtomicAlloc                 m_AtomicAllocator;

        Atomic<u64>                 m_CASLock;
        Atomic<i64>                 m_VacantPos       = 0;
        Atomic<u64>                 m_VacantAvailable = 0;

        std::mutex                  m_Mutex;

    public:
        // Definitions of the RObjectRef
        class IFRIT_APIDECL RObjectRef
        {
        private:
            RIndexedPtr  m_Index;
            T*           m_Ref;
            Atomic<u64>* m_RefCount;
            RObjectPool* m_Pool;

        private:
            void Release()
            {
                if (!m_Ref)
                    return;
                auto x = m_RefCount->fetch_sub(1, std::memory_order::acq_rel);
                if (x == 1)
                {

                    m_Pool->DeallocateIndexed(m_Index);
                    m_Pool->DeallocateAtomic(m_RefCount);
                    m_Ref      = nullptr;
                    m_RefCount = nullptr;
                }
            }

        public:
            RObjectRef() : m_Index(0), m_Ref(nullptr), m_RefCount(nullptr), m_Pool(nullptr) {}
            explicit RObjectRef(RIndexedPtr idx, T* ref, Atomic<u64>* refCount, RObjectPool* pool)
                : m_Index(idx), m_Ref(ref), m_RefCount(refCount), m_Pool(pool)
            {
                m_RefCount->store(1, std::memory_order::release);
            }

            RObjectRef(const RObjectRef& other)
                : m_Index(other.m_Index), m_Ref(other.m_Ref), m_RefCount(other.m_RefCount), m_Pool(other.m_Pool)
            {
                if (m_RefCount)
                {
                    m_RefCount->fetch_add(1, std::memory_order::relaxed);
                }
            }

            RObjectRef(RObjectRef&& other) IF_NOEXCEPT :
                m_Index(other.m_Index),
                m_Ref(other.m_Ref),
                m_RefCount(other.m_RefCount),
                m_Pool(other.m_Pool)
            {
                other.m_Index    = { 0 };
                other.m_Ref      = nullptr;
                other.m_RefCount = nullptr;
                other.m_Pool     = nullptr;
            }

            RObjectRef& operator=(const RObjectRef& other)
            {
                if (this != &other)
                {
                    Release();
                    m_Index    = other.m_Index;
                    m_Ref      = other.m_Ref;
                    m_RefCount = other.m_RefCount;
                    m_Pool     = other.m_Pool;
                    if (m_RefCount)
                    {
                        m_RefCount->fetch_add(1, std::memory_order::relaxed);
                    }
                }
                return *this;
            }

            RObjectRef& operator=(RObjectRef&& other)
            {
                if (this != &other)
                {
                    Release();
                    m_Index          = other.m_Index;
                    m_Ref            = other.m_Ref;
                    m_RefCount       = other.m_RefCount;
                    m_Pool           = other.m_Pool;
                    other.m_Index    = { 0 };
                    other.m_Ref      = nullptr;
                    other.m_RefCount = nullptr;
                    other.m_Pool     = nullptr;
                }
                return *this;
            }

            ~RObjectRef() { Release(); }

            T*   operator->() const { return m_Ref; }
            T&   operator*() const { return *m_Ref; }

            T*   Get() const { return m_Ref; }

            bool operator==(const RObjectRef& other) const { return m_Ref == other.m_Ref; }
            bool operator!=(const RObjectRef& other) const { return m_Ref != other.m_Ref; }
        };

    private:
        void ExpandPool()
        {
            auto memSize = GetNextMemAllocSize();
            auto mem     = m_Allocator.allocate(memSize);
            m_Memory.push_back(mem);
            m_MemorySize.push_back(memSize);
            auto curCandidates = m_IdToPtr.size();
            for (u64 i = 0; i < memSize; ++i)
            {
                m_IdToPtr.push_back(mem + i);
            }
            for (u64 i = 0; i < memSize; ++i)
            {
                VacantPlaceBack(RIndexedPtr(curCandidates + i));
            }
        }

        template <typename... Args> Pair<T*, IntPtr> AllocateInternal(Args&&... args)
        {
            auto id = m_VacantList.PopFront();
            while (id = m_VacantList.PopFront(), id == 0)
            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                while (id = m_VacantList.PopFront(), id == 0)
                {
                    ExpandPool();
                }
            }

            auto obj    = id;
            auto objPtr = m_IdToPtr[obj];
            new (objPtr) T(std::forward<Args>(args)...);
            return { objPtr, obj };
        }

        void        VacantPlaceBack(RIndexedPtr index) { m_VacantList.PushBack(index.Ptr()); }

        RIndexedPtr VacantPopBack() { return RIndexedPtr(m_VacantList.PopFront()); }

    public:
        // Definitions of RObjectPool

        RObjectPool()
        {
            m_IdToPtr.push_back(nullptr); // 0 is reserved for nullptr
        }
        ~RObjectPool()
        {
            for (int i = 0; i < m_Memory.size(); ++i)
            {
                m_Allocator.deallocate(m_Memory[i], m_MemorySize[i]);
            }
            m_Memory.clear();
            m_MemorySize.clear();
        }

        // This interface is aimed to alleviate the ABA problem for some concurrent
        // algorithms. It is not recommended to use this interface in normal cases.
        template <typename... Args> RIndexedPtr AllocateIndexed(Args&&... args)
        {
            auto obj = AllocateInternal(std::forward<Args>(args)...);
            return RIndexedPtr{ obj.second };
        }

        T* GetPtrFromIndex(RIndexedPtr index)
        {
            if (index == nullptr)
            {
                return nullptr;
            }
            if (index.m_Ptr >= m_IdToPtr.size())
            {
                iError("RObjectQueue: Invalid index: {}. Max size: {}", index.m_Ptr, m_IdToPtr.size());
                std::abort();
            }
            return m_IdToPtr[index.m_Ptr];
        }

        IF_FORCEINLINE void DeallocateIndexed(RIndexedPtr index)
        {
            if (index == nullptr)
            {
                return;
            }
            if (index.m_Ptr >= m_IdToPtr.size())
            {
                iError("RObjectQueue: Invalid index: {}. Max size: {}", index.m_Ptr, m_IdToPtr.size());
                std::abort();
            }
            auto obj = m_IdToPtr[index.m_Ptr];
            obj->~T();
            // m_VacantList.push_back(index.m_Ptr);
            VacantPlaceBack(index);
        }

    private:
        Atomic<u64>* AllocateAtomic()
        {
            if (m_AtomVacantList.empty())
            {
                auto memSize = GetNextMemAllocSize();
                auto mem     = m_AtomicAllocator.allocate(memSize);
                m_AtomMemory.push_back(mem);
                m_AtomMemorySize.push_back(memSize);
                for (u64 i = 0; i < memSize; ++i)
                {
                    m_AtomVacantList.push_back(mem + i);
                }
            }
            auto obj = m_AtomVacantList.back();
            m_AtomVacantList.pop_back();
            new (obj) Atomic<u64>(0);
            return obj;
        }

        IF_FORCEINLINE void DeallocateAtomic(Atomic<u64>* obj)
        {
            obj->~Atomic<u64>();
            m_AtomVacantList.push_back(obj);
        }
        IF_FORCEINLINE u64 GetNextMemAllocSize() { return 16384; }

    public:
        template <typename... Args> RObjectRef<T> Create(Args&&... args)
        {
            auto obj      = AllocateIndexed(std::forward<Args>(args)...);
            auto ref      = GetPtrFromIndex(obj);
            auto refCount = AllocateAtomic();
            return RObjectRef<T>(obj, ref, refCount, this);
        }
    };

} // namespace Ifrit