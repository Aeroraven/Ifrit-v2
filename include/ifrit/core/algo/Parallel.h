#pragma once
#include "ifrit/core/base/IfritBase.h"
#include <algorithm>
#include <execution>
#include <functional>
#include <ranges>
namespace Ifrit
{
    template <class T> void UnorderedFor(T start, T end, std::function<void(T)> func)
    {
        auto rng = std::ranges::views::iota(start, end);
        std::for_each(std::execution::par_unseq, rng.begin(), rng.end(), [func](T x) { func(x); });
    }

    template <class T> void SequentialFor(int start, int end, std::function<void(T)> func)
    {
        for (int i = start; i < end; i++)
        {
            func(i);
        }
    }

    using FSpinLock = Atomic<i32>;
    IF_FORCEINLINE void SpinLockAcquire(FSpinLock& lock)
    {
        i32 expected = 0;
        while (lock.compare_exchange_strong(expected, 1, std::memory_order::acq_rel, std::memory_order::acquire))
        {
            expected = 0;
        }
    }
    IF_FORCEINLINE void SpinLockRelease(FSpinLock& lock) { lock.store(0, std::memory_order::release); }

    class FSpinLockGuard
    {
    private:
        FSpinLock& m_Lock;

    public:
        FSpinLockGuard(FSpinLock& lock) : m_Lock(lock) { SpinLockAcquire(m_Lock); }
        ~FSpinLockGuard() { SpinLockRelease(m_Lock); }
    };

} // namespace Ifrit