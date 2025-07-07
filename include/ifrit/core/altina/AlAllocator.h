#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/hal/HalMemory.h"
#include "ifrit/core/altina/AlBase.h"
namespace Ifrit::Altina
{
    template <typename T> class TAlAllocator
    {
    public:
        IF_CONSTEXPR static AlPtrSizeTp kDefaultAlignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        IF_CONSTEXPR static AlPtrSizeTp kAlignment        = alignof(T);

        TAlAllocator() = default;

        void Deallocate(T* p, AlPtrSizeTp) noexcept { HAL::MemFree(p); }

        T*   Allocate(AlPtrSizeTp n)
        {
            if (n == 0) IF_UNLIKELY
            {
                return nullptr;
            }

            if IF_CONSTEXPR (kAlignment > kDefaultAlignment)
            {
                return static_cast<T*>(HAL::MemAllocAligned(n * sizeof(T), kAlignment));
            }
            return static_cast<T*>(HAL::MemAllocAligned(n * sizeof(T), kDefaultAlignment));
        }

        template <typename U, typename... Args> inline void Construct(U* p, Args&&... args)
        {
            new (p) U(std::forward<Args>(args)...);
        }

        template <typename U> inline void Destroy(U* p) { p->~U(); }
    };
} // namespace Ifrit::Altina