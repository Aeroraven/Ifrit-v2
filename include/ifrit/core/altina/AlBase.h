#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/base/CoreBase.h"

namespace Ifrit::Altina
{
    using AlContainerSizeTp = u32;
    using AlPtrSizeTp       = usize;

    inline IF_CONSTEXPR bool kAlDebugMode = false;

    IFRIT_CORE_API void      AlErrorImpl(const char* message);

    inline void              AlError(const char* message)
    {
        if IF_CONSTEXPR (kAlDebugMode)
        {
            AlErrorImpl(message);
            std::abort();
        }
    }

    inline void AlAssert(bool condition, const char* message)
    {
        if IF_CONSTEXPR (kAlDebugMode)
        {
            if (!condition)
            {
                AlErrorImpl(message);
                std::abort();
            }
        }
    }

} // namespace Ifrit::Altina