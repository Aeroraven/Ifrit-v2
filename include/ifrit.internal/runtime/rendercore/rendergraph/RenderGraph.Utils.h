#pragma once
#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"

namespace Ifrit::Runtime::RenderCore::RDG
{
    // ===== Utils =====
    IF_NODISCARD constexpr IF_FORCEINLINE u32 GetMinPassIndex(u32 a, u32 b) noexcept
    {
        if (a == ~0u)
            return b;
        if (b == ~0u)
            return a;
        return std::min(a, b);
    }

    IF_NODISCARD constexpr IF_FORCEINLINE u32 GetMaxPassIndex(u32 a, u32 b) noexcept
    {
        if (a == ~0u)
            return b;
        if (b == ~0u)
            return a;
        return std::max(a, b);
    }

    IF_FORCEINLINE bool IsStateInUAV(RHI::ERhiResourceState state) noexcept
    {
        return state == RHI::ERhiResourceState::UnorderedAccess || state == RHI::ERhiResourceState::UnorderedAccess_Read
            || state == RHI::ERhiResourceState::UnorderedAccess_Write;
    }

} // namespace Ifrit::Runtime::RenderCore::RDG