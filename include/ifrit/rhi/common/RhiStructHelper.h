#pragma once
#include "RhiBaseTypes.h"
#include "ifrit/core/math/VectorDefs.h"

namespace Ifrit::RHI
{
    // Clear value creator

    template <typename T IF_REQUIRES(std::is_same_v<T, f32> || std::is_same_v<T, u32> || std::is_same_v<T, i32>)>
    IF_FORCEINLINE RhiClearColorValue CreateRhiClearColorValue(const Vector4g<T>& v)
    {
        RhiClearColorValue value;
        if IF_CONSTEXPR (std::is_same_v<T, f32>)
        {
            value.m_Type        = RhiTypeFlags::Float32;
            value.m_ValueF32[0] = v.x;
            value.m_ValueF32[1] = v.y;
            value.m_ValueF32[2] = v.z;
            value.m_ValueF32[3] = v.w;
        }
        else if IF_CONSTEXPR (std::is_same_v<T, u32>)
        {
            value.m_Type        = RhiTypeFlags::UInt32;
            value.m_ValueU32[0] = static_cast<u32>(v.x);
            value.m_ValueU32[1] = static_cast<u32>(v.y);
            value.m_ValueU32[2] = static_cast<u32>(v.z);
            value.m_ValueU32[3] = static_cast<u32>(v.w);
        }
        else if IF_CONSTEXPR (std::is_same_v<T, i32>)
        {
            value.m_Type        = RhiTypeFlags::Int32;
            value.m_ValueI32[0] = static_cast<i32>(v.x);
            value.m_ValueI32[1] = static_cast<i32>(v.y);
            value.m_ValueI32[2] = static_cast<i32>(v.z);
            value.m_ValueI32[3] = static_cast<i32>(v.w);
        }
        return value;
    }

    IF_FORCEINLINE RhiClearColorValue CreateRhiClearColorValue(u64 clearValue)
    {
        RhiClearColorValue value;
        value.m_Type        = ERhiTypeFlags::UInt32;
        value.m_ValueU32[0] = static_cast<u32>(clearValue & 0xFFFFFFFF);
        value.m_ValueU32[1] = static_cast<u32>((clearValue >> 32) & 0xFFFFFFFF);
        value.m_ValueU32[2] = static_cast<u32>(clearValue & 0xFFFFFFFF);
        value.m_ValueU32[3] = static_cast<u32>((clearValue >> 32) & 0xFFFFFFFF);
        return value;
    }

    IF_FORCEINLINE RhiClearDepthStencilValue CreateRhiClearDepthStencilValue(f32 depth, uint32_t stencil)
    {
        RhiClearDepthStencilValue value;
        value.m_Depth   = depth;
        value.m_Stencil = stencil;
        return value;
    }

} // namespace Ifrit::RHI