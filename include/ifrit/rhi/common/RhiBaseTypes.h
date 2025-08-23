
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

#include "RhiForwardingTypes.h"
#include "ifrit/core/platform/ApiConv.h"

namespace Ifrit::RHI
{
    enum class ERhiTypeFlags : u8
    {
        Float32 = 0x01,
        Float64 = 0x02,
        Int8    = 0x03,
        Int16   = 0x04,
        Int32   = 0x05,
        Int64   = 0x06,
        UInt8   = 0x07,
        UInt16  = 0x08,
        UInt32  = 0x09,
        UInt64  = 0x0A,
    };

    struct RhiAttachmentBlendInfo
    {
        bool            mBlendEnable         = false;
        ERhiBlendFactor mSrcColorBlendFactor = ERhiBlendFactor::One;
        ERhiBlendFactor mDstColorBlendFactor = ERhiBlendFactor::Zero;
        ERhiBlendOp     mColorBlendOp        = ERhiBlendOp::Add;
        ERhiBlendFactor mSrcAlphaBlendFactor = ERhiBlendFactor::One;
        ERhiBlendFactor mDstAlphaBlendFactor = ERhiBlendFactor::Zero;
        ERhiBlendOp     mAlphaBlendOp        = ERhiBlendOp::Add;
    };

    struct RhiClearColorValue
    {
        RhiClearColorValue() = default;
        ERhiTypeFlags m_Type;
        union
        {
            f32 m_ValueF32[4];
            u32 m_ValueU32[4];
            i32 m_ValueI32[4];
        };

        RhiClearColorValue(const RhiClearColorValue& other) : m_Type(other.m_Type)
        {
            if (m_Type == ERhiTypeFlags::Float32)
            {
                m_ValueF32[0] = other.m_ValueF32[0];
                m_ValueF32[1] = other.m_ValueF32[1];
                m_ValueF32[2] = other.m_ValueF32[2];
                m_ValueF32[3] = other.m_ValueF32[3];
            }
            else if (m_Type == ERhiTypeFlags::UInt32)
            {
                m_ValueU32[0] = other.m_ValueU32[0];
                m_ValueU32[1] = other.m_ValueU32[1];
                m_ValueU32[2] = other.m_ValueU32[2];
                m_ValueU32[3] = other.m_ValueU32[3];
            }
            else // if (m_Type == RhiTypeFlags::Int32)
            {
                m_ValueI32[0] = other.m_ValueI32[0];
                m_ValueI32[1] = other.m_ValueI32[1];
                m_ValueI32[2] = other.m_ValueI32[2];
                m_ValueI32[3] = other.m_ValueI32[3];
            }
        }

        RhiClearColorValue& operator=(const RhiClearColorValue& other)
        {
            if (this != &other)
            {
                m_Type = other.m_Type;
                if (m_Type == ERhiTypeFlags::Float32)
                {
                    m_ValueF32[0] = other.m_ValueF32[0];
                    m_ValueF32[1] = other.m_ValueF32[1];
                    m_ValueF32[2] = other.m_ValueF32[2];
                    m_ValueF32[3] = other.m_ValueF32[3];
                }
                else if (m_Type == ERhiTypeFlags::UInt32)
                {
                    m_ValueU32[0] = other.m_ValueU32[0];
                    m_ValueU32[1] = other.m_ValueU32[1];
                    m_ValueU32[2] = other.m_ValueU32[2];
                    m_ValueU32[3] = other.m_ValueU32[3];
                }
                else // if (m_Type == RhiTypeFlags::Int32)
                {
                    m_ValueI32[0] = other.m_ValueI32[0];
                    m_ValueI32[1] = other.m_ValueI32[1];
                    m_ValueI32[2] = other.m_ValueI32[2];
                    m_ValueI32[3] = other.m_ValueI32[3];
                }
            }
            return *this;
        }
    };

    struct RhiClearDepthStencilValue
    {
        f32 m_Depth;
        u32 m_Stencil;
    };

    enum class ERhiClearValueType : u8
    {
        Color        = 0x01,
        DepthStencil = 0x02,
    };

    struct RhiClearValue2
    {
        ERhiClearValueType m_Type;
        union
        {
            RhiClearColorValue        m_Color;
            RhiClearDepthStencilValue m_DepthStencil;
        };

        RhiClearValue2() = default;
        RhiClearValue2(const RhiClearColorValue& color) : m_Type(ERhiClearValueType::Color), m_Color(color) {}
        RhiClearValue2(const RhiClearDepthStencilValue& depthStencil)
            : m_Type(ERhiClearValueType::DepthStencil), m_DepthStencil(depthStencil)
        {
        }
        RhiClearValue2(const RhiClearValue2& other) : m_Type(other.m_Type)
        {
            if (m_Type == ERhiClearValueType::Color)
            {
                m_Color = other.m_Color;
            }
            else // if (m_Type == ERhiClearValueType::DepthStencil)
            {
                m_DepthStencil = other.m_DepthStencil;
            }
        }

        RhiClearValue2& operator=(const RhiClearValue2& other)
        {
            if (this != &other)
            {
                m_Type = other.m_Type;
                if (m_Type == ERhiClearValueType::Color)
                {
                    m_Color = other.m_Color;
                }
                else // if (m_Type == ERhiClearValueType::DepthStencil)
                {
                    m_DepthStencil = other.m_DepthStencil;
                }
            }
            return *this;
        }
    };

    struct RhiViewport
    {
        f32 x;
        f32 y;
        f32 width;
        f32 height;
        f32 minDepth;
        f32 maxDepth;
    };

    struct RhiScissor
    {
        int32_t x;
        int32_t y;
        u32     width;
        u32     height;
    };

    struct RhiImageSubResource
    {
        u32 mipLevel;
        u32 arrayLayer;
        u32 mipCount   = 1;
        u32 layerCount = 1;
    };

    // Update 250326: This is a deprecated struct, the bindless descriptor index is disentangled with
    // the resource itself, causing the "Dangling Descriptor" issue (after the resource is destroyed)
    // Now for each resource, we maintain a descriptor handle with type and index

    // Update 250822: Removed
    // struct RhiDescHandleLegacy
    // {
    //     u32         activeFrame;
    //     Vec<u32>    ids;
    //     inline u32  GetActiveId() const { return ids[activeFrame]; }
    //     inline void SetFromId(u32 frame) { activeFrame = frame; }
    // };

    enum class ERhiDescriptorHeapType : u32
    {
        UniformBuffer,
        StorageBuffer,
        ReadOnlyStorageBuffer,
        CombinedImageSampler,
        StorageImage,
        SampledImage,
        Sampler,
        Invalid
    };

    enum class ERhiDescriptorHandleType : u32
    {
        Trivial,
        Bindless,
    };

    struct RhiDescriptorHandle
    {
        ERhiDescriptorHeapType   mType = ERhiDescriptorHeapType::Invalid;
        u32                      mIndex;
        ERhiDescriptorHandleType mHandleType = ERhiDescriptorHandleType::Bindless;

        RhiDescriptorHandle() = default;
        RhiDescriptorHandle(ERhiDescriptorHeapType type, u32 index) : mType(type), mIndex(index) {}
        inline ERhiDescriptorHeapType GetType() const { return mType; }
        inline u32                    GetId() const { return mIndex; }
    };

} // namespace Ifrit::RHI