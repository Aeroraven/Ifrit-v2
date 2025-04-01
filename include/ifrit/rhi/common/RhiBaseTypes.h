
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

namespace Ifrit::Graphics::Rhi
{
    struct RhiAttachmentBlendInfo
    {
        bool           m_blendEnable         = false;
        RhiBlendFactor m_srcColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE;
        RhiBlendFactor m_dstColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ZERO;
        RhiBlendOp     m_colorBlendOp        = RhiBlendOp::RHI_BLEND_OP_ADD;
        RhiBlendFactor m_srcAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE;
        RhiBlendFactor m_dstAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ZERO;
        RhiBlendOp     m_alphaBlendOp        = RhiBlendOp::RHI_BLEND_OP_ADD;
    };

    struct RhiClearValue
    {
        f32 m_color[4];
        f32 m_depth;
        u32 m_stencil;
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
    struct RhiDescHandleLegacy
    {
        u32         activeFrame;
        Vec<u32>    ids;
        inline u32  GetActiveId() const { return ids[activeFrame]; }
        inline void SetFromId(u32 frame) { activeFrame = frame; }
    };

    enum class RhiDescriptorHeapType : u32
    {
        UniformBuffer,
        StorageBuffer,
        CombinedImageSampler,
        StorageImage,
        Invalid
    };

    struct RhiDescriptorHandle
    {
        RhiDescriptorHeapType m_type;
        u32                   m_index;

        RhiDescriptorHandle(RhiDescriptorHeapType type, u32 index) : m_type(type), m_index(index) {}
        inline RhiDescriptorHeapType GetType() const { return m_type; }
        inline u32                   GetId() const { return m_index; }
    };

} // namespace Ifrit::Graphics::Rhi