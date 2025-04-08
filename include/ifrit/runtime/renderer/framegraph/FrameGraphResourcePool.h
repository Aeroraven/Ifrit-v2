
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
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/runtime/base/Base.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/core/algo/Memory.h"

namespace Ifrit::Runtime
{
    struct FrameGraphBufferDesc
    {
        u32 m_Size;
        u32 m_Usage;

        struct Hash
        {
            u32 operator()(const FrameGraphBufferDesc& desc) const
            {
                u32 hash = 0;
                hash ^= std::hash<u32>()(desc.m_Size);
                hash ^= std::hash<u32>()(desc.m_Usage);
                return hash;
            }
        };

        inline bool operator==(const FrameGraphBufferDesc& other) const
        {
            return m_Size == other.m_Size && m_Usage == other.m_Usage;
        }
    };

    struct FrameGraphTextureDesc
    {
        u32                           m_Width;
        u32                           m_Height;
        u32                           m_Depth;
        Graphics::Rhi::RhiImageFormat m_Format;
        u32                           m_Usage;

        struct Hash
        {
            u32 operator()(const FrameGraphTextureDesc& desc) const
            {
                u32 hash = 0;
                hash ^= std::hash<u32>()(desc.m_Width);
                hash ^= std::hash<u32>()(desc.m_Height);
                hash ^= std::hash<u32>()(desc.m_Depth);
                hash ^= std::hash<u32>()(desc.m_Format);
                hash ^= std::hash<u32>()(desc.m_Usage);
                return hash;
            }
        };

        inline bool operator==(const FrameGraphTextureDesc& other) const
        {
            return m_Width == other.m_Width && m_Height == other.m_Height && m_Depth == other.m_Depth
                && m_Format == other.m_Format && m_Usage == other.m_Usage;
        }
    };

    struct FrameGraphManagedBuffer
    {
        FrameGraphBufferDesc        m_Desc;
        Graphics::Rhi::RhiBufferRef m_Buffer;
        u32                         m_AutoReleaseLifetime = 0;
        bool                        m_Active              = false;
        RIndexedPtr                 m_PooledResId;

        inline bool                 CompatibleWithDesc(const FrameGraphBufferDesc& desc) const
        {
            return m_Desc.m_Size == desc.m_Size && m_Desc.m_Usage == desc.m_Usage;
        }
    };

    struct FrameGraphManagedTexture
    {
        FrameGraphTextureDesc        m_Desc;
        Graphics::Rhi::RhiTextureRef m_Texture;
        u32                          m_AutoReleaseLifetime = 0;
        bool                         m_Active              = false;
        RIndexedPtr                  m_PooledResId;

        inline bool                  CompatibleWithDesc(const FrameGraphTextureDesc& desc) const
        {
            return m_Desc.m_Width == desc.m_Width && m_Desc.m_Height == desc.m_Height && m_Desc.m_Depth == desc.m_Depth
                && m_Desc.m_Format == desc.m_Format && m_Desc.m_Usage == desc.m_Usage;
        }
    };

    using FGManagedTextureRef = Graphics::Rhi::RhiTexture*;
    using FGManagedBufferRef  = Graphics::Rhi::RhiBuffer*;

    struct FrameGraphPoolTexAllocResult
    {
        FGManagedTextureRef m_Texture = nullptr;
        RIndexedPtr         m_PooledResId;
    };

    struct FrameGraphPoolBufAllocResult
    {
        FGManagedBufferRef m_Buffer = nullptr;
        RIndexedPtr        m_PooledResId;
    };

    class IFRIT_RUNTIME_API FrameGraphResourcePool
    {
    private:
        Graphics::Rhi::RhiBackend*                                                            m_Rhi = nullptr;

        RObjectPool<FrameGraphManagedBuffer>                                                  m_BufferPool;
        RObjectPool<FrameGraphManagedTexture>                                                 m_TexturePool;

        Vec<RIndexedPtr>                                                                      m_ManagedBuffers;
        Vec<RIndexedPtr>                                                                      m_ManagedTextures;

        CustomHashMap<FrameGraphBufferDesc, Queue<RIndexedPtr>, FrameGraphBufferDesc::Hash>   m_AvailableBuffers;
        CustomHashMap<FrameGraphTextureDesc, Queue<RIndexedPtr>, FrameGraphTextureDesc::Hash> m_AvailableTextures;

    public:
        FrameGraphResourcePool(Graphics::Rhi::RhiBackend* rhi);
        ~FrameGraphResourcePool();

        FrameGraphPoolBufAllocResult CreateBuffer(const FrameGraphBufferDesc& desc);
        FrameGraphPoolTexAllocResult CreateTexture(const FrameGraphTextureDesc& desc);

        void                         ReleaseBuffer(RIndexedPtr buffer);
        void                         ReleaseTexture(RIndexedPtr texture);
    };
} // namespace Ifrit::Runtime