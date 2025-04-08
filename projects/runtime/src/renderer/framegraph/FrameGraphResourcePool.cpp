
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

#include "ifrit/runtime/renderer/framegraph/FrameGraphResourcePool.h"

namespace Ifrit::Runtime
{
    FrameGraphResourcePool::FrameGraphResourcePool(Graphics::Rhi::RhiBackend* rhi) : m_Rhi(rhi) {}

    IFRIT_APIDECL FrameGraphPoolBufAllocResult FrameGraphResourcePool::CreateBuffer(const FrameGraphBufferDesc& desc)
    {
        auto&                        hset = m_AvailableBuffers[desc];
        FrameGraphPoolBufAllocResult alloc;
        if (hset.empty())
        {
            auto buffer                = m_Rhi->CreateBuffer("FGBuffer", desc.m_Size, desc.m_Usage, false, true);
            auto id                    = m_BufferPool.AllocateIndexed();
            auto ptr                   = m_BufferPool.GetPtrFromIndex(id);
            ptr->m_Buffer              = buffer;
            ptr->m_Desc                = desc;
            ptr->m_Active              = true;
            ptr->m_AutoReleaseLifetime = 3;
            ptr->m_PooledResId         = id;

            m_ManagedBuffers.push_back(id);
            alloc.m_Buffer      = ptr->m_Buffer.get();
            alloc.m_PooledResId = id;
        }
        else
        {
            auto id = hset.front();
            hset.pop();
            auto ptr                   = m_BufferPool.GetPtrFromIndex(id);
            ptr->m_Active              = true;
            ptr->m_AutoReleaseLifetime = 3;

            alloc.m_PooledResId = id;
            alloc.m_Buffer      = ptr->m_Buffer.get();
        }
        return alloc;
    }

    IFRIT_APIDECL FrameGraphPoolTexAllocResult FrameGraphResourcePool::CreateTexture(const FrameGraphTextureDesc& desc)
    {
        auto&                        hset = m_AvailableTextures[desc];
        FrameGraphPoolTexAllocResult alloc;
        if (hset.empty())
        {
            auto isStorage = (desc.m_Usage & Graphics::Rhi::RhiImgUsage_UnorderedAccess) != 0;
            auto texture   = m_Rhi->CreateTexture3D(
                "FGTexture", desc.m_Width, desc.m_Height, desc.m_Depth, desc.m_Format, desc.m_Usage, isStorage);

            auto id                    = m_TexturePool.AllocateIndexed();
            auto ptr                   = m_TexturePool.GetPtrFromIndex(id);
            ptr->m_Texture             = texture;
            ptr->m_Desc                = desc;
            ptr->m_Active              = true;
            ptr->m_AutoReleaseLifetime = 3;
            ptr->m_PooledResId         = id;

            m_ManagedTextures.push_back(id);
            alloc.m_Texture     = ptr->m_Texture.get();
            alloc.m_PooledResId = id;
        }
        else
        {
            auto id = hset.front();
            hset.pop();
            auto ptr = m_TexturePool.GetPtrFromIndex(id);

            alloc.m_Texture = ptr->m_Texture.get();
            ptr->m_Active   = true;

            alloc.m_PooledResId = id;
            alloc.m_Texture     = ptr->m_Texture.get();
        }
        return alloc;
    }

    IFRIT_APIDECL void FrameGraphResourcePool::ReleaseBuffer(RIndexedPtr id)
    {
        auto ptr                   = m_BufferPool.GetPtrFromIndex(id);
        ptr->m_Active              = false;
        ptr->m_AutoReleaseLifetime = 0;
        m_AvailableBuffers[ptr->m_Desc].push(id);
    }

    IFRIT_APIDECL void FrameGraphResourcePool::ReleaseTexture(RIndexedPtr id)
    {
        auto ptr                   = m_TexturePool.GetPtrFromIndex(id);
        ptr->m_Active              = false;
        ptr->m_AutoReleaseLifetime = 0;
        m_AvailableTextures[ptr->m_Desc].push(id);
    }

    IFRIT_APIDECL FrameGraphResourcePool::~FrameGraphResourcePool()
    {
        for (auto id : m_ManagedBuffers)
        {
            auto ptr = m_BufferPool.GetPtrFromIndex(id);
            m_BufferPool.DeallocateIndexed(id);
        }
        for (auto id : m_ManagedTextures)
        {
            auto ptr = m_TexturePool.GetPtrFromIndex(id);
            m_TexturePool.DeallocateIndexed(id);
        }
    }

} // namespace Ifrit::Runtime