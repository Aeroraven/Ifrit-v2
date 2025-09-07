#include "ifrit/runtime/rendercore/rendergraph/RenderGraphResource.h"

namespace Ifrit::Runtime::RDG
{
    RDGResourcePool::RDGResourcePool(RHI::RhiBackend* rhi) : m_Rhi(rhi) {}

    IFRIT_APIDECL RDGPoolBufAllocResult RDGResourcePool::CreateBuffer(const RDGBufferDesc& desc, const String& name)
    {
        auto&                 hset = m_AvailableBuffers[desc];
        RDGPoolBufAllocResult alloc;
        auto                  debugName = "RDGManaged." + name;
        if (hset.empty())
        {
            auto buffer                = m_Rhi->CreateBuffer(debugName, desc.m_Size, desc.m_Usage, false, true);
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
            ptr->m_Buffer->SetDebugName(debugName);
        }
        return alloc;
    }

    IFRIT_APIDECL RDGPoolTexAllocResult RDGResourcePool::CreateTexture(const RDGTextureDesc& desc, const String& name)
    {
        auto&                 hset = m_AvailableTextures[desc];
        RDGPoolTexAllocResult alloc;
        auto                  debugName = "RDGManaged." + name;
        if (hset.empty())
        {
            auto isStorage = (desc.m_Usage & RHI::RhiImgUsage_UnorderedAccess) != 0;
            auto texture   = m_Rhi->CreateTexture3D(
                debugName, desc.m_Width, desc.m_Height, desc.m_Depth, desc.m_Format, desc.m_Usage, isStorage);

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

            ptr->m_Texture->SetDebugName(debugName);
        }
        return alloc;
    }

    IFRIT_APIDECL void RDGResourcePool::ReleaseBuffer(FIndexedPtr id)
    {
        auto ptr                   = m_BufferPool.GetPtrFromIndex(id);
        ptr->m_Active              = false;
        ptr->m_AutoReleaseLifetime = 0;
        m_AvailableBuffers[ptr->m_Desc].push(id);
    }

    IFRIT_APIDECL void RDGResourcePool::ReleaseTexture(FIndexedPtr id)
    {
        auto ptr                   = m_TexturePool.GetPtrFromIndex(id);
        ptr->m_Active              = false;
        ptr->m_AutoReleaseLifetime = 0;
        m_AvailableTextures[ptr->m_Desc].push(id);
    }

    IFRIT_APIDECL RDGResourcePool::~RDGResourcePool()
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

} // namespace Ifrit::Runtime::RDG