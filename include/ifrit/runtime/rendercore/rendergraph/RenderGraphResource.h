#pragma once
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Base.h"
#include "ifrit/core/algo/Memory.h"

namespace Ifrit::Runtime::RDG
{
    struct RDGBufferDesc
    {
        u32 m_Size;
        u32 m_Usage;

        struct Hash
        {
            u32 operator()(const RDGBufferDesc& desc) const
            {
                u32 hash = 0;
                hash ^= std::hash<u32>()(desc.m_Size);
                hash ^= std::hash<u32>()(desc.m_Usage);
                return hash;
            }
        };

        inline bool operator==(const RDGBufferDesc& other) const
        {
            return m_Size == other.m_Size && m_Usage == other.m_Usage;
        }

        RDGBufferDesc(u32 size, u32 usage) : m_Size(size), m_Usage(usage) {}
        RDGBufferDesc() : m_Size(0), m_Usage(0) {}
    };

    struct RDGTextureDesc
    {
        u32                 m_Width;
        u32                 m_Height;
        u32                 m_Depth;
        RHI::RhiImageFormat m_Format;
        u32                 m_Usage;

        struct Hash
        {
            u32 operator()(const RDGTextureDesc& desc) const
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

        inline bool operator==(const RDGTextureDesc& other) const
        {
            return m_Width == other.m_Width && m_Height == other.m_Height && m_Depth == other.m_Depth
                && m_Format == other.m_Format && m_Usage == other.m_Usage;
        }

        RDGTextureDesc(u32 width, u32 height, u32 depth, RHI::RhiImageFormat format, u32 usage)
            : m_Width(width), m_Height(height), m_Depth(depth), m_Format(format), m_Usage(usage)
        {
        }
        RDGTextureDesc()
            : m_Width(0), m_Height(0), m_Depth(0), m_Format(RHI::RhiImageFormat::RhiImgFmt_UNDEFINED), m_Usage(0)
        {
        }
    };

    struct RDGManagedBuffer
    {
        RDGBufferDesc     m_Desc;
        RHI::RhiBufferRef m_Buffer;
        u32               m_AutoReleaseLifetime = 0;
        bool              m_Active              = false;
        FIndexedPtr       m_PooledResId;

        inline bool       CompatibleWithDesc(const RDGBufferDesc& desc) const
        {
            return m_Desc.m_Size == desc.m_Size && m_Desc.m_Usage == desc.m_Usage;
        }
    };

    struct RDGManagedTexture
    {
        RDGTextureDesc     m_Desc;
        RHI::RhiTextureRef m_Texture;
        u32                m_AutoReleaseLifetime = 0;
        bool               m_Active              = false;
        FIndexedPtr        m_PooledResId;

        inline bool        CompatibleWithDesc(const RDGTextureDesc& desc) const
        {
            return m_Desc.m_Width == desc.m_Width && m_Desc.m_Height == desc.m_Height && m_Desc.m_Depth == desc.m_Depth
                && m_Desc.m_Format == desc.m_Format && m_Desc.m_Usage == desc.m_Usage;
        }
    };

    using FGManagedTextureRef = RHI::RhiTexture*;
    using FGManagedBufferRef  = RHI::RhiBuffer*;

    struct RDGPoolTexAllocResult
    {
        FGManagedTextureRef m_Texture = nullptr;
        FIndexedPtr         m_PooledResId;
    };

    struct RDGPoolBufAllocResult
    {
        FGManagedBufferRef m_Buffer = nullptr;
        FIndexedPtr        m_PooledResId;
    };

    class IFRIT_RUNTIME_API RDGResourcePool
    {
    public:
        RDGResourcePool(RHI::RhiBackend* rhi);
        ~RDGResourcePool();

        RDGPoolBufAllocResult CreateBuffer(const RDGBufferDesc& desc, const String& name);
        RDGPoolTexAllocResult CreateTexture(const RDGTextureDesc& desc, const String& name);

        void                  ReleaseBuffer(FIndexedPtr buffer);
        void                  ReleaseTexture(FIndexedPtr texture);

    private:
        RHI::RhiBackend*                                                      m_Rhi = nullptr;

        TObjectPool<RDGManagedBuffer>                                         m_BufferPool;
        TObjectPool<RDGManagedTexture>                                        m_TexturePool;

        Vec<FIndexedPtr>                                                      m_ManagedBuffers;
        Vec<FIndexedPtr>                                                      m_ManagedTextures;

        CustomHashMap<RDGBufferDesc, Queue<FIndexedPtr>, RDGBufferDesc::Hash> m_AvailableBuffers;
        CustomHashMap<RDGTextureDesc, Queue<FIndexedPtr>, RDGTextureDesc::Hash> m_AvailableTextures;
    };
} // namespace Ifrit::Runtime::RDG