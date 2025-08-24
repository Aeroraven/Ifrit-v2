#pragma once

#include "ifrit/rhi/common/RhiBaseTypes.h"
#include "ifrit/rhi/common/RhiDevice.h"
#include "ifrit/rhi/common/RhiApi.h"
#include <queue>
#include <cstddef>

namespace Ifrit::RHI
{

    // =====  Resources =====
    class IFRIT_RHI_API RhiDeviceResource : public RhiDeviceChild
    {
    public:
        RhiDeviceResource(ERhiResourceType resType) : mType(resType) {}

        virtual ~RhiDeviceResource() {}

        inline virtual void AddRef() { mRefCount.fetch_add(1); }
        inline virtual void Release()
        {
            if (mRefCount.fetch_sub(1) == 1)
            {
                if (!mIsUnmanaged)
                {
                    MarkForDelete();
                }
            }
        }
        IF_FORCEINLINE virtual void MarkForDelete()
        {
            GetDevice()->GetResourceDeleteQueue()->AddResourceToDeleteQueue(this);
        }
        virtual void          SetDebugName(const String& name) { mDebugName = name; }
        virtual const String& GetDebugName() const { return mDebugName; }

    protected:
        inline void              SetState(ERhiResourceState state) { mState = state; }
        inline ERhiResourceState GetState() const { return mState; }

    private:
        Atomic<u32>       mRefCount = 0;
        String            mDebugName;
        ERhiResourceState mState       = ERhiResourceState::Undefined;
        ERhiResourceType  mType        = ERhiResourceType::Unknown;
        bool              mIsUnmanaged = false; // unmanaged resources are EXTERNAL resources
    };

    // ===== Buffers =====

    struct RhiBufferDesc
    {
        String          mName   = "";
        u64             mSize   = 0;
        u32             mStride = 0;
        ERhiBufferUsage mFlags  = ERhiBufferUsageFlag::None;
    };

    class IFRIT_RHI_API RhiBuffer : public RhiDeviceResource
    {
    public:
        RhiBuffer(const RhiBufferDesc& inDesc) : RhiDeviceResource(ERhiResourceType::Buffer), mDesc(inDesc) {}
        virtual ~RhiBuffer() = default;

        virtual void          MapMemory()                                         = 0;
        virtual void          UnmapMemory()                                       = 0;
        virtual void          FlushBuffer()                                       = 0;
        virtual void          ReadBuffer(void* data, u32 size, u32 offset)        = 0;
        virtual void          WriteBuffer(const void* data, u32 size, u32 offset) = 0;

        virtual RhiDeviceAddr GetDeviceAddress() const = 0;

        friend class RhiCommandListContext;

    private:
        RhiBufferDesc mDesc;
    };

    class IFRIT_RHI_API RhiStagedSingleBuffer : public RhiDeviceChild
    {
    public:
        virtual ~RhiStagedSingleBuffer() = default;
        virtual void CmdCopyToDevice(const RhiCommandListContext* cmd, const void* data, u32 size, u32 localOffset) = 0;
    };

    // ===== Textures =====

    struct RhiTextureDesc
    {
        ERhiImageUsage     mUsage        = ERhiImageUsageFlag::None;
        ERhiImageDimension mDimension    = ERhiImageDimension::Unknown;
        ERhiImageFormat    mFormat       = ERhiImageFormat::Undefined;
        ERhiResourceState  mInitialState = ERhiResourceState::Undefined;
        u32                mWidth        = 0;
        u32                mHeight       = 0;
        u32                mDepth        = 0;
        u32                mMips         = 1;
        u32                mSamples      = 1;
        u32                mArraySize    = 1;
    };

    class IFRIT_RHI_API RhiTexture : public RhiDeviceResource
    {
    public:
        RhiTexture(const RhiTextureDesc& inDesc) : RhiDeviceResource(ERhiResourceType::Texture), mDesc(inDesc) {}
        virtual ~RhiTexture() = default;

        inline u32                GetHeight() const { return mDesc.mHeight; }
        inline u32                GetWidth() const { return mDesc.mWidth; }
        inline u32                GetDepth() const { return mDesc.mDepth; }

        inline u32                GetMipLevels() const { return mDesc.mMips; }
        inline u32                GetArraySize() const { return mDesc.mArraySize; }
        inline ERhiImageDimension GetDimension() const { return mDesc.mDimension; }
        inline bool               IsSwapchainImage() const { return mRhiSwapchainImage; }
        inline bool               IsDepthTexture() const { return (mDesc.mUsage & ERhiImageUsageFlag::Depth) != 0; }
        inline u32                GetSamples() const { return mDesc.mSamples; }
        inline ERhiImageUsage     GetUsage() const { return mDesc.mUsage; }

        virtual RhiRawHandle      GetRawHandle() const = 0;

        friend class RhiCommandListContext;

    protected:
        RhiTextureDesc mDesc;
        bool           mRhiSwapchainImage = false;
    };

    // ===== Samplers =====
    class IFRIT_RHI_API RhiSampler : public RhiDeviceResource
    {
    protected:
        RhiSampler() : RhiDeviceResource(ERhiResourceType::SamplerState) {}
        virtual ~RhiSampler() = default;

    public:
        virtual RhiRawHandle GetRawHandle() const = 0;
    };

    // ===== Raytracing =====

    struct IFRIT_RHI_API RhiRTGeometryReference
    {
        RhiDeviceAddr mVertex;
        RhiDeviceAddr mIndex;
        RhiDeviceAddr mTransform;
        u32           mNumVertices;
        u32           mNumIndices;
        u32           mVertexComponents = 3;
        u32           mVertexStride     = 12;
    };

    // ===== Resource View =====
    enum class ERhiResourceViewedType
    {
        Buffer,
        Texture,
    };

    struct IFRIT_RHI_API RhiResourceViewDesc
    {
        struct BufferViewDesc
        {
            u32 mOffset = 0;
            u32 mSize   = 0;
        };

        struct TextureViewDesc
        {
            RhiImageSubResource mSubResource;
        };

        ERhiResourceViewedType mType;
        union
        {
            BufferViewDesc  mBufferView;
            TextureViewDesc mTextureView;
        };
    };

    class IFRIT_RHI_API RhiResourceView : public RhiDeviceResource
    {
    public:
        RhiResourceView(RhiTexture* texture, const RhiResourceViewDesc& desc)
            : RhiDeviceResource(ERhiResourceType::View), mTexture(texture), mDesc(desc)

        {
        }

        RhiResourceView(RhiBuffer* buffer, const RhiResourceViewDesc& desc)
            : RhiDeviceResource(ERhiResourceType::View), mBuffer(buffer), mDesc(desc)
        {
        }

        RhiTexture*                GetUnderlyingTexture() const;
        RhiBuffer*                 GetUnderlyingBuffer() const;

        virtual void               AcquireHandle() = 0;
        virtual void               ReleaseHandle() = 0;

        inline RhiDescriptorHandle GetHandle() const { return mHandle; }

        virtual ~RhiResourceView()                = default;
        virtual RhiRawHandle GetRawHandle() const = 0;

    protected:
        RhiTexture*         mTexture = nullptr;
        RhiBuffer*          mBuffer  = nullptr;
        RhiResourceViewDesc mDesc;
        RhiDescriptorHandle mHandle;

    private:
        void InternalCheck();
    };

    class IFRIT_RHI_API RhiShaderReadView : public RhiResourceView
    {
    public:
        virtual void AcquireHandle() = 0;
        virtual void ReleaseHandle() = 0;
    };

    class IFRIT_RHI_API RhiUnorderedAccessView : public RhiResourceView
    {
    public:
        virtual void AcquireHandle() = 0;
        virtual void ReleaseHandle() = 0;
    };

    // ===== Raytracing =====
    class IFRIT_RHI_API RhiRTInstance
    {
    public:
        virtual RhiDeviceAddr GetDeviceAddress() const = 0;
    };

    class IFRIT_RHI_API RhiRTScene
    {
    public:
        virtual RhiDeviceAddr GetDeviceAddress() const = 0;
    };

} // namespace Ifrit::RHI