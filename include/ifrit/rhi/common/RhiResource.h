#pragma once

#include "ifrit/rhi/common/RhiBaseTypes.h"
#include "ifrit/rhi/common/RhiDevice.h"
#include "ifrit/rhi/common/RhiApi.h"
#include "ifrit/core/base/containers/Atomic.h"

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

        virtual u32           GetRefCount() const { return mRefCount.load(); }

    protected:
        inline void              SetState(ERhiResourceState state) { mState = state; }
        inline ERhiResourceState GetState() const { return mState; }

    private:
        TAtomic<u32>      mRefCount = 0;
        String            mDebugName;
        ERhiResourceState mState       = ERhiResourceState::Undefined;
        ERhiResourceType  mType        = ERhiResourceType::Unknown;
        bool              mIsUnmanaged = false; // unmanaged resources are EXTERNAL resources
    };

    // ===== Memory =====
    namespace ERhiMemoryFlagBits
    {
        enum Enum : u32
        {
            None      = 0,
            CPUAccess = 1 << 0,
            GPUAccess = 1 << 1,
        };
    } // namespace ERhiMemoryFlagBits
    using ERhiMemoryFlags = u32;

    struct RhiDeviceMemoryDesc
    {
        u64             mSize      = 0;
        u64             mAlignment = 0;
        ERhiMemoryFlags mFlags     = ERhiMemoryFlagBits::None;
    };

    class IFRIT_RHI_API RhiDeviceMemory : public RhiDeviceResource
    {
    public:
        RhiDeviceMemory(const RhiDeviceMemoryDesc& inDesc) : RhiDeviceResource(ERhiResourceType::Buffer), mDesc(inDesc)
        {
        }
        virtual ~RhiDeviceMemory()                           = default;
        virtual RhiRawHandle GetRawHandle_Allocation() const = 0;

    private:
        RhiDeviceMemoryDesc mDesc;
    };

    // ===== Memory Desc =====

    struct RhiDeviceMemoryPtr
    {
        RhiDeviceMemoryRef mMemory = nullptr;
        u64                mOffset = 0;
    };

    // ===== Buffers =====

    enum class ERhiBufferMapType
    {
        CPUReadOnly,
        CPUWriteOnly,
    };

    struct RhiBufferDesc
    {
        String             mName   = "";
        u64                mSize   = 0;
        u32                mStride = 0;
        ERhiBufferUsage    mFlags  = ERhiBufferUsageFlag::None;
        RhiDeviceMemoryPtr mManualMemory{};
    };

    class IFRIT_RHI_API RhiBuffer : public RhiDeviceResource
    {
    public:
        RhiBuffer(const RhiBufferDesc& inDesc) : RhiDeviceResource(ERhiResourceType::Buffer), mDesc(inDesc) {}
        virtual ~RhiBuffer() = default;

        virtual RhiDeviceAddr GetDeviceAddress() const = 0;
        virtual RhiRawHandle  GetRawHandle() const     = 0;

        friend class RhiCommandListContext;

    protected:
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
        ERhiImageUsage        mUsage        = ERhiImageUsageFlag::None;
        ERhiImageDimension    mDimension    = ERhiImageDimension::Unknown;
        ERhiImageFormat       mFormat       = ERhiImageFormat::Undefined;
        ERhiResourceState     mInitialState = ERhiResourceState::Undefined;
        RhiClearColorValue    mClearValue;
        u32                   mWidth     = 0;
        u32                   mHeight    = 0;
        u32                   mDepth     = 0;
        u32                   mMips      = 1;
        u32                   mSamples   = 1;
        u32                   mArraySize = 1;

        RhiDeviceMemoryPtr    mManualMemory{};

        static RhiTextureDesc CreateTexture2D(
            u32 width, u32 height, ERhiImageFormat format, ERhiImageUsage usage = 0, u32 mipLevels = 1)
        {
            RhiTextureDesc desc;
            desc.mDimension = ERhiImageDimension::Texture2D;
            desc.mWidth     = width;
            desc.mHeight    = height;
            desc.mDepth     = 1;
            desc.mMips      = mipLevels;
            desc.mFormat    = format;
            desc.mUsage     = usage;
            return desc;
        }
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
        IF_NODISCARD inline ERhiResourceState   GetInitialState() const noexcept { return mDesc.mInitialState; }
        IF_NODISCARD inline RhiImageSubResource GetFullSubresource() const noexcept
        {
            RhiImageSubResource subRes;
            subRes.mipLevel   = 0;
            subRes.arrayLayer = 0;
            subRes.mipCount   = mDesc.mMips;
            subRes.layerCount = mDesc.mArraySize;
            return subRes;
        }

        virtual RhiRawHandle GetRawHandle() const = 0;

        friend class RhiCommandListContext;

    protected:
        RhiTextureDesc mDesc;
        bool           mRhiSwapchainImage = false;
    };

    // ===== Samplers =====

    struct RhiSamplerDesc
    {
        ERhiSamplerFilter   mFilterMode    = ERhiSamplerFilter::Linear;
        ERhiSamplerWrapMode mWrapModeU     = ERhiSamplerWrapMode::Repeat;
        ERhiSamplerWrapMode mWrapModeV     = ERhiSamplerWrapMode::Repeat;
        ERhiSamplerWrapMode mWrapModeW     = ERhiSamplerWrapMode::Repeat;
        f32                 mLodBias       = 0.0f;
        f32                 mMinLod        = 0.0f;
        f32                 mMaxLod        = FLT_MAX;
        u32                 mMaxAnisotropy = 1;
    };

    class IFRIT_RHI_API RhiSampler : public RhiDeviceResource
    {
    public:
        RhiSampler(const RhiSamplerDesc& inDesc) : RhiDeviceResource(ERhiResourceType::SamplerState), mDesc(inDesc) {}
        virtual ~RhiSampler()                     = default;
        virtual RhiRawHandle GetRawHandle() const = 0;

    protected:
        RhiSamplerDesc mDesc;
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
        RhiResourceViewDesc() {};

        struct BufferViewDesc
        {
            u32 mOffset = 0;
            u32 mSize   = ~0u;
        };

        struct TextureViewDesc
        {
            RhiImageSubResource mSubResource;
        };

        ERhiResourceViewedType mType;

        BufferViewDesc         mBufferView;
        TextureViewDesc        mTextureView;
    };

    class IFRIT_RHI_API RhiResourceView : public RhiDeviceResource
    {
    public:
        RhiResourceView(RhiTexture* texture, const RhiResourceViewDesc& desc)
            : RhiDeviceResource(ERhiResourceType::View), mTexture(texture), mDesc(desc)

        {
            InternalCheck();
        }

        RhiResourceView(RhiBuffer* buffer, const RhiResourceViewDesc& desc)
            : RhiDeviceResource(ERhiResourceType::View), mBuffer(buffer), mDesc(desc)
        {
            InternalCheck();
        }

        RhiTexture*                       GetUnderlyingTexture() const;
        RhiBuffer*                        GetUnderlyingBuffer() const;

        virtual void                      AcquireHandle() = 0;
        virtual void                      ReleaseHandle() = 0;

        bool                              IsTextureView() const { return mTexture != nullptr; }
        bool                              IsBufferView() const { return mBuffer != nullptr; }

        inline RhiDescriptorHandle        GetHandle() const { return mHandle; }
        inline const RhiResourceViewDesc& GetDesc() const { return mDesc; }

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
        using RhiResourceView::RhiResourceView;
        virtual void AcquireHandle() = 0;
        virtual void ReleaseHandle() = 0;
    };

    class IFRIT_RHI_API RhiUnorderedAccessView : public RhiResourceView
    {
    public:
        using RhiResourceView::RhiResourceView;
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