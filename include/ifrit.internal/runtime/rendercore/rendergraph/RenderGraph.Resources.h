#pragma once
#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"

#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.GlobalVars.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/RenderGraph.Utils.h"
#include "ifrit.internal/runtime/rendercore/rendergraph/LogUtils.h"

namespace Ifrit::Runtime::RenderCore::RDG
{
    // ===== RDG Resources =====
    enum class ERDGLifetimeOverlapTestResult
    {
        Overlap,
        Before,
        After,
    };

    struct RDGResourceLifetime
    {
        Array<u32, ERDGQueueType::Count> mLastUsePassIndex;
        u32                              mFirstUsePassIndex = ~0u;

        constexpr RDGResourceLifetime() noexcept
        {
            mLastUsePassIndex[ERDGQueueType::Graphics]      = ~0u;
            mLastUsePassIndex[ERDGQueueType::AsyncCompute]  = ~0u;
            mLastUsePassIndex[ERDGQueueType::AsyncTransfer] = ~0u;
        }

        ERDGLifetimeOverlapTestResult CheckOverlappingWith(
            const RDGResourceLifetime& rhs, const Vec<Array<u32, ERDGQueueType::Count>>& dependencies) const
        {
            // Reference from: Resource Management with Frame Graph in Messiah
            bool before = true;
            bool after  = true;

            auto fnReorder = [](u32 a) -> u32 {
                if (a == ~0u)
                    return 0;
                return a + 1;
            };

            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                before &= (fnReorder(dependencies[rhs.mFirstUsePassIndex][q]) >= fnReorder(mLastUsePassIndex[q]));
                after &= (fnReorder(dependencies[mFirstUsePassIndex][q]) >= fnReorder(rhs.mLastUsePassIndex[q]));
            }
            if (!before && !after)
                return ERDGLifetimeOverlapTestResult::Overlap;
            return before ? ERDGLifetimeOverlapTestResult::Before : ERDGLifetimeOverlapTestResult::After;
        }

        void ExtendLifetime(const RDGResourceLifetime& rhs)
        {
            if (mFirstUsePassIndex == ~0u
                || (rhs.mFirstUsePassIndex != ~0u && rhs.mFirstUsePassIndex < mFirstUsePassIndex))
            {
                mFirstUsePassIndex = rhs.mFirstUsePassIndex;
            }
            for (u32 q = 0; q < ERDGQueueType::Count; ++q)
            {
                if (rhs.mLastUsePassIndex[q] != ~0u)
                {
                    if (mLastUsePassIndex[q] == ~0u || rhs.mLastUsePassIndex[q] > mLastUsePassIndex[q])
                    {
                        mLastUsePassIndex[q] = rhs.mLastUsePassIndex[q];
                    }
                }
            }
        }
    };

    struct RDGManagedResourceAllocationRequest
    {
        u32                mResourceIndex = ~0u;

        // For aliasing
        ERDGMemoryHeapType mHeapType = ERDGMemoryHeapType::DeviceLocal;
        u32                mHeapId   = ~0u;
        u64                mOffset   = ~0ull;
    };

    struct RDGResourceAccess_UAV : public IRDGAccess_ResourceView
    {
        RHI::RhiUnorderedAccessView* mUAV;
        RHI::RhiDescriptorHandle     GetDescriptor() override
        {
            RDG_NOTNULL(mUAV, "UAV is null in RDGResourceAccess_UAV");
            return mUAV->GetHandle();
        }
    };
    struct RDGResourceAccess_SRV : public IRDGAccess_ResourceView
    {
        RHI::RhiShaderReadView*  mSRV;
        RHI::RhiDescriptorHandle GetDescriptor() override
        {
            RDG_NOTNULL(mSRV, "SRV is null in RDGResourceAccess_SRV");
            return mSRV->GetHandle();
        }
    };
    struct RDGResourceAccess_Texture : public IRDGAccess_Texture
    {
        RHI::RhiTexture* mTex;
        RHI::RhiTexture* GetTexture() override
        {
            RDG_NOTNULL(mTex, "Texture is null in RDGResourceAccess_Texture");
            return mTex;
        }
    };
    struct RDGResourceAccess_Buffer : public IRDGAccess_Buffer
    {
        RHI::RhiBuffer* mBuf;
        RHI::RhiBuffer* GetBuffer() override
        {
            RDG_NOTNULL(mBuf, "Buffer is null in RDGResourceAccess_Buffer");
            return mBuf;
        }
    };

    class RDGResource
    {
    public:
        RDGResource(RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags) noexcept
            : mBuilder(builder), mName(name), mFlags(flags)
        {
        }
        virtual ~RDGResource() = default;

        IF_FORCEINLINE RDGResourceLifetime& GetLifetime() noexcept { return mLifetime; }
        IF_FORCEINLINE bool IsImported() const noexcept { return HasFlagBit(mFlags, ERDGResourceFlag::Imported); }
        IF_FORCEINLINE const String&      GetName() const noexcept { return mName; }
        IF_FORCEINLINE ERDGMemoryHeapType GetHeapType() const noexcept { return mHeapType; }
        IF_FORCEINLINE u32                GetTransientResourceId() const noexcept { return mTransientResourceId; }

        IF_FORCEINLINE void               SetHeapType(ERDGMemoryHeapType heapType) noexcept { mHeapType = heapType; }
        IF_FORCEINLINE void               SetTransientResourceId(u32 id) noexcept { mTransientResourceId = id; }

        virtual ERDGResourceType          GetType() const               = 0;
        virtual u64                       GetRequiredMemorySize() const = 0;
        virtual RHI::ERhiResourceState    GetInitialState() const       = 0;

    private:
        String              mName;
        RDGResourceLifetime mLifetime;
        ERDGResourceFlags   mFlags               = 0;
        ERDGMemoryHeapType  mHeapType            = ERDGMemoryHeapType::DeviceLocal;
        RDGGraphBuilder*    mBuilder             = nullptr;
        u32                 mTransientResourceId = ~0u;
    };

    class RDGTextureResource : public RDGResource
    {
    public:
        RDGTextureResource(
            RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags, const RDGTextureDesc& desc) noexcept
            : RDGResource(builder, name, flags), mDesc(desc), mImportedTexture(nullptr)
        {
        }
        RDGTextureResource(
            RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags, RHI::RhiTextureRef importedTex)
            : RDGResource(builder, name, flags), mImportedTexture(importedTex)
        {
            RDG_NOTNULL(importedTex, "Imported texture is null in RDGTextureResource");
            RDG_ASSERTION(HasFlagBit(flags, ERDGResourceFlag::Imported),
                "Imported texture must be marked as Imported in RDGTextureResource");
        }
        virtual ~RDGTextureResource() = default;

        virtual u64 GetRequiredMemorySize() const override
        {
            RDG_NOT_IMPLEMENTED();
            return 0;
        }

        virtual RHI::ERhiResourceState GetInitialState() const override
        {
            if (IsImported())
            {
                return mImportedTexture->GetInitialState();
            }
            return RHI::ERhiResourceState::Undefined;
        }

        RHI::RhiImageSubResource GetFullSubresource() const noexcept
        {
            if (IsImported())
            {
                return mImportedTexture->GetFullSubresource();
            }
            RHI::RhiImageSubResource subRes;
            subRes.mipLevel   = 0;
            subRes.arrayLayer = 0;
            subRes.mipCount   = mDesc.mMips;
            subRes.layerCount = mDesc.mArraySize;
            return subRes;
        }

        RDGResourceAccess_SRV* GetOrCreateSRV(const TOptional<RHI::RhiImageSubResource>& subRes)
        {
            RDG_ASSERTION(!OptionalNotEmpty(subRes), "Subresource view is not supported yet in RDGTextureResource");
            auto subresource = subRes.has_value() ? *subRes : GetFullSubresource();
            auto it          = mSRVs.find(subresource);
            if (it != mSRVs.end())
            {
                return it->second.get();
            }
            // Create new SRV
            Owner<RDGResourceAccess_SRV> srvAccess = MakeOwner<RDGResourceAccess_SRV>();
            mSRVs.emplace(subresource, std::move(srvAccess));
            return mSRVs[subresource].get();
        }
        RDGResourceAccess_UAV* GetOrCreateUAV(const TOptional<RHI::RhiImageSubResource>& subRes)
        {
            RDG_ASSERTION(!OptionalNotEmpty(subRes), "Subresource view is not supported yet in RDGTextureResource");
            auto subresource = subRes.has_value() ? *subRes : GetFullSubresource();
            auto it          = mUAVs.find(subresource);
            if (it != mUAVs.end())
            {
                return it->second.get();
            }
            // Create new UAV
            Owner<RDGResourceAccess_UAV> uavAccess = MakeOwner<RDGResourceAccess_UAV>();
            mUAVs.emplace(subresource, std::move(uavAccess));
            return mUAVs[subresource].get();
        }
        RDGResourceAccess_Texture* GetTextureAccess() { return &mTexAccess; }
        virtual ERDGResourceType   GetType() const override { return ERDGResourceType::Texture; }

        const RDGTextureDesc&      GetDesc() const noexcept { return mDesc; }

    private:
        RDGTextureDesc                                                   mDesc;
        RHI::RhiTextureRef                                               mImportedTexture = nullptr;

        RDGResourceAccess_Texture                                        mTexAccess;
        THashMap<RHI::RhiImageSubResource, Owner<RDGResourceAccess_SRV>> mSRVs;
        THashMap<RHI::RhiImageSubResource, Owner<RDGResourceAccess_UAV>> mUAVs;
    };

    class RDGBufferResource : public RDGResource
    {
    public:
        RDGBufferResource(
            RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags, const RDGBufferDesc& desc) noexcept
            : RDGResource(builder, name, flags), mDesc(desc), mImportedBuffer(nullptr)
        {
        }
        RDGBufferResource(RDGGraphBuilder* builder, const String& name, ERDGResourceFlags flags,
            RHI::RhiBufferRef importedBuf) noexcept
            : RDGResource(builder, name, flags), mImportedBuffer(importedBuf)
        {
            RDG_NOTNULL(importedBuf, "Imported buffer is null in RDGBufferResource");
            RDG_ASSERTION(HasFlagBit(flags, ERDGResourceFlag::Imported),
                "Imported buffer must be marked as Imported in RDGBufferResource");
        }
        virtual ~RDGBufferResource() = default;

        virtual u64 GetRequiredMemorySize() const override
        {
            RDG_NOT_IMPLEMENTED();
            return 0;
        }

        virtual RHI::ERhiResourceState GetInitialState() const override
        {
            return RHI::ERhiResourceState::UnorderedAccess;
        }
        virtual ERDGResourceType  GetType() const override { return ERDGResourceType::Buffer; }

        RDGResourceAccess_SRV*    GetSRV() { return &mSRV; }
        RDGResourceAccess_UAV*    GetUAV() { return &mUAV; }
        RDGResourceAccess_Buffer* GetBufferAccess() { return &mBufAccess; }

        const RDGBufferDesc&      GetDesc() const noexcept { return mDesc; }

    private:
        RDGBufferDesc            mDesc;
        RHI::RhiBufferRef        mImportedBuffer = nullptr;

        RDGResourceAccess_Buffer mBufAccess;
        RDGResourceAccess_SRV    mSRV;
        RDGResourceAccess_UAV    mUAV;
    };

    // ===== RDG Physical Resource =====
    class RDGPhysicalResource
    {
    };

    class RDGPhysicalTexture : public RDGPhysicalResource
    {
    public:
        RDGPhysicalTexture(const RDGTextureDesc& desc) : mDesc(desc), mTexture(nullptr) {}
        virtual ~RDGPhysicalTexture() = default;

        const RDGTextureDesc& GetDesc() const noexcept { return mDesc; }

    private:
        RDGTextureDesc                                     mDesc;
        RHI::RhiTextureRef                                 mTexture;
        THashMap<RHI::RhiImageSubResource, RHI::RhiSRVRef> mSRVs;
        THashMap<RHI::RhiImageSubResource, RHI::RhiUAVRef> mUAVs;
    };
    class RDGPhysicalBuffer : public RDGPhysicalResource
    {
    public:
        RDGPhysicalBuffer(const RDGBufferDesc& desc) : mDesc(desc), mBuffer(nullptr), mSRV(nullptr), mUAV(nullptr) {}
        virtual ~RDGPhysicalBuffer() = default;

        const RDGBufferDesc& GetDesc() const noexcept { return mDesc; }

    private:
        RDGBufferDesc     mDesc;
        RHI::RhiBufferRef mBuffer;
        RHI::RhiSRVRef    mSRV;
        RHI::RhiUAVRef    mUAV;
    };

} // namespace Ifrit::Runtime::RenderCore::RDG