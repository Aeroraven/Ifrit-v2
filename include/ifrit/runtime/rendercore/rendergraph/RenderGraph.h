#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/core/base/containers/Optional.h"
#include "ifrit/core/typing/EnumUtils.h"
#include "ifrit/core/reflection/Object.h"

#ifdef __INTELLISENSE__
    #define IFRIT_RDG_API
#else
    #define IFRIT_RDG_API IFRIT_RUNTIME_API
#endif

namespace Ifrit::Runtime::RenderCore::RDG
{
    class RDGResource;
    class RDGPass;
    class RDGGraphBuilder;

    class RDGTextureResource;
    class RDGBufferResource;
    class RDGScopeMarker;

    namespace ERDGResourceFlag_Scope
    {
        enum Enum : u8
        {
            None      = 0,
            Imported  = 0x1,
            Uploading = 0x2,
        };
    } // namespace ERDGResourceFlag_Scope

    namespace ERDGResourceAccessFlag_Scope
    {
        enum Enum : u32
        {
            SRVRead         = 0x1,
            UAVWrite        = 0x2,
            UAVRead         = 0x4,
            IndirectArgRead = 0x8,
            CopyDst         = 0x10,
            CopySrc         = 0x20,
            RenderTarget    = 0x40,
            DepthStencil    = 0x80,
        };
    } // namespace ERDGResourceAccessFlag_Scope

    enum class ERDGResourceType : u32
    {
        None    = 0,
        Texture = 1,
        Buffer  = 2,
    };

    enum class ERDGPassType : u32
    {
        Invalid  = 0,
        Compute  = 1,
        Graphics = 2,
        Transfer = 3,

        AsyncCompute  = 16,
        AsyncTransfer = 17,
    };

    enum class ERDGDebugVisualizationMode : u32
    {
        PassDAG               = 1,
        PassDAGWithResources  = 2,
        PhysicalResourceAlloc = 3,
    };

    namespace ERDGQueueType_Scope
    {
        enum Enum : u32
        {
            Graphics      = 0,
            AsyncCompute  = 1,
            AsyncTransfer = 2,
            Count         = 3,
        };
    } // namespace ERDGQueueType_Scope

    using ERDGQueueType = ERDGQueueType_Scope::Enum;

    namespace ERDGReadWriteMode_Scope
    {
        enum Enum : u8
        {
            None      = 0,
            Read      = 1,
            Write     = 2,
            ReadWrite = 3,
        };
    } // namespace ERDGReadWriteMode_Scope

    using ERDGResourceFlag       = ERDGResourceFlag_Scope::Enum;
    using ERDGResourceAccessFlag = ERDGResourceAccessFlag_Scope::Enum;
    using ERDGReadWriteModeFlag  = ERDGReadWriteMode_Scope::Enum;

    using ERDGReadWriteMode  = TEnumBitMask<ERDGReadWriteMode_Scope::Enum>;
    using ERDGResourceFlags  = TEnumBitMask<ERDGResourceFlag_Scope::Enum>;
    using ERDGResourceAccess = TEnumBitMask<ERDGResourceAccessFlag_Scope::Enum>;

    // ===== RDG Handles =====

    struct RDGResourceHandle
    {
        RDGGraphBuilder* mGraph = nullptr;
        u32              mIndex = 0;
        ERDGResourceType mType  = ERDGResourceType::None;

        RDGResourceHandle(RDGGraphBuilder& graph, u32 index, ERDGResourceType type)
            : mGraph(&graph), mIndex(index), mType(type)
        {
        }
    };

    struct RDGTextureHandle : public RDGResourceHandle
    {
        RDGTextureHandle(RDGGraphBuilder& graph, u32 index) : RDGResourceHandle(graph, index, ERDGResourceType::Texture)
        {
        }
    };

    struct RDGBufferHandle : public RDGResourceHandle
    {
        RDGBufferHandle(RDGGraphBuilder& graph, u32 index) : RDGResourceHandle(graph, index, ERDGResourceType::Buffer)
        {
        }
    };

    struct RDGPassHandle
    {
        RDGGraphBuilder* mGraph = nullptr;
        u32              mIndex = 0;

        RDGPassHandle(RDGGraphBuilder& graph, u32 index) : mGraph(&graph), mIndex(index) {}
    };

    // ===== RDG Resource/Pass Description =====

    struct RDGTextureDesc
    {
        RHI::ERhiImageUsage      mUsage     = RHI::ERhiImageUsageFlag::None;
        RHI::ERhiImageDimension  mDimension = RHI::ERhiImageDimension::Unknown;
        RHI::ERhiImageFormat     mFormat    = RHI::ERhiImageFormat::Undefined;
        u32                      mWidth     = 0;
        u32                      mHeight    = 0;
        u32                      mDepth     = 0;
        u32                      mMips      = 1;
        u32                      mSamples   = 1;
        u32                      mArraySize = 1;
        RHI::RhiClearColorValue  mClearValue;

        IF_NODISCARD inline bool operator==(const RDGTextureDesc& other) const noexcept
        {
            return mUsage == other.mUsage && mDimension == other.mDimension && mFormat == other.mFormat
                && mWidth == other.mWidth && mHeight == other.mHeight && mDepth == other.mDepth && mMips == other.mMips
                && mSamples == other.mSamples && mArraySize == other.mArraySize && mClearValue == other.mClearValue;
        }

        static inline RDGTextureDesc CreateTexture2D(
            u32 width, u32 height, RHI::ERhiImageFormat format, RHI::ERhiImageUsage usage = 0, u32 mipLevels = 1)
        {
            RDGTextureDesc desc;
            desc.mDimension = RHI::ERhiImageDimension::Texture2D;
            desc.mWidth     = width;
            desc.mHeight    = height;
            desc.mDepth     = 1;
            desc.mMips      = mipLevels;
            desc.mFormat    = format;
            desc.mUsage     = usage;
            return desc;
        }
    };

    struct RDGBufferDesc
    {
        RHI::ERhiBufferUsage     mUsage  = RHI::ERhiBufferUsageFlag::None;
        u64                      mSize   = 0;
        u32                      mStride = 0;

        IF_NODISCARD inline bool operator==(const RDGBufferDesc& other) const noexcept
        {
            return mUsage == other.mUsage && mSize == other.mSize && mStride == other.mStride;
        }
    };

    class IFRIT_RDG_API IRDGAccess_ResourceView
    {
    public:
        virtual RHI::RhiDescriptorHandle GetDescriptor() = 0;
    };

    class IFRIT_RDG_API IRDGAccess_Texture
    {
    public:
        virtual RHI::RhiTexture* GetTexture() = 0;
    };

    class IFRIT_RDG_API IRDGAccess_Buffer
    {
    public:
        virtual RHI::RhiBuffer* GetBuffer() = 0;
    };

    // ===== RDG Builder =====

    class IFRIT_RDG_API IRDGGraphBuilderSetupContext
    {
    public:
        virtual ~IRDGGraphBuilderSetupContext() = default;

        virtual IRDGAccess_ResourceView* CreateSRV(
            RDGTextureHandle handle, const TOptional<RHI::RhiImageSubResource>& subRes) = 0;
        virtual IRDGAccess_ResourceView* CreateSRV(RDGBufferHandle handle)              = 0;
        virtual IRDGAccess_ResourceView* CreateUAV(
            RDGTextureHandle handle, const TOptional<RHI::RhiImageSubResource>& subRes, ERDGReadWriteMode mode) = 0;
        virtual IRDGAccess_ResourceView* CreateUAV(RDGBufferHandle handle, ERDGReadWriteMode mode)              = 0;

        virtual IRDGAccess_Buffer*       AsIndirectArg(RDGBufferHandle handle) = 0;
        virtual IRDGAccess_Texture*      AsRenderTarget(RDGTextureHandle handle,
                 TOptional<RHI::RhiImageSubResource> clearValue, RHI::ERhiRenderTargetLoadOp loadOp,
                 TOptional<RHI::RhiClearColorValue> clearColor)                = 0;
        virtual IRDGAccess_Texture*      AsDepthStencil(RDGTextureHandle handle,
                 TOptional<RHI::RhiImageSubResource> clearValue, RHI::ERhiRenderTargetLoadOp loadOp,
                 TOptional<f32> clearDepth, TOptional<u32> clearStencil)       = 0;
    };

    struct RDGGraphBuilderExecuteContext
    {
    };

    template <typename T, typename TPassData>
    concept IConceptRDGPassFnSetup = requires(T t, TPassData& data, IRDGGraphBuilderSetupContext& context) {
        { t.operator()(data, context) };
    };

    template <typename T, typename TPassData>
    concept IConceptRDGPassFnExecute = requires(T t, const TPassData& data, RDGGraphBuilderExecuteContext& context) {
        { t.operator()(data, context) };
    };

    template <typename T> using TRDGExecuteFuncPtr = void (*)(const T&, RDGGraphBuilderExecuteContext&);

    struct RDGGraphBuilderInternal;
    struct RDGGraphBuilderArgs
    {
        bool mEnableAsyncCompute   = true;
        bool mEnableAsyncTransfer  = true;
        bool mEnableMemoryAliasing = false;
    };

    class IFRIT_RDG_API RDGGraphBuilder
    {
    public:
        using RDGPassData = Ifrit::Reflection::Object;

        RDGGraphBuilder(const RDGGraphBuilderArgs& args = RDGGraphBuilderArgs());
        ~RDGGraphBuilder();

        RDGTextureHandle DeclareTexture(const String& name, const RDGTextureDesc& desc);
        RDGBufferHandle  DeclareBuffer(const String& name, const RDGBufferDesc& desc);
        RDGTextureHandle ImportTexture(RHI::RhiTextureRef texture);
        RDGBufferHandle  ImportBuffer(RHI::RhiBufferRef buffer);

        void             Compile();
        void             DumpDebugFile(const String& path, ERDGDebugVisualizationMode mode);

        template <IDefaultCopyable TPassData, typename TFnSetup, typename TFnExecute>
            requires IConceptRDGPassFnSetup<TFnSetup, TPassData> && IConceptRDGPassFnExecute<TFnExecute, TPassData>
            && IConceptConvertibleToFuncPtr<TFnExecute>
        RDGPassHandle AddPass(const String& name, ERDGPassType type, TFnSetup setup, TFnExecute execute);

    protected:
        u32                            PreAllocatePass();
        void                           ActivatePassInSetupContext(u32 index);
        RDGPassData&                   GetPassData(u32 index);
        IRDGGraphBuilderSetupContext&  GetSetupContext();
        RDGGraphBuilderExecuteContext& GetExecuteContext();

        RDGPassHandle AddPassInternal(u32 idx, const String& name, TSinkArg<RDGPassData> passData, ERDGPassType type,
            TSinkArg<Fn<void()>> fnSetup, TSinkArg<Fn<void()>> fnExecute);

    private:
        RDGGraphBuilderInternal* mData;
    };

} // namespace Ifrit::Runtime::RenderCore::RDG

namespace std
{
    template <> struct hash<Ifrit::Runtime::RenderCore::RDG::RDGTextureDesc>
    {
        size_t operator()(const Ifrit::Runtime::RenderCore::RDG::RDGTextureDesc& desc) const noexcept
        {
            using namespace Ifrit;
            usize seed = 0;
            seed       = HashCombine(seed, static_cast<u32>(desc.mUsage));
            seed       = HashCombine(seed, static_cast<u32>(desc.mDimension));
            seed       = HashCombine(seed, static_cast<u32>(desc.mFormat));
            seed       = HashCombine(seed, desc.mWidth);
            seed       = HashCombine(seed, desc.mHeight);
            seed       = HashCombine(seed, desc.mDepth);
            seed       = HashCombine(seed, desc.mMips);
            seed       = HashCombine(seed, desc.mSamples);
            seed       = HashCombine(seed, desc.mArraySize);
            return seed;
        }
    };
    template <> struct hash<Ifrit::Runtime::RenderCore::RDG::RDGBufferDesc>
    {
        size_t operator()(const Ifrit::Runtime::RenderCore::RDG::RDGBufferDesc& desc) const noexcept
        {
            using namespace Ifrit;
            usize seed = 0;
            seed       = HashCombine(seed, static_cast<u32>(desc.mUsage));
            seed       = HashCombine(seed, static_cast<u64>(desc.mSize));
            seed       = HashCombine(seed, desc.mStride);
            return seed;
        }
    };
} // namespace std

#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.inl"