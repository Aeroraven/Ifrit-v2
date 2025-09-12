#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/core/base/containers/Optional.h"

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

    namespace ERDGResourceFlag
    {
        enum Enum
        {
            Imported  = 0x1,
            Uploading = 0x2,
        };
    } // namespace ERDGResourceFlag

    enum class ERDGResourceAccessFlag : u32
    {
        SRVRead         = 0x1,
        UAVWrite        = 0x2,
        UAVRead         = 0x4,
        IndirectArgRead = 0x8,
        CopyDst         = 0x10,
        CopySrc         = 0x20,
        RenderTarget    = 0x40,
    };

    enum class ERDGResourceType : u32
    {
        None    = 0,
        Texture = 1,
        Buffer  = 2,
    };

    enum ERDGPassType
    {
        Invalid  = 0,
        Compute  = 1,
        Graphics = 2,
        Transfer = 3,

        AsyncCompute  = 16,
        AsyncTransfer = 17,
    };

    enum ERDGQueueType
    {
        Graphics      = 0,
        AsyncCompute  = 1,
        AsyncTransfer = 2,
        Count         = 3,
    };

    enum class ERDGReadWriteMode : u8
    {
        None      = 0,
        Read      = 1,
        Write     = 2,
        ReadWrite = 3,
    };

    using ERDGResourceFlags  = u32;
    using ERDGResourceAccess = u32;

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
        RHI::ERhiImageUsage     mUsage     = RHI::ERhiImageUsageFlag::None;
        RHI::ERhiImageDimension mDimension = RHI::ERhiImageDimension::Unknown;
        RHI::ERhiImageFormat    mFormat    = RHI::ERhiImageFormat::Undefined;
        u32                     mWidth     = 0;
        u32                     mHeight    = 0;
        u32                     mDepth     = 0;
        u32                     mMips      = 1;
        u32                     mSamples   = 1;
        u32                     mArraySize = 1;
        RHI::RhiClearColorValue mClearValue;
    };

    struct RDGBufferDesc
    {
        RHI::ERhiBufferUsage mUsage  = RHI::ERhiBufferUsageFlag::None;
        u64                  mSize   = 0;
        u32                  mStride = 0;
    };

    class IFRIT_RDG_API IRDGAccess_ResourceView
    {
    public:
        virtual RHI::RhiDescriptorHandle GetDescriptor() = 0;
    };

    class IFRIT_RDG_API IRDGAccess_Texture
    {
    public:
        virtual RHI::RhiTextureRef GetTexture() = 0;
    };

    class IFRIT_RDG_API IRDGAccess_Buffer
    {
    public:
        virtual RHI::RhiBufferRef GetBuffer() = 0;
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
    concept IConceptRDGPassFnSetup = requires(TPassData& data, IRDGGraphBuilderSetupContext& context) {
        { T()(data, context) };
    };

    template <typename T, typename TPassData>
    concept IConceptRDGPassFnExecute = requires(const TPassData& data, RDGGraphBuilderExecuteContext& context) {
        { T()(data, context) };
    };

    struct RDGGraphBuilderInternal;

    struct RDGGraphBuilderArgs
    {
        bool mEnableAsyncCompute   = false;
        bool mEnableAsyncTransfer  = false;
        bool mEnableMemoryAliasing = false;
    };

    class IFRIT_RDG_API RDGGraphBuilder
    {
    public:
        RDGGraphBuilder(const RDGGraphBuilderArgs& args = RDGGraphBuilderArgs());
        ~RDGGraphBuilder();

        RDGTextureHandle DeclareTexture(const String& name, const RDGTextureDesc& desc);
        RDGBufferHandle  DeclareBuffer(const String& name, const RDGBufferDesc& desc);
        RDGTextureHandle ImportTexture(RHI::RhiTextureRef texture);
        RDGBufferHandle  ImportBuffer(RHI::RhiBufferRef buffer);

        template <typename TPassData, typename TFnSetup, typename TFnExecute>
            requires IConceptRDGPassFnSetup<TFnSetup, TPassData> && IConceptRDGPassFnExecute<TFnExecute, TPassData>
        RDGPassHandle AddPass(const String& name, ERDGPassType type, TFnSetup setup, TFnExecute execute);

        void          Compile();

    protected:
        RDGPassHandle AddPassInternal(const String& name, ERDGPassType type, Fn<void()> setup, Fn<void()> execute);

    private:
        RDGGraphBuilderInternal* mData;
    };

} // namespace Ifrit::Runtime::RenderCore::RDG

#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.inl"