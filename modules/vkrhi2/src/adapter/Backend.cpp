#include "ifrit/vkrhi2/adapter/Backend.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/adapter/CommandContext.h"
#include "ifrit/vkrhi2/adapter/MemoryResource.h"
#include "ifrit/vkrhi2/adapter/DisplayViewport.h"
#include "ifrit/vkrhi2/adapter/DescriptorHeap.h"
#include "ifrit/vkrhi2/adapter/PipelineState.h"

namespace Ifrit::RHI::VulkanRHI2
{
    struct VA_BackendInternal
    {
        bool                mSetup = false;
        Owner<VA_Device>    mDevice;
        Owner<VA_Swapchain> mSwapchain;
    };

    IFRIT_VKRHI2_API VA_Backend::VA_Backend() { mInternal = new VA_BackendInternal(); }

    IFRIT_VKRHI2_API VA_Backend::~VA_Backend()
    {
        if (mInternal->mSetup)
            Finalize();
        delete mInternal;
        mInternal = nullptr;
    }

    IFRIT_VKRHI2_API void VA_Backend::Init(const RhiInitializeArguments& args)
    {
        // Device
        mInternal->mDevice = MakeOwner<VA_Device>(args);

        // Swapchain
        auto             initArgs = mInternal->mDevice->GetInitializationArgs();
        VA_SwapchainDesc swapchainDesc;
        swapchainDesc.mWidth                 = initArgs.mSurfaceWidth;
        swapchainDesc.mHeight                = initArgs.mSurfaceHeight;
        swapchainDesc.mDesiredNumBackBuffers = initArgs.mExpectedSwapchainImageCount;
        swapchainDesc.mFormat                = ERhiImageFormat::B8G8R8A8_SRGB;
        mInternal->mSwapchain                = MakeOwner<VA_Swapchain>(mInternal->mDevice.get(), swapchainDesc);

        InitRenderResources();

        mInternal->mSetup = true;
    }

    IFRIT_VKRHI2_API void VA_Backend::Finalize()
    {
        mInternal->mSwapchain = nullptr;
        mInternal->mDevice    = nullptr;
        mInternal->mSetup     = false;
    }

    IFRIT_VKRHI2_API VA_Device*               VA_Backend::GetDevice() const { return mInternal->mDevice.get(); }
    IFRIT_VKRHI2_API VA_StagingBufferManager* VA_Backend::GetStagingBufferManager() const
    {
        return mInternal->mDevice->GetStagingBufferManager();
    }
    IFRIT_VKRHI2_API IRhiCommandContext* VA_Backend::GetImmediateContext()
    {
        auto ret = mInternal->mDevice->GetImmediateContext();
        return ret;
    }

    IFRIT_VKRHI2_API Owner<IRhiCommandContext> VA_Backend::GetUploadContext()
    {
        return mInternal->mDevice->GetUploadContext();
    }
    IFRIT_VKRHI2_API Owner<IRhiCommandContext> VA_Backend::GetCommandContext(ERhiCommandListPipelineType type)
    {
        return mInternal->mDevice->GetCommandContext(type);
    }
    RhiTextureRef VA_Backend::CreateTexture(const RhiTextureDesc& desc)
    {
        auto texRaw = new VA_Texture(static_cast<RhiDevice*>(mInternal->mDevice.get()), desc);
        return MakeCountRef<RhiTexture>(texRaw);
    }
    RhiBufferRef VA_Backend::CreateBuffer(const RhiBufferDesc& desc)
    {
        auto bufRaw = new VA_Buffer(static_cast<RhiDevice*>(mInternal->mDevice.get()), desc);
        return MakeCountRef<RhiBuffer>(bufRaw);
    }

    IFRIT_VKRHI2_API void VA_Backend::BeginFrame()
    {
        // Acquire next image from swapchain
        mInternal->mSwapchain->AcquireNextImage();

        // Frame advance
        auto queues = mInternal->mDevice->GetActiveQueues();
        queues.mGraphics->RecycleCmdLists();
        queues.mAsyncCompute->RecycleCmdLists();
        queues.mTransfer->RecycleCmdLists();

        mInternal->mDevice->FrameAdvance();
        auto cmdList      = GetCommandListExecutor()->GetImmediateCmdList();
        auto cmdCtx       = cmdList->GetActiveContext();
        auto nativeCmdCtx = static_cast<VA_CommandListContext*>(cmdCtx);
        nativeCmdCtx->RegisterDependencies({ mInternal->mSwapchain->GetCurrentImageAcquiredSemaphore() });
    }
    IFRIT_VKRHI2_API void VA_Backend::EndFrame()
    {
        // Present the current image
        auto cmdList = GetCommandListExecutor()->GetImmediateCmdList();
        auto cmdCtx  = mInternal->mDevice->GetImmediateContext();
        mInternal->mSwapchain->Present(static_cast<VA_CommandListContext*>(cmdCtx));
    }
    IFRIT_VKRHI2_API RhiDynamicUtils* VA_Backend::GetDynamicUtils()
    {
        return mInternal->mDevice->GetDeviceRHIFunctions();
    }
    IFRIT_VKRHI2_API RhiUAVRef VA_Backend::CreateUAV(RhiTexture* texture, RhiImageSubResource subResource)
    {
        RhiResourceViewDesc desc;
        desc.mType                     = ERhiResourceViewedType::Texture;
        desc.mTextureView.mSubResource = subResource;
        auto view = new VA_ResourceViewUAV(desc, static_cast<VA_Texture*>(texture), mInternal->mDevice.get());
        return MakeCountRef<RhiUnorderedAccessView>(view);
    }
    IFRIT_VKRHI2_API RhiUAVRef VA_Backend::CreateUAV(RhiBuffer* buffer)
    {
        RhiResourceViewDesc desc;
        desc.mType               = ERhiResourceViewedType::Buffer;
        desc.mBufferView.mOffset = 0;
        desc.mBufferView.mSize   = ~0u;
        auto view = new VA_ResourceViewUAV(desc, static_cast<VA_Buffer*>(buffer), mInternal->mDevice.get());
        return MakeCountRef<RhiUnorderedAccessView>(view);
    }
    IFRIT_VKRHI2_API RhiSRVRef VA_Backend::CreateSRV(RhiTexture* texture, RhiImageSubResource subResource)
    {
        RhiResourceViewDesc desc;
        desc.mType                     = ERhiResourceViewedType::Texture;
        desc.mTextureView.mSubResource = subResource;
        auto view = new VA_ResourceViewSRV(desc, static_cast<VA_Texture*>(texture), mInternal->mDevice.get());
        return MakeCountRef<RhiShaderReadView>(view);
    }
    IFRIT_VKRHI2_API RhiSRVRef VA_Backend::CreateSRV(RhiBuffer* buffer)
    {
        RhiResourceViewDesc desc;
        desc.mType               = ERhiResourceViewedType::Buffer;
        desc.mBufferView.mOffset = 0;
        desc.mBufferView.mSize   = ~0u;
        auto view = new VA_ResourceViewSRV(desc, static_cast<VA_Buffer*>(buffer), mInternal->mDevice.get());
        return MakeCountRef<RhiShaderReadView>(view);
    }

    IFRIT_VKRHI2_API RhiComputePipeline* VA_Backend::Experimental_GetComputePipeline(
        const RhiComputePipelineStateDesc& desc)
    {
        auto psoCache = mInternal->mDevice->GetPipelineStateCache();
        auto pipeline = psoCache->GetComputePipeline(desc);
        return pipeline;
    }
} // namespace Ifrit::RHI::VulkanRHI2