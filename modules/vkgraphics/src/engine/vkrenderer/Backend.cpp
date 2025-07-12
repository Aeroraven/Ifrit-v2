
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

#include "ifrit/vkgraphics/engine/vkrenderer/Backend.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderPass.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderTargets.h"
#include "ifrit/vkgraphics/engine/vkrenderer/StagedMemoryResource.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Timer.h"

#include "ifrit/vkgraphics/engine/fsr2extension/FSR2Processor.h"

#include "ifrit/core/algo/ConcurrentVector.h"
using namespace Ifrit;

namespace Ifrit::RHI::VulkanAdapter
{
    inline VkFormat toVkFormat(RHI::RhiImageFormat format) { return static_cast<VkFormat>(format); }

    struct RhiVulkanBackendImplDetails : public NonCopyable
    {
        Owner<CommandExecutor>                         m_commandExecutor;
        Owner<DescriptorManager>                       m_descriptorManager;
        Owner<ResourceManager>                         m_resourceManager;
        Vec<Owner<StagedSingleBuffer>>                 m_stagedSingleBuffer;
        Owner<PipelineCache>                           m_pipelineCache;

        Owner<RegisteredResourceMapper>                m_mapper;

        // managed passes
        Vec<Owner<ComputePass>>                        m_computePasses;
        Vec<Owner<GraphicsPass>>                       m_graphicsPasses;
        Vec<Owner<DescriptorBindlessIndices>>          m_bindlessIndices;

        // managed descriptors
        Vec<Ref<RHI::RhiDescHandleLegacy>>             m_bindlessIdRefs;

        // some utility buffers
        RHI::RhiBufferRef                              m_fullScreenQuadVertexBuffer;
        Ref<VertexBufferDescriptor>                    m_fullScreenQuadVertexBufferDescriptor;

        // timers
        Vec<Ref<DeviceTimer>>                          m_deviceTimers;

        TConcurrentGrowthVector<Ref<ShaderCollection>> m_shaderModule;
    };

    IFRIT_APIDECL
    RhiVulkanBackend::RhiVulkanBackend(const RHI::RhiInitializeArguments& args)
    {
        m_device                           = MakeOwner<EngineContext>(args);
        auto engineContext                 = CheckedCast<EngineContext>(m_device.get());
        m_swapChain                        = MakeOwner<Swapchain>(engineContext);
        auto swapchain                     = CheckedCast<Swapchain>(m_swapChain.get());
        m_implDetails                      = new RhiVulkanBackendImplDetails();
        m_implDetails->m_descriptorManager = MakeOwner<DescriptorManager>(engineContext);
        m_implDetails->m_resourceManager   = MakeOwner<ResourceManager>(engineContext);
        m_implDetails->m_commandExecutor   = MakeOwner<CommandExecutor>(
            engineContext, swapchain, m_implDetails->m_descriptorManager.get(), m_implDetails->m_resourceManager.get());
        m_implDetails->m_pipelineCache = MakeOwner<PipelineCache>(engineContext);
        m_implDetails->m_mapper        = MakeOwner<RegisteredResourceMapper>();
        m_implDetails->m_commandExecutor->setQueues(1, args.m_expectedGraphicsQueueCount,
            args.m_expectedComputeQueueCount, args.m_expectedTransferQueueCount,
            args.m_expectedSwapchainImageCount + 1);

        // All done, then make a full screen quad buffer
        BufferCreateInfo ci{}; // One Triangle,
        ci.size        = 3 * 2 * sizeof(float);
        ci.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ci.hostVisible = false;
        m_implDetails->m_fullScreenQuadVertexBuffer = m_implDetails->m_resourceManager->CreateSimpleBufferUnmanaged(ci);

        auto singleBufferPtr = CheckedCast<SingleBuffer>(m_implDetails->m_fullScreenQuadVertexBuffer.get());
        StagedSingleBuffer stagedQuadBuffer(engineContext, singleBufferPtr);

        auto               transferQueue = m_implDetails->m_commandExecutor->GetQueue(QueueRequirement::Transfer);
        transferQueue->RunSyncCommand([&](const RHI::RhiCommandList* cmd) {
            float data[] = {
                0.0f, 0.0f, //
                4.0f, 0.0f, //
                0.0f, 4.0f, //
            };
            stagedQuadBuffer.CmdCopyToDevice(cmd, data, sizeof(data), 0);
        });
        m_implDetails->m_fullScreenQuadVertexBufferDescriptor = MakeRef<VertexBufferDescriptor>();
        m_implDetails->m_fullScreenQuadVertexBufferDescriptor->AddBinding(
            { 0 }, { RHI::RhiImageFormat::RhiImgFmt_R32G32_SFLOAT }, { 0 }, 2 * sizeof(float));
    }

    IFRIT_APIDECL void RhiVulkanBackend::WaitDeviceIdle()
    {
        auto p = CheckedCast<EngineContext>(m_device.get());
        p->WaitIdle();
    }

    IFRIT_APIDECL RHI::RhiCapabilityList RhiVulkanBackend::GetCapabilities() const
    {
        auto ctx = CheckedCast<EngineContext>(m_device.get());
        return ctx->GetCapabilities();
    }

    IFRIT_APIDECL Ref<RHI::RhiDeviceTimer> RhiVulkanBackend::CreateDeviceTimer()
    {
        auto swapchain        = CheckedCast<Swapchain>(m_swapChain.get());
        auto numFrameInFlight = swapchain->GetNumBackbuffers();
        auto p                = MakeRef<DeviceTimer>(CheckedCast<EngineContext>(m_device.get()), numFrameInFlight);
        m_implDetails->m_deviceTimers.push_back(p);
        return p;
    }

    IFRIT_APIDECL RHI::RhiBufferRef RhiVulkanBackend::CreateBuffer(
        const String& name, u32 size, u32 usage, bool hostVisible, bool addUAV) const
    {
        BufferCreateInfo ci{};
        ci.size        = size;
        ci.usage       = usage;
        ci.hostVisible = hostVisible;
        auto p         = m_implDetails->m_resourceManager->CreateSimpleBufferUnmanaged(ci);
        p->SetDebugName(name);

        if (addUAV)
        {
            auto buffer            = CheckedCast<SingleBuffer>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageBuffer(buffer);
            p->SetDescriptorHandle(RHI::RhiDescriptorHandle(RHI::RhiDescriptorHeapType::StorageBuffer, id));
        }
        return p;
    }

    IFRIT_APIDECL RHI::RhiBufferRef RhiVulkanBackend::GetFullScreenQuadVertexBuffer() const
    {
        return m_implDetails->m_fullScreenQuadVertexBuffer;
    }

    IFRIT_APIDECL RHI::RhiBufferRef RhiVulkanBackend::CreateBufferDevice(
        const String& name, u32 size, u32 usage, bool addUAV) const
    {
        iAssertion(size > 0, "Backend: Buffer size should be larger than 0");
        BufferCreateInfo ci{};
        ci.size        = size;
        ci.usage       = usage;
        ci.hostVisible = false;
        auto p         = m_implDetails->m_resourceManager->CreateSimpleBufferUnmanaged(ci);
        p->SetDebugName(name);
        if (addUAV)
        {
            auto buffer            = CheckedCast<SingleBuffer>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageBuffer(buffer);
            p->SetDescriptorHandle(RHI::RhiDescriptorHandle(RHI::RhiDescriptorHeapType::StorageBuffer, id));
        }
        return p;
    }
    IFRIT_APIDECL Ref<RHI::RhiMultiBuffer> RhiVulkanBackend::CreateBufferCoherent(
        u32 size, u32 usage, u32 numCopies) const
    {
        BufferCreateInfo ci{};
        ci.size        = size;
        ci.usage       = usage;
        ci.hostVisible = true;
        if (numCopies == ~0u)
        {
            // Use num backbuffers
            auto swapchain = CheckedCast<Swapchain>(m_swapChain.get());
            numCopies      = swapchain->GetNumBackbuffers();
        }
        return m_implDetails->m_resourceManager->CreateTracedMultipleBuffer(ci, numCopies);
    }

    IFRIT_APIDECL Ref<RHI::RhiStagedSingleBuffer> RhiVulkanBackend::CreateStagedSingleBuffer(RHI::RhiBuffer* target)
    {
        // TODO: release memory, (not managed)
        auto buffer        = CheckedCast<SingleBuffer>(target);
        auto engineContext = CheckedCast<EngineContext>(m_device.get());
        auto ptr           = MakeRef<StagedSingleBuffer>(engineContext, buffer);
        return ptr;
    }

    IFRIT_APIDECL RHI::RhiQueue* RhiVulkanBackend::GetQueue(RHI::RhiQueueCapability req)
    {
        QueueRequirement reqs;
        if (req == RHI::RhiQueueCapability::RhiQueue_Graphics)
        {
            reqs = QueueRequirement::Graphics;
        }
        else if (req == RHI::RhiQueueCapability::RhiQueue_Compute)
        {
            reqs = QueueRequirement::Compute;
        }
        else if (req == RHI::RhiQueueCapability::RhiQueue_Transfer)
        {
            reqs = QueueRequirement::Transfer;
        }
        else if (req == (RHI::RhiQueueCapability::RhiQueue_Graphics | RHI::RhiQueueCapability::RhiQueue_Compute))
        {
            reqs = QueueRequirement::Graphics_Compute;
        }
        else if (req == (RHI::RhiQueueCapability::RhiQueue_Graphics | RHI::RhiQueueCapability::RhiQueue_Transfer))
        {
            reqs = QueueRequirement::Graphics_Transfer;
        }
        else if (req == (RHI::RhiQueueCapability::RhiQueue_Compute | RHI::RhiQueueCapability::RhiQueue_Transfer))
        {
            reqs = QueueRequirement::Compute_Transfer;
        }
        else if (req
            == (RHI::RhiQueueCapability::RhiQueue_Graphics | RHI::RhiQueueCapability::RhiQueue_Compute
                | RHI::RhiQueueCapability::RhiQueue_Transfer))
        {
            reqs = QueueRequirement::Universal;
        }
        auto s = m_implDetails->m_commandExecutor->GetQueue(reqs);
        if (s == nullptr)
        {
            throw std::runtime_error("Queue not found");
        }
        return s;
    }

    IFRIT_APIDECL Ref<RHI::RhiShaderCollection> RhiVulkanBackend::CreateShader(const String& name,
        const Vec<char>& code, const String& entry, RHI::RhiShaderStage stage, RHI::RhiShaderSourceType sourceType)
    {
        ShaderCollectionCI ci{};
        ci.m_Code             = code;
        ci.m_EntryPoint       = entry;
        ci.m_Stage            = stage;
        ci.m_SourceType       = sourceType;
        ci.m_FileName         = name;
        auto shaderCollection = MakeRef<ShaderCollection>(CheckedCast<EngineContext>(m_device.get()), ci);
        m_implDetails->m_shaderModule.PushBack(shaderCollection);
        return shaderCollection;
    }

    IFRIT_APIDECL RHI::RhiTextureRef RhiVulkanBackend::CreateTexture2D(
        const String& name, u32 width, u32 height, RHI::RhiImageFormat format, u32 extraFlags, bool addUAV)
    {
        auto p = m_implDetails->m_resourceManager->CreateTexture2DDeviceUnmanaged(
            width, height, toVkFormat(format), extraFlags);
        p->SetDebugName(name);
        if (addUAV)
        {
            auto texture           = CheckedCast<SingleDeviceImage>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageImage(texture, { 0, 0, 1, 1 });
            p->SetDescriptorHandle(RHI::RhiDescriptorHandle(RHI::RhiDescriptorHeapType::StorageImage, id));
        }
        return p;
    }

    IFRIT_APIDECL RHI::RhiTextureRef RhiVulkanBackend::CreateTexture2DMsaa(
        const String& name, u32 width, u32 height, RHI::RhiImageFormat format, u32 extraFlags, u32 samples)
    {
        auto p = m_implDetails->m_resourceManager->CreateTexture2DDeviceUnmanaged(
            width, height, toVkFormat(format), extraFlags, samples);
        p->SetDebugName(name);
        return p;
    }

    IFRIT_APIDECL RHI::RhiTextureRef RhiVulkanBackend::CreateDepthTexture(
        const String& name, u32 width, u32 height, bool addUAV)
    {
        auto p = m_implDetails->m_resourceManager->CreateDepthAttachment(width, height);
        p->SetDebugName(name);
        if (addUAV)
        {
            auto texture           = CheckedCast<SingleDeviceImage>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageImage(texture, { 0, 0, 1, 1 });
            p->SetDescriptorHandle(RHI::RhiDescriptorHandle(RHI::RhiDescriptorHeapType::StorageImage, id));
        }
        return p;
    }

    IFRIT_APIDECL RHI::RhiTextureRef RhiVulkanBackend::CreateTexture3D(
        const String& name, u32 width, u32 height, u32 depth, RHI::RhiImageFormat format, u32 extraFlags, bool addUAV)
    {
        auto p =
            m_implDetails->m_resourceManager->CreateTexture3D(width, height, depth, toVkFormat(format), extraFlags);
        p->SetDebugName(name);
        if (addUAV)
        {
            auto texture           = CheckedCast<SingleDeviceImage>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageImage(texture, { 0, 0, 1, 1 });
            p->SetDescriptorHandle(RHI::RhiDescriptorHandle(RHI::RhiDescriptorHeapType::StorageImage, id));
        }
        return p;
    };

    IFRIT_APIDECL RHI::RhiTextureRef RhiVulkanBackend::CreateMipMapTexture(
        const String& name, u32 width, u32 height, u32 mips, RHI::RhiImageFormat format, u32 extraFlags, bool addUAV)
    {
        auto p =
            m_implDetails->m_resourceManager->createMipTexture(width, height, mips, toVkFormat(format), extraFlags);
        p->SetDebugName(name);
        if (addUAV)
        {
            auto texture           = CheckedCast<SingleDeviceImage>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageImage(texture, { 0, 0, 1, 1 });
            p->SetDescriptorHandle(RHI::RhiDescriptorHandle(RHI::RhiDescriptorHeapType::StorageImage, id));
        }
        return p;
    }

    IFRIT_APIDECL RHI::RhiSamplerRef RhiVulkanBackend::CreateSampler(
        RHI::RhiSamplerFilter filter, RHI::RhiSamplerWrapMode addressMode, bool addBinding)
    {
        RHI::RhiSamplerRef sampler = nullptr;
        if (filter == RHI::RhiSamplerFilter::Linear)
        {
            if (addressMode == RHI::RhiSamplerWrapMode::Clamp)
            {
                sampler = m_implDetails->m_resourceManager->CreateTrivialBilinearSampler(false);
            }
            else if (addressMode == RHI::RhiSamplerWrapMode::Repeat)
            {
                sampler = m_implDetails->m_resourceManager->CreateTrivialBilinearSampler(true);
            }
        }
        else if (filter == RHI::RhiSamplerFilter::Nearest)
        {
            if (addressMode == RHI::RhiSamplerWrapMode::Clamp)
            {
                sampler = m_implDetails->m_resourceManager->CreateTrivialNearestSampler(false);
            }
            else if (addressMode == RHI::RhiSamplerWrapMode::Repeat)
            {
                sampler = m_implDetails->m_resourceManager->CreateTrivialNearestSampler(true);
            }
        }
        else
        {
            throw std::runtime_error("Unsupported sampler filter or address mode");
        }

        if (addBinding)
        {
            auto samplerPtr = CheckedCast<Sampler>(sampler.get());
            auto id         = m_implDetails->m_descriptorManager->RegisterSamplers(samplerPtr);
            sampler->SetDescriptorHandle(RHI::RhiDescriptorHandle(RHI::RhiDescriptorHeapType::Sampler, id));
        }
        return sampler;
    }

    // Deprecating
    IFRIT_APIDECL RHI::RhiComputePass* RhiVulkanBackend::CreateComputePass()
    {
        auto pass =
            MakeOwner<ComputePass>(CheckedCast<EngineContext>(m_device.get()), m_implDetails->m_pipelineCache.get(),
                m_implDetails->m_descriptorManager.get(), m_implDetails->m_mapper.get());
        auto ptr = pass.get();
        ptr->SetDefaultNumMultiBuffers(m_swapChain->GetNumBackbuffers());
        m_implDetails->m_computePasses.push_back(std::move(pass));
        return ptr;
    }

    // Deprecating
    IFRIT_APIDECL RHI::RhiGraphicsPass* RhiVulkanBackend::CreateGraphicsPass()
    {
        auto pass =
            MakeOwner<GraphicsPass>(CheckedCast<EngineContext>(m_device.get()), m_implDetails->m_pipelineCache.get(),
                m_implDetails->m_descriptorManager.get(), m_implDetails->m_mapper.get());
        auto ptr = pass.get();
        ptr->SetDefaultNumMultiBuffers(m_swapChain->GetNumBackbuffers());
        m_implDetails->m_graphicsPasses.push_back(std::move(pass));
        return ptr;
    }

    IFRIT_APIDECL Owner<RHI::RhiComputePass> RhiVulkanBackend::CreateComputePass2()
    {
        auto pass =
            MakeOwner<ComputePass>(CheckedCast<EngineContext>(m_device.get()), m_implDetails->m_pipelineCache.get(),
                m_implDetails->m_descriptorManager.get(), m_implDetails->m_mapper.get());
        auto ptr = pass.get();
        ptr->SetDefaultNumMultiBuffers(m_swapChain->GetNumBackbuffers());
        return pass;
    }
    IFRIT_APIDECL Owner<RHI::RhiGraphicsPass> RhiVulkanBackend::CreateGraphicsPass2()
    {
        auto pass =
            MakeOwner<GraphicsPass>(CheckedCast<EngineContext>(m_device.get()), m_implDetails->m_pipelineCache.get(),
                m_implDetails->m_descriptorManager.get(), m_implDetails->m_mapper.get());
        auto ptr = pass.get();
        ptr->SetDefaultNumMultiBuffers(m_swapChain->GetNumBackbuffers());
        return pass;
    }

    IFRIT_APIDECL RHI::RhiTexture* RhiVulkanBackend::GetSwapchainImage()
    {
        return m_implDetails->m_commandExecutor->GetSwapchainImageResource();
    }

    IFRIT_APIDECL void RhiVulkanBackend::BeginFrame()
    {
        m_implDetails->m_commandExecutor->BeginFrame();
        m_implDetails->m_resourceManager->SetActiveFrame(m_swapChain->GetCurrentImageIndex());
        for (auto& desc : m_implDetails->m_bindlessIndices)
        {
            desc->SetActiveFrame(m_swapChain->GetCurrentImageIndex());
        }
        for (auto& idRef : m_implDetails->m_bindlessIdRefs)
        {
            idRef->activeFrame = m_swapChain->GetCurrentImageIndex();
        }
        for (auto& timer : m_implDetails->m_deviceTimers)
        {
            timer->FrameProceed();
        }

        // get engine context
        auto engineContext = CheckedCast<EngineContext>(m_device.get());
        auto deleteList    = engineContext->GetDeleteQueue();
        auto nums          = deleteList->ProcessDeleteQueue();
    }
    IFRIT_APIDECL void RhiVulkanBackend::EndFrame() { m_implDetails->m_commandExecutor->EndFrame(); }
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> RhiVulkanBackend::GetSwapchainFrameReadyEventHandler()
    {
        auto                  swapchain = CheckedCast<Swapchain>(m_swapChain.get());
        auto                  sema      = swapchain->GetImageAvailableSemaphoreCurrentFrame();
        TimelineSemaphoreWait wait;
        wait.m_isSwapchainSemaphore = true;
        wait.m_semaphore            = sema;
        return MakeOwner<TimelineSemaphoreWait>(wait);
    }
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> RhiVulkanBackend::GetSwapchainRenderDoneEventHandler()
    {
        auto                  swapchain = CheckedCast<Swapchain>(m_swapChain.get());
        auto                  sema      = swapchain->GetRenderingFinishSemaphoreCurrentFrame();
        auto                  fence     = swapchain->GetCurrentFrameFence();
        TimelineSemaphoreWait wait;
        wait.m_isSwapchainSemaphore = true;
        wait.m_semaphore            = sema;
        wait.m_fence                = fence;
        return MakeOwner<TimelineSemaphoreWait>(wait);
    }

    Ref<RHI::RhiColorAttachment> RhiVulkanBackend::CreateRenderTarget(RHI::RhiTexture* renderTarget,
        RHI::RhiClearValue2 clearValue, RHI::RhiRenderTargetLoadOp loadOp, u32 mips, u32 layers)
    {
        auto attachment = MakeRef<ColorAttachment>(renderTarget, clearValue, loadOp, mips, layers);
        return attachment;
    }

    Ref<RHI::RhiDepthStencilAttachment> RhiVulkanBackend::CreateRenderTargetDepthStencil(
        RHI::RhiTexture* renderTarget, RHI::RhiClearValue2 clearValue, RHI::RhiRenderTargetLoadOp loadOp)
    {
        auto attachment = MakeRef<DepthStencilAttachment>(renderTarget, clearValue, loadOp);
        return attachment;
    }

    Ref<RHI::RhiRenderTargets> RhiVulkanBackend::CreateRenderTargets()
    {
        auto ctx = CheckedCast<EngineContext>(m_device.get());
        return MakeRef<RenderTargets>(ctx);
    }

    IFRIT_APIDECL RhiVulkanBackend::~RhiVulkanBackend() { delete m_implDetails; }

    IFRIT_APIDECL RHI::RhiBindlessDescriptorRef* RhiVulkanBackend::CreateBindlessDescriptorRef()
    {
        auto ref = MakeOwner<DescriptorBindlessIndices>(CheckedCast<EngineContext>(m_device.get()),
            m_implDetails->m_descriptorManager.get(), m_swapChain->GetNumBackbuffers());
        auto ptr = ref.get();
        m_implDetails->m_bindlessIndices.push_back(std::move(ref));
        return ptr;
    }

    IFRIT_APIDECL Ref<RHI::RhiDescHandleLegacy> RhiVulkanBackend::RegisterUniformBuffer(RHI::RhiMultiBuffer* buffer)
    {
        Vec<u32> ids;
        auto     descriptorManager = m_implDetails->m_descriptorManager.get();
        auto     multiBuffer       = CheckedCast<MultiBuffer>(buffer);
        auto     numBackbuffers    = m_swapChain->GetNumBackbuffers();
        for (u32 i = 0; i < numBackbuffers; i++)
        {
            auto id = descriptorManager->RegisterUniformBuffer(multiBuffer->GetBuffer(i));
            ids.push_back(id);
        }
        auto p         = MakeRef<RHI::RhiDescHandleLegacy>();
        p->ids         = ids;
        p->activeFrame = m_swapChain->GetCurrentImageIndex();
        m_implDetails->m_bindlessIdRefs.push_back(p);
        return p;
    }

    IFRIT_APIDECL Ref<RHI::RhiDescHandleLegacy> RhiVulkanBackend::RegisterStorageBufferShared(
        RHI::RhiMultiBuffer* buffer)
    {
        // TODO
        Vec<u32> ids;
        auto     descriptorManager = m_implDetails->m_descriptorManager.get();
        auto     multiBuffer       = CheckedCast<MultiBuffer>(buffer);
        auto     numBackbuffers    = m_swapChain->GetNumBackbuffers();
        for (u32 i = 0; i < numBackbuffers; i++)
        {
            auto                     id = descriptorManager->RegisterStorageBuffer(multiBuffer->GetBuffer(i));
            RHI::RhiDescriptorHandle handle(RHI::RhiDescriptorHeapType::StorageBuffer, id);
            multiBuffer->GetBuffer(i)->SetDescriptorHandle(handle);
            ids.push_back(id);
        }
        auto p         = MakeRef<RHI::RhiDescHandleLegacy>();
        p->ids         = ids;
        p->activeFrame = m_swapChain->GetCurrentImageIndex();
        m_implDetails->m_bindlessIdRefs.push_back(p);
        return p;
    }

    IFRIT_APIDECL RHI::RhiSRVDesc RhiVulkanBackend::GetSRVDescriptor(
        RHI::RhiTexture* texture, RHI::RhiImageSubResource subResource)
    {
        auto dm  = m_implDetails->m_descriptorManager.get();
        auto tex = CheckedCast<SingleDeviceImage>(texture);
        auto p   = dm->RegisterSampledImage(tex, subResource);
        return p;
    }
    IFRIT_APIDECL RHI::RhiUAVDesc RhiVulkanBackend::GetUAVDescriptor(
        RHI::RhiTexture* texture, RHI::RhiImageSubResource subResource)
    {
        bool isMainLayer = (subResource.mipCount == 1) && (subResource.layerCount == 1) && (subResource.arrayLayer == 0)
            && (subResource.mipLevel == 0);
        if (texture->GetDescId(true) != ~0u && isMainLayer)
        {
            return texture->GetDescId();
        }
        auto dm  = m_implDetails->m_descriptorManager.get();
        auto tex = CheckedCast<SingleDeviceImage>(texture);
        auto p   = dm->RegisterStorageImage(tex, subResource);
        if (isMainLayer)
            texture->SetDescriptorHandle(RHI::RhiDescriptorHandle(RHI::RhiDescriptorHeapType::StorageImage, p));
        return p;
    }
    IFRIT_APIDECL RHI::RhiSRVDesc RhiVulkanBackend::GetSRVDescriptor(RHI::RhiTexture* texture)
    {
        return GetSRVDescriptor(texture, { 0, 0, 1, 1 });
    }
    IFRIT_APIDECL RHI::RhiUAVDesc RhiVulkanBackend::GetUAVDescriptor(RHI::RhiTexture* texture)
    {
        return GetUAVDescriptor(texture, { 0, 0, 1, 1 });
    }
    IFRIT_APIDECL RHI::RhiSRVDesc RhiVulkanBackend::GetSRVDescriptor(RHI::RhiBuffer* buffer)
    {
        // vulkan seems to not support buffer SRV, so we just return UAV
        // update 250710:
        auto dm  = m_implDetails->m_descriptorManager.get();
        auto buf = CheckedCast<SingleBuffer>(buffer);
        auto p   = dm->RegisterStorageBufferSRV(buf);
        return p;
    }
    IFRIT_APIDECL RHI::RhiCBVDesc RhiVulkanBackend::GetCBVDescriptor(RHI::RhiBuffer* buffer)
    {

        auto dm  = m_implDetails->m_descriptorManager.get();
        auto buf = CheckedCast<SingleBuffer>(buffer);
        auto p   = dm->RegisterUniformBuffer(buf);
        return p;
    }
    IFRIT_APIDECL RHI::RhiUAVDesc RhiVulkanBackend::GetUAVDescriptor(RHI::RhiBuffer* buffer)
    {
        if (buffer->GetDescId(true) != ~0u)
        {
            return buffer->GetDescId();
        }
        auto dm  = m_implDetails->m_descriptorManager.get();
        auto buf = CheckedCast<SingleBuffer>(buffer);
        auto p   = dm->RegisterStorageBuffer(buf);
        buffer->SetDescriptorHandle(RHI::RhiDescriptorHandle(RHI::RhiDescriptorHeapType::StorageBuffer, p));
        return p;
    }

    IFRIT_APIDECL Ref<RHI::RhiVertexBufferView> RhiVulkanBackend::CreateVertexBufferView()
    {
        auto view = MakeRef<VertexBufferDescriptor>();
        return view;
    }

    IFRIT_APIDECL Ref<RHI::RhiVertexBufferView> RhiVulkanBackend::GetFullScreenQuadVertexBufferView() const
    {
        return m_implDetails->m_fullScreenQuadVertexBufferDescriptor;
    }

    IFRIT_APIDECL Owner<RHI::FSR2::RhiFsr2Processor> RhiVulkanBackend::CreateFsr2Processor()
    {
        auto ctx = CheckedCast<EngineContext>(m_device.get());
        return MakeOwner<VulkanAdapter::FSR2::FSR2Processor>(ctx);
    }

    IFRIT_APIDECL void RhiVulkanBackend::SetCacheDirectory(const std::string& dir)
    {
        auto engineContext = CheckedCast<EngineContext>(m_device.get());
        engineContext->SetCacheDirectory(dir);
    }
    IFRIT_APIDECL std::string RhiVulkanBackend::GetCacheDir() const
    {
        auto engineContext = CheckedCast<EngineContext>(m_device.get());
        return engineContext->GetCacheDir();
    }

    IFRIT_APIDECL Owner<RHI::RhiBackend> RhiVulkanBackendBuilder::CreateBackend(const RHI::RhiInitializeArguments& args)
    {
        return MakeOwner<RhiVulkanBackend>(args);
    }

    IFRIT_APIDECL void GetRhiBackendBuilder_Vulkan(Owner<RHI::RhiBackendFactory>& ptr)
    {
        ptr = MakeOwner<RhiVulkanBackendBuilder>();
    }
} // namespace Ifrit::RHI::VulkanAdapter