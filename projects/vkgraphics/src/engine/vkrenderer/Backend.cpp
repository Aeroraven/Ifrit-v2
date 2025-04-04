
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
#include "ifrit/core/typing/Util.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderGraph.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderTargets.h"
#include "ifrit/vkgraphics/engine/vkrenderer/StagedMemoryResource.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Timer.h"

#include "ifrit/core/logging/Logging.h"
#include "ifrit/vkgraphics/engine/fsr2extension/FSR2Processor.h"

#include "ifrit/core/algo/ConcurrentVector.h"
using namespace Ifrit;

namespace Ifrit::Graphics::VulkanGraphics
{
    inline VkFormat toVkFormat(Rhi::RhiImageFormat format) { return static_cast<VkFormat>(format); }

    struct RhiVulkanBackendImplDetails : public NonCopyable
    {
        Uref<CommandExecutor>                       m_commandExecutor;
        Uref<DescriptorManager>                     m_descriptorManager;
        Uref<ResourceManager>                       m_resourceManager;
        Vec<Uref<StagedSingleBuffer>>               m_stagedSingleBuffer;
        Uref<PipelineCache>                         m_pipelineCache;

        Uref<RegisteredResourceMapper>              m_mapper;

        // managed passes
        Vec<Uref<ComputePass>>                      m_computePasses;
        Vec<Uref<GraphicsPass>>                     m_graphicsPasses;
        Vec<Uref<DescriptorBindlessIndices>>        m_bindlessIndices;

        // managed descriptors
        Vec<Ref<Rhi::RhiDescHandleLegacy>>          m_bindlessIdRefs;

        // some utility buffers
        Rhi::RhiBufferRef                           m_fullScreenQuadVertexBuffer;
        Ref<VertexBufferDescriptor>                 m_fullScreenQuadVertexBufferDescriptor;

        // timers
        Vec<Ref<DeviceTimer>>                       m_deviceTimers;

        RConcurrentGrowthVector<Uref<ShaderModule>> m_shaderModule;
    };

    IFRIT_APIDECL
    RhiVulkanBackend::RhiVulkanBackend(const Rhi::RhiInitializeArguments& args)
    {
        m_device                           = std::make_unique<EngineContext>(args);
        auto engineContext                 = CheckedCast<EngineContext>(m_device.get());
        m_swapChain                        = std::make_unique<Swapchain>(engineContext);
        auto swapchain                     = CheckedCast<Swapchain>(m_swapChain.get());
        m_implDetails                      = new RhiVulkanBackendImplDetails();
        m_implDetails->m_descriptorManager = std::make_unique<DescriptorManager>(engineContext);
        m_implDetails->m_resourceManager   = std::make_unique<ResourceManager>(engineContext);
        m_implDetails->m_commandExecutor   = std::make_unique<CommandExecutor>(
            engineContext, swapchain, m_implDetails->m_descriptorManager.get(), m_implDetails->m_resourceManager.get());
        m_implDetails->m_pipelineCache = std::make_unique<PipelineCache>(engineContext);
        m_implDetails->m_mapper        = std::make_unique<RegisteredResourceMapper>();
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
        transferQueue->RunSyncCommand([&](const Rhi::RhiCommandList* cmd) {
            float data[] = {
                0.0f, 0.0f, //
                4.0f, 0.0f, //
                0.0f, 4.0f, //
            };
            stagedQuadBuffer.CmdCopyToDevice(cmd, data, sizeof(data), 0);
        });
        m_implDetails->m_fullScreenQuadVertexBufferDescriptor = std::make_shared<VertexBufferDescriptor>();
        m_implDetails->m_fullScreenQuadVertexBufferDescriptor->AddBinding(
            { 0 }, { Rhi::RhiImageFormat::RhiImgFmt_R32G32_SFLOAT }, { 0 }, 2 * sizeof(float));
    }

    IFRIT_APIDECL void RhiVulkanBackend::WaitDeviceIdle()
    {
        auto p = CheckedCast<EngineContext>(m_device.get());
        p->WaitIdle();
    }

    IFRIT_APIDECL Ref<Rhi::RhiDeviceTimer> RhiVulkanBackend::CreateDeviceTimer()
    {
        auto swapchain        = CheckedCast<Swapchain>(m_swapChain.get());
        auto numFrameInFlight = swapchain->GetNumBackbuffers();
        auto p = std::make_shared<DeviceTimer>(CheckedCast<EngineContext>(m_device.get()), numFrameInFlight);
        m_implDetails->m_deviceTimers.push_back(p);
        return p;
    }

    IFRIT_APIDECL Rhi::RhiBufferRef RhiVulkanBackend::CreateBuffer(
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
            p->SetDescriptorHandle(Rhi::RhiDescriptorHandle(Rhi::RhiDescriptorHeapType::StorageBuffer, id));
        }
        return p;
    }

    IFRIT_APIDECL Rhi::RhiBufferRef RhiVulkanBackend::GetFullScreenQuadVertexBuffer() const
    {
        return m_implDetails->m_fullScreenQuadVertexBuffer;
    }

    IFRIT_APIDECL Rhi::RhiBufferRef RhiVulkanBackend::CreateBufferDevice(
        const String& name, u32 size, u32 usage, bool addUAV) const
    {
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
            p->SetDescriptorHandle(Rhi::RhiDescriptorHandle(Rhi::RhiDescriptorHeapType::StorageBuffer, id));
        }
        return p;
    }
    IFRIT_APIDECL Ref<Rhi::RhiMultiBuffer> RhiVulkanBackend::CreateBufferCoherent(
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

    IFRIT_APIDECL Ref<Rhi::RhiStagedSingleBuffer> RhiVulkanBackend::CreateStagedSingleBuffer(Rhi::RhiBuffer* target)
    {
        // TODO: release memory, (not managed)
        auto buffer        = CheckedCast<SingleBuffer>(target);
        auto engineContext = CheckedCast<EngineContext>(m_device.get());
        auto ptr           = std::make_shared<StagedSingleBuffer>(engineContext, buffer);
        return ptr;
    }

    IFRIT_APIDECL Rhi::RhiQueue* RhiVulkanBackend::GetQueue(Rhi::RhiQueueCapability req)
    {
        QueueRequirement reqs;
        if (req == Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT)
        {
            reqs = QueueRequirement::Graphics;
        }
        else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT)
        {
            reqs = QueueRequirement::Compute;
        }
        else if (req == Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT)
        {
            reqs = QueueRequirement::Transfer;
        }
        else if (req
            == (Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT | Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT))
        {
            reqs = QueueRequirement::Graphics_Compute;
        }
        else if (req
            == (Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT | Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT))
        {
            reqs = QueueRequirement::Graphics_Transfer;
        }
        else if (req
            == (Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT | Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT))
        {
            reqs = QueueRequirement::Compute_Transfer;
        }
        else if (req
            == (Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT | Rhi::RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT
                | Rhi::RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT))
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

    IFRIT_APIDECL Rhi::RhiShader* RhiVulkanBackend::CreateShader(const std::string& name, const Vec<char>& code,
        const std::string& entry, Rhi::RhiShaderStage stage, Rhi::RhiShaderSourceType sourceType)
    {
        ShaderModuleCI ci{};
        ci.code           = code;
        ci.entryPoint     = entry;
        ci.stage          = stage;
        ci.sourceType     = sourceType;
        ci.fileName       = name;
        auto shaderModule = std::make_unique<ShaderModule>(CheckedCast<EngineContext>(m_device.get()), ci);
        auto ptr          = shaderModule.get();
        m_implDetails->m_shaderModule.PushBack(std::move(shaderModule));
        return ptr;
    }

    IFRIT_APIDECL Rhi::RhiTextureRef RhiVulkanBackend::CreateTexture2D(
        const String& name, u32 width, u32 height, Rhi::RhiImageFormat format, u32 extraFlags, bool addUAV)
    {
        auto p = m_implDetails->m_resourceManager->CreateTexture2DDeviceUnmanaged(
            width, height, toVkFormat(format), extraFlags);
        p->SetDebugName(name);
        if (addUAV)
        {
            auto texture           = CheckedCast<SingleDeviceImage>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageImage(texture, { 0, 0, 1, 1 });
            p->SetDescriptorHandle(Rhi::RhiDescriptorHandle(Rhi::RhiDescriptorHeapType::StorageImage, id));
        }
        return p;
    }

    IFRIT_APIDECL Rhi::RhiTextureRef RhiVulkanBackend::CreateDepthTexture(
        const String& name, u32 width, u32 height, bool addUAV)
    {
        auto p = m_implDetails->m_resourceManager->CreateDepthAttachment(width, height);
        p->SetDebugName(name);
        if (addUAV)
        {
            auto texture           = CheckedCast<SingleDeviceImage>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageImage(texture, { 0, 0, 1, 1 });
            p->SetDescriptorHandle(Rhi::RhiDescriptorHandle(Rhi::RhiDescriptorHeapType::StorageImage, id));
        }
        return p;
    }

    IFRIT_APIDECL Rhi::RhiTextureRef RhiVulkanBackend::CreateTexture3D(
        const String& name, u32 width, u32 height, u32 depth, Rhi::RhiImageFormat format, u32 extraFlags, bool addUAV)
    {
        auto p =
            m_implDetails->m_resourceManager->CreateTexture3D(width, height, depth, toVkFormat(format), extraFlags);
        p->SetDebugName(name);
        if (addUAV)
        {
            auto texture           = CheckedCast<SingleDeviceImage>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageImage(texture, { 0, 0, 1, 1 });
            p->SetDescriptorHandle(Rhi::RhiDescriptorHandle(Rhi::RhiDescriptorHeapType::StorageImage, id));
        }
        return p;
    };

    IFRIT_APIDECL Rhi::RhiTextureRef RhiVulkanBackend::CreateMipMapTexture(
        const String& name, u32 width, u32 height, u32 mips, Rhi::RhiImageFormat format, u32 extraFlags, bool addUAV)
    {
        auto p =
            m_implDetails->m_resourceManager->createMipTexture(width, height, mips, toVkFormat(format), extraFlags);
        p->SetDebugName(name);
        if (addUAV)
        {
            auto texture           = CheckedCast<SingleDeviceImage>(p.get());
            auto descriptorManager = (m_implDetails->m_descriptorManager.get());
            auto id                = descriptorManager->RegisterStorageImage(texture, { 0, 0, 1, 1 });
            p->SetDescriptorHandle(Rhi::RhiDescriptorHandle(Rhi::RhiDescriptorHeapType::StorageImage, id));
        }
        return p;
    }

    IFRIT_APIDECL Rhi::RhiSamplerRef RhiVulkanBackend::CreateTrivialSampler()
    {
        return m_implDetails->m_resourceManager->CreateTrivialRenderTargetSampler();
    }

    IFRIT_APIDECL Rhi::RhiSamplerRef RhiVulkanBackend::CreateTrivialBilinearSampler(bool repeat)
    {
        return m_implDetails->m_resourceManager->CreateTrivialBilinearSampler(repeat);
    }

    IFRIT_APIDECL Rhi::RhiSamplerRef RhiVulkanBackend::CreateTrivialNearestSampler(bool repeat)
    {
        return m_implDetails->m_resourceManager->CreateTrivialNearestSampler(repeat);
    }

    // Deprecating
    IFRIT_APIDECL Rhi::RhiComputePass* RhiVulkanBackend::CreateComputePass()
    {
        auto pass = std::make_unique<ComputePass>(CheckedCast<EngineContext>(m_device.get()),
            m_implDetails->m_pipelineCache.get(), m_implDetails->m_descriptorManager.get(),
            m_implDetails->m_mapper.get());
        auto ptr  = pass.get();
        ptr->SetDefaultNumMultiBuffers(m_swapChain->GetNumBackbuffers());
        m_implDetails->m_computePasses.push_back(std::move(pass));
        return ptr;
    }

    // Deprecating
    IFRIT_APIDECL Rhi::RhiGraphicsPass* RhiVulkanBackend::CreateGraphicsPass()
    {
        auto pass = std::make_unique<GraphicsPass>(CheckedCast<EngineContext>(m_device.get()),
            m_implDetails->m_pipelineCache.get(), m_implDetails->m_descriptorManager.get(),
            m_implDetails->m_mapper.get());
        auto ptr  = pass.get();
        ptr->SetDefaultNumMultiBuffers(m_swapChain->GetNumBackbuffers());
        m_implDetails->m_graphicsPasses.push_back(std::move(pass));
        return ptr;
    }

    IFRIT_APIDECL Uref<Rhi::RhiComputePass> RhiVulkanBackend::CreateComputePass2()
    {
        auto pass = std::make_unique<ComputePass>(CheckedCast<EngineContext>(m_device.get()),
            m_implDetails->m_pipelineCache.get(), m_implDetails->m_descriptorManager.get(),
            m_implDetails->m_mapper.get());
        auto ptr  = pass.get();
        ptr->SetDefaultNumMultiBuffers(m_swapChain->GetNumBackbuffers());
        return pass;
    }
    IFRIT_APIDECL Uref<Rhi::RhiGraphicsPass> RhiVulkanBackend::CreateGraphicsPass2()
    {
        auto pass = std::make_unique<GraphicsPass>(CheckedCast<EngineContext>(m_device.get()),
            m_implDetails->m_pipelineCache.get(), m_implDetails->m_descriptorManager.get(),
            m_implDetails->m_mapper.get());
        auto ptr  = pass.get();
        ptr->SetDefaultNumMultiBuffers(m_swapChain->GetNumBackbuffers());
        return pass;
    }

    IFRIT_APIDECL Rhi::RhiTexture* RhiVulkanBackend::GetSwapchainImage()
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
        for (auto& pass : m_implDetails->m_computePasses)
        {
            pass->SetActiveFrame(m_swapChain->GetCurrentImageIndex());
        }
        for (auto& pass : m_implDetails->m_graphicsPasses)
        {
            pass->SetActiveFrame(m_swapChain->GetCurrentImageIndex());
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
        if (nums > 0)
        {
            iDebug("Deleted {} resources", nums);
        }
    }
    IFRIT_APIDECL void RhiVulkanBackend::EndFrame() { m_implDetails->m_commandExecutor->EndFrame(); }
    IFRIT_APIDECL Uref<Rhi::RhiTaskSubmission> RhiVulkanBackend::GetSwapchainFrameReadyEventHandler()
    {
        auto                  swapchain = CheckedCast<Swapchain>(m_swapChain.get());
        auto                  sema      = swapchain->GetImageAvailableSemaphoreCurrentFrame();
        TimelineSemaphoreWait wait;
        wait.m_isSwapchainSemaphore = true;
        wait.m_semaphore            = sema;
        return std::make_unique<TimelineSemaphoreWait>(wait);
    }
    IFRIT_APIDECL Uref<Rhi::RhiTaskSubmission> RhiVulkanBackend::GetSwapchainRenderDoneEventHandler()
    {
        auto                  swapchain = CheckedCast<Swapchain>(m_swapChain.get());
        auto                  sema      = swapchain->GetRenderingFinishSemaphoreCurrentFrame();
        auto                  fence     = swapchain->GetCurrentFrameFence();
        TimelineSemaphoreWait wait;
        wait.m_isSwapchainSemaphore = true;
        wait.m_semaphore            = sema;
        wait.m_fence                = fence;
        return std::make_unique<TimelineSemaphoreWait>(wait);
    }

    Ref<Rhi::RhiColorAttachment> RhiVulkanBackend::CreateRenderTarget(Rhi::RhiTexture* renderTarget,
        Rhi::RhiClearValue clearValue, Rhi::RhiRenderTargetLoadOp loadOp, u32 mips, u32 layers)
    {
        auto attachment = std::make_shared<ColorAttachment>(renderTarget, clearValue, loadOp, mips, layers);
        return attachment;
    }

    Ref<Rhi::RhiDepthStencilAttachment> RhiVulkanBackend::CreateRenderTargetDepthStencil(
        Rhi::RhiTexture* renderTarget, Rhi::RhiClearValue clearValue, Rhi::RhiRenderTargetLoadOp loadOp)
    {
        auto attachment = std::make_shared<DepthStencilAttachment>(renderTarget, clearValue, loadOp);
        return attachment;
    }

    Ref<Rhi::RhiRenderTargets> RhiVulkanBackend::CreateRenderTargets()
    {
        auto ctx = CheckedCast<EngineContext>(m_device.get());
        return std::make_shared<RenderTargets>(ctx);
    }

    IFRIT_APIDECL RhiVulkanBackend::~RhiVulkanBackend() { delete m_implDetails; }

    IFRIT_APIDECL Rhi::RhiBindlessDescriptorRef* RhiVulkanBackend::createBindlessDescriptorRef()
    {
        auto ref = std::make_unique<DescriptorBindlessIndices>(CheckedCast<EngineContext>(m_device.get()),
            m_implDetails->m_descriptorManager.get(), m_swapChain->GetNumBackbuffers());
        auto ptr = ref.get();
        m_implDetails->m_bindlessIndices.push_back(std::move(ref));
        return ptr;
    }

    IFRIT_APIDECL Ref<Rhi::RhiDescHandleLegacy> RhiVulkanBackend::RegisterUniformBuffer(Rhi::RhiMultiBuffer* buffer)
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
        auto p         = std::make_shared<Rhi::RhiDescHandleLegacy>();
        p->ids         = ids;
        p->activeFrame = m_swapChain->GetCurrentImageIndex();
        m_implDetails->m_bindlessIdRefs.push_back(p);
        return p;
    }

    Ref<Rhi::RhiDescHandleLegacy> RhiVulkanBackend::RegisterCombinedImageSampler(
        Rhi::RhiTexture* texture, Rhi::RhiSampler* sampler)
    {
        auto descriptorManager = m_implDetails->m_descriptorManager.get();
        auto tex               = CheckedCast<SingleDeviceImage>(texture);
        auto sam               = CheckedCast<Sampler>(sampler);
        auto id                = descriptorManager->RegisterCombinedImageSampler(tex, sam);
        auto p                 = std::make_shared<Rhi::RhiDescHandleLegacy>();
        p->ids.push_back(id);
        p->activeFrame = 0;
        return p;
    }

    IFRIT_APIDECL Ref<Rhi::RhiDescHandleLegacy> RhiVulkanBackend::RegisterUAVImage2(
        Rhi::RhiTexture* texture, Rhi::RhiImageSubResource subResource)
    {
        auto descriptorManager = m_implDetails->m_descriptorManager.get();
        auto tex               = CheckedCast<SingleDeviceImage>(texture);
        auto id                = descriptorManager->RegisterStorageImage(tex, subResource);
        auto p                 = std::make_shared<Rhi::RhiDescHandleLegacy>();
        p->ids.push_back(id);
        p->activeFrame = 0;
        return p;
    }

    IFRIT_APIDECL Ref<Rhi::RhiDescHandleLegacy> RhiVulkanBackend::RegisterStorageBufferShared(
        Rhi::RhiMultiBuffer* buffer)
    {
        // TODO
        Vec<u32> ids;
        auto     descriptorManager = m_implDetails->m_descriptorManager.get();
        auto     multiBuffer       = CheckedCast<MultiBuffer>(buffer);
        auto     numBackbuffers    = m_swapChain->GetNumBackbuffers();
        for (u32 i = 0; i < numBackbuffers; i++)
        {
            auto id = descriptorManager->RegisterStorageBuffer(multiBuffer->GetBuffer(i));
            ids.push_back(id);
        }
        auto p         = std::make_shared<Rhi::RhiDescHandleLegacy>();
        p->ids         = ids;
        p->activeFrame = m_swapChain->GetCurrentImageIndex();
        m_implDetails->m_bindlessIdRefs.push_back(p);
        return p;
    }

    IFRIT_APIDECL Ref<Rhi::RhiVertexBufferView> RhiVulkanBackend::CreateVertexBufferView()
    {
        auto view = std::make_shared<VertexBufferDescriptor>();
        return view;
    }

    IFRIT_APIDECL Ref<Rhi::RhiVertexBufferView> RhiVulkanBackend::GetFullScreenQuadVertexBufferView() const
    {
        return m_implDetails->m_fullScreenQuadVertexBufferDescriptor;
    }

    IFRIT_APIDECL Uref<Rhi::FSR2::RhiFsr2Processor> RhiVulkanBackend::CreateFsr2Processor()
    {
        auto ctx = CheckedCast<EngineContext>(m_device.get());
        return std::make_unique<VulkanGraphics::FSR2::FSR2Processor>(ctx);
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

    IFRIT_APIDECL Uref<Rhi::RhiBackend> RhiVulkanBackendBuilder::CreateBackend(const Rhi::RhiInitializeArguments& args)
    {
        return std::make_unique<RhiVulkanBackend>(args);
    }

    IFRIT_APIDECL void GetRhiBackendBuilder_Vulkan(Uref<Rhi::RhiBackendFactory>& ptr)
    {
        ptr = std::make_unique<RhiVulkanBackendBuilder>();
    }
} // namespace Ifrit::Graphics::VulkanGraphics