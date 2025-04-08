
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

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <memory>

namespace Ifrit::Graphics::VulkanGraphics
{

    struct RhiVulkanBackendImplDetails;
    class IFRIT_APIDECL RhiVulkanBackend : public Rhi::RhiBackend
    {
    protected:
        // Note that Destructor order matters here
        // https://isocpp.org/wiki/faq/dtors#order-dtors-for-members
        Uref<Rhi::RhiDevice>         m_device;
        Uref<Rhi::RhiSwapchain>      m_swapChain;
        RhiVulkanBackendImplDetails* m_implDetails;

    public:
        RhiVulkanBackend(const Rhi::RhiInitializeArguments& args);
        ~RhiVulkanBackend();

        void                     WaitDeviceIdle() override;
        Ref<Rhi::RhiDeviceTimer> CreateDeviceTimer() override;
        Rhi::RhiBufferRef        CreateBuffer(
                   const String& name, u32 size, u32 usage, bool hostVisible, bool addUAV) const override;
        Rhi::RhiBufferRef CreateBufferDevice(const String& name, u32 size, u32 usage, bool addUAV) const override;
        Ref<Rhi::RhiMultiBuffer>        CreateBufferCoherent(u32 size, u32 usage, u32 numCopies = ~0u) const override;
        Ref<Rhi::RhiStagedSingleBuffer> CreateStagedSingleBuffer(Rhi::RhiBuffer* target) override;
        Rhi::RhiBufferRef               GetFullScreenQuadVertexBuffer() const override;

        // Command execution
        Rhi::RhiQueue*                  GetQueue(Rhi::RhiQueueCapability req) override;

        // Shader
        Rhi::RhiShader*    CreateShader(const String& name, const std::vector<char>& code, const String& entry,
               Rhi::RhiShaderStage stage, Rhi::RhiShaderSourceType sourceType) override;

        // Texture
        Rhi::RhiTextureRef CreateTexture2D(const String& name, u32 width, u32 height, Rhi::RhiImageFormat format,
            u32 extraFlags, bool addUAV) override;
        Rhi::RhiTextureRef CreateTexture2DMsaa(const String& name, u32 width, u32 height, Rhi::RhiImageFormat format,
            u32 extraFlags, u32 samples) override;
        Rhi::RhiTextureRef CreateDepthTexture(const String& name, u32 width, u32 height, bool addUAV) override;
        Rhi::RhiTextureRef CreateTexture3D(const String& name, u32 width, u32 height, u32 depth,
            Rhi::RhiImageFormat format, u32 extraFlags, bool addUAV) override;

        Rhi::RhiTextureRef CreateMipMapTexture(const String& name, u32 width, u32 height, u32 mips,
            Rhi::RhiImageFormat format, u32 extraFlags, bool addUAV) override;

        Rhi::RhiSamplerRef CreateSampler(
            Rhi::RhiSamplerFilter filter, Rhi::RhiSamplerWrapMode addressMode, bool addBinding) override;

        // Pass
        Rhi::RhiComputePass*                   CreateComputePass() override;
        Rhi::RhiGraphicsPass*                  CreateGraphicsPass() override;

        Uref<Rhi::RhiComputePass>              CreateComputePass2() override;
        Uref<Rhi::RhiGraphicsPass>             CreateGraphicsPass2() override;

        // Swapchain
        Rhi::RhiTexture*                       GetSwapchainImage() override;
        void                                   BeginFrame() override;
        void                                   EndFrame() override;
        Uref<Rhi::RhiTaskSubmission>           GetSwapchainFrameReadyEventHandler() override;
        Uref<Rhi::RhiTaskSubmission>           GetSwapchainRenderDoneEventHandler() override;

        // Descriptor
        virtual Rhi::RhiBindlessDescriptorRef* CreateBindlessDescriptorRef() override;
        virtual Ref<Rhi::RhiDescHandleLegacy>  RegisterUniformBuffer(Rhi::RhiMultiBuffer* buffer) override;
        virtual Ref<Rhi::RhiDescHandleLegacy>  RegisterStorageBufferShared(Rhi::RhiMultiBuffer* buffer) override;
        virtual Ref<Rhi::RhiDescHandleLegacy>  RegisterCombinedImageSampler(
             Rhi::RhiTexture* texture, Rhi::RhiSampler* sampler) override;

        // Descriptor, refactored
        virtual Rhi::RhiSRVDesc GetSRVDescriptor(
            Rhi::RhiTexture* texture, Rhi::RhiImageSubResource subResource) override;
        virtual Rhi::RhiUAVDesc GetUAVDescriptor(
            Rhi::RhiTexture* texture, Rhi::RhiImageSubResource subResource) override;
        virtual Rhi::RhiSRVDesc                     GetSRVDescriptor(Rhi::RhiTexture* texture) override;
        virtual Rhi::RhiUAVDesc                     GetUAVDescriptor(Rhi::RhiTexture* texture) override;
        virtual Rhi::RhiSRVDesc                     GetSRVDescriptor(Rhi::RhiBuffer* buffer) override;
        virtual Rhi::RhiUAVDesc                     GetUAVDescriptor(Rhi::RhiBuffer* buffer) override;

        // Render targets
        virtual Ref<Rhi::RhiColorAttachment>        CreateRenderTarget(Rhi::RhiTexture* renderTarget,
                   Rhi::RhiClearValue clearValue, Rhi::RhiRenderTargetLoadOp loadOp, u32 mips, u32 layers) override;

        virtual Ref<Rhi::RhiDepthStencilAttachment> CreateRenderTargetDepthStencil(
            Rhi::RhiTexture* renderTarget, Rhi::RhiClearValue clearValue, Rhi::RhiRenderTargetLoadOp loadOp) override;

        virtual Ref<Rhi::RhiRenderTargets>         CreateRenderTargets() override;

        // Vertex buffer
        virtual Ref<Rhi::RhiVertexBufferView>      CreateVertexBufferView() override;
        virtual Ref<Rhi::RhiVertexBufferView>      GetFullScreenQuadVertexBufferView() const override;

        // Cache
        virtual void                               SetCacheDirectory(const String& dir) override;
        virtual String                             GetCacheDir() const override;

        // Extension
        virtual Uref<Rhi::FSR2::RhiFsr2Processor>  CreateFsr2Processor() override;

        // Raytracing
        virtual Uref<Rhi::RhiRTInstance>           CreateTLAS() { return nullptr; }
        virtual Uref<Rhi::RhiRTScene>              CreateBLAS() { return nullptr; }
        virtual Uref<Rhi::RhiRTShaderBindingTable> CreateShaderBindingTable() { return nullptr; }

        virtual Uref<Rhi::RhiRTPass>               CreateRaytracingPass() { return nullptr; }
    };

    class IFRIT_APIDECL RhiVulkanBackendBuilder : public Rhi::RhiBackendFactory, public NonCopyable
    {
    public:
        Uref<Rhi::RhiBackend> CreateBackend(const Rhi::RhiInitializeArguments& args) override;
    };
} // namespace Ifrit::Graphics::VulkanGraphics