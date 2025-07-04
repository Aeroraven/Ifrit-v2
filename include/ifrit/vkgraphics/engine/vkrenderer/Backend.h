
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
#include "ifrit/vkgraphics/common/Pch.h"

namespace Ifrit::RHI::VulkanAdapter
{

    struct RhiVulkanBackendImplDetails;
    class IFRIT_APIDECL RhiVulkanBackend : public RHI::RhiBackend
    {
    protected:
        // Note that Destructor order matters here
        // https://isocpp.org/wiki/faq/dtors#order-dtors-for-members
        Owner<RHI::RhiDevice>        m_device;
        Owner<RHI::RhiSwapchain>     m_swapChain;
        RhiVulkanBackendImplDetails* m_implDetails;
        RHI::RhiCapabilityList       m_Capability = {};

    public:
        RhiVulkanBackend(const RHI::RhiInitializeArguments& args);
        ~RhiVulkanBackend();

        void                     WaitDeviceIdle() override;
        RHI::RhiCapabilityList   GetCapabilities() const override;
        Ref<RHI::RhiDeviceTimer> CreateDeviceTimer() override;
        RHI::RhiBufferRef        CreateBuffer(
                   const String& name, u32 size, u32 usage, bool hostVisible, bool addUAV) const override;
        RHI::RhiBufferRef CreateBufferDevice(const String& name, u32 size, u32 usage, bool addUAV) const override;
        Ref<RHI::RhiMultiBuffer>        CreateBufferCoherent(u32 size, u32 usage, u32 numCopies = ~0u) const override;
        Ref<RHI::RhiStagedSingleBuffer> CreateStagedSingleBuffer(RHI::RhiBuffer* target) override;
        RHI::RhiBufferRef               GetFullScreenQuadVertexBuffer() const override;

        // Command execution
        RHI::RhiQueue*                  GetQueue(RHI::RhiQueueCapability req) override;

        // Shader
        Ref<RHI::RhiShaderCollection>   CreateShader(const String& name, const Vec<char>& code, const String& entry,
              RHI::RhiShaderStage stage, RHI::RhiShaderSourceType sourceType) override;

        // Texture
        RHI::RhiTextureRef CreateTexture2D(const String& name, u32 width, u32 height, RHI::RhiImageFormat format,
            u32 extraFlags, bool addUAV) override;
        RHI::RhiTextureRef CreateTexture2DMsaa(const String& name, u32 width, u32 height, RHI::RhiImageFormat format,
            u32 extraFlags, u32 samples) override;
        RHI::RhiTextureRef CreateDepthTexture(const String& name, u32 width, u32 height, bool addUAV) override;
        RHI::RhiTextureRef CreateTexture3D(const String& name, u32 width, u32 height, u32 depth,
            RHI::RhiImageFormat format, u32 extraFlags, bool addUAV) override;

        RHI::RhiTextureRef CreateMipMapTexture(const String& name, u32 width, u32 height, u32 mips,
            RHI::RhiImageFormat format, u32 extraFlags, bool addUAV) override;

        RHI::RhiSamplerRef CreateSampler(
            RHI::RhiSamplerFilter filter, RHI::RhiSamplerWrapMode addressMode, bool addBinding) override;

        // Pass
        RHI::RhiComputePass*                   CreateComputePass() override;
        RHI::RhiGraphicsPass*                  CreateGraphicsPass() override;

        Owner<RHI::RhiComputePass>             CreateComputePass2() override;
        Owner<RHI::RhiGraphicsPass>            CreateGraphicsPass2() override;

        // Swapchain
        RHI::RhiTexture*                       GetSwapchainImage() override;
        void                                   BeginFrame() override;
        void                                   EndFrame() override;
        Owner<RHI::RhiTaskSubmission>          GetSwapchainFrameReadyEventHandler() override;
        Owner<RHI::RhiTaskSubmission>          GetSwapchainRenderDoneEventHandler() override;

        // Descriptor
        virtual RHI::RhiBindlessDescriptorRef* CreateBindlessDescriptorRef() override;
        virtual Ref<RHI::RhiDescHandleLegacy>  RegisterUniformBuffer(RHI::RhiMultiBuffer* buffer) override;
        virtual Ref<RHI::RhiDescHandleLegacy>  RegisterStorageBufferShared(RHI::RhiMultiBuffer* buffer) override;
        // Descriptor, refactored
        virtual RHI::RhiSRVDesc                GetSRVDescriptor(
                           RHI::RhiTexture* texture, RHI::RhiImageSubResource subResource) override;
        virtual RHI::RhiUAVDesc GetUAVDescriptor(
            RHI::RhiTexture* texture, RHI::RhiImageSubResource subResource) override;
        virtual RHI::RhiSRVDesc                     GetSRVDescriptor(RHI::RhiTexture* texture) override;
        virtual RHI::RhiUAVDesc                     GetUAVDescriptor(RHI::RhiTexture* texture) override;
        virtual RHI::RhiSRVDesc                     GetSRVDescriptor(RHI::RhiBuffer* buffer) override;
        virtual RHI::RhiUAVDesc                     GetUAVDescriptor(RHI::RhiBuffer* buffer) override;

        virtual RHI::RhiCBVDesc                     GetCBVDescriptor(RHI::RhiBuffer* buffer) override;

        // Render targets
        virtual Ref<RHI::RhiColorAttachment>        CreateRenderTarget(RHI::RhiTexture* renderTarget,
                   RHI::RhiClearValue2 clearValue, RHI::RhiRenderTargetLoadOp loadOp, u32 mips, u32 layers) override;

        virtual Ref<RHI::RhiDepthStencilAttachment> CreateRenderTargetDepthStencil(
            RHI::RhiTexture* renderTarget, RHI::RhiClearValue2 clearValue, RHI::RhiRenderTargetLoadOp loadOp) override;

        virtual Ref<RHI::RhiRenderTargets>          CreateRenderTargets() override;

        // Vertex buffer
        virtual Ref<RHI::RhiVertexBufferView>       CreateVertexBufferView() override;
        virtual Ref<RHI::RhiVertexBufferView>       GetFullScreenQuadVertexBufferView() const override;

        // Cache
        virtual void                                SetCacheDirectory(const String& dir) override;
        virtual String                              GetCacheDir() const override;

        // Extension
        virtual Owner<RHI::FSR2::RhiFsr2Processor>  CreateFsr2Processor() override;

        // Raytracing
        virtual Owner<RHI::RhiRTInstance>           CreateTLAS() { return nullptr; }
        virtual Owner<RHI::RhiRTScene>              CreateBLAS() { return nullptr; }
        virtual Owner<RHI::RhiRTShaderBindingTable> CreateShaderBindingTable() { return nullptr; }

        virtual Owner<RHI::RhiRTPass>               CreateRaytracingPass() { return nullptr; }
    };

    class IFRIT_APIDECL RhiVulkanBackendBuilder : public RHI::RhiBackendFactory, public NonCopyable
    {
    public:
        Owner<RHI::RhiBackend> CreateBackend(const RHI::RhiInitializeArguments& args) override;
    };
} // namespace Ifrit::RHI::VulkanAdapter