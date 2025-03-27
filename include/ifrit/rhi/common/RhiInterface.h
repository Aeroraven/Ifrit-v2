
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

#include "RhiBaseTypes.h"
#include "RhiFsr2Processor.h"

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <Windows.h>
#endif

namespace Ifrit::Graphics::Rhi
{

    // Structs
    struct RhiInitializeArguments
    {
        Fn<const char**(u32*)> m_extensionGetter;
        bool                   m_enableValidationLayer       = true;
        bool                   m_enableHardwareRayTracing    = false;
        u32                    m_surfaceWidth                = -1;
        u32                    m_surfaceHeight               = -1;
        u32                    m_expectedSwapchainImageCount = 3;
        u32                    m_expectedGraphicsQueueCount  = 1;
        u32                    m_expectedComputeQueueCount   = 1;
        u32                    m_expectedTransferQueueCount  = 1;
#ifdef _WIN32
        struct
        {
            HINSTANCE m_hInstance;
            HWND      m_hWnd;
        } m_win32;
#else
        struct
        {
            void* m_hInstance;
            void* m_hWnd;
        } m_win32;

#endif
    };

    // classes
    class IFRIT_APIDECL RhiBackendFactory
    {
    public:
        virtual ~RhiBackendFactory()                                               = default;
        virtual Uref<RhiBackend> CreateBackend(const RhiInitializeArguments& args) = 0;
    };

    class IFRIT_APIDECL RhiBackend
    {
    protected:
        RhiDevice* m_context;

    public:
        virtual ~RhiBackend() = default;
        // Timer

        virtual Ref<RhiDeviceTimer>           CreateDeviceTimer() = 0;
        // Memory resource
        virtual void                          WaitDeviceIdle() = 0;

        // Create a general buffer
        virtual RhiBufferRef                  CreateBuffer(const String& name, u32 size, u32 usage, bool hostVisible, bool addUAV) const = 0;
        virtual RhiBufferRef                  CreateBufferDevice(const String& name, u32 size, u32 usage, bool addUAV) const             = 0;
        virtual Ref<RhiMultiBuffer>           CreateBufferCoherent(u32 size, u32 usage, u32 numCopies = ~0u) const                       = 0;
        virtual RhiBufferRef                  GetFullScreenQuadVertexBuffer() const                                                      = 0;

        // Note that the texture created can only be accessed by the GPU
        virtual RhiTextureRef                 CreateDepthTexture(const String& name, u32 width, u32 height, bool addUAV)                                                   = 0;
        virtual RhiTextureRef                 CreateTexture2D(const String& name, u32 width, u32 height, RhiImageFormat format, u32 extraFlags, bool addUAV)               = 0;
        virtual RhiTextureRef                 CreateTexture3D(const String& name, u32 width, u32 height, u32 depth, RhiImageFormat format, u32 extraFlags, bool addUAV)    = 0;
        virtual RhiTextureRef                 CreateMipMapTexture(const String& name, u32 width, u32 height, u32 mips, RhiImageFormat format, u32 extraFlags, bool addUAV) = 0;
        virtual RhiSamplerRef                 CreateTrivialSampler()                                                                                                       = 0;
        virtual RhiSamplerRef                 CreateTrivialBilinearSampler(bool repeat)                                                                                    = 0;
        virtual RhiSamplerRef                 CreateTrivialNearestSampler(bool repeat)                                                                                     = 0;

        virtual Ref<RhiStagedSingleBuffer>    CreateStagedSingleBuffer(RhiBuffer* target) = 0;

        // Command execution
        virtual RhiQueue*                     GetQueue(RhiQueueCapability req)       = 0;
        virtual RhiShader*                    CreateShader(const String& name, const Vec<char>& code, const String& entry,
                               RhiShaderStage stage, RhiShaderSourceType sourceType) = 0;

        // Pass execution
        virtual RhiComputePass*               CreateComputePass()  = 0;
        virtual RhiGraphicsPass*              CreateGraphicsPass() = 0;

        // Swapchain
        virtual RhiTexture*                   GetSwapchainImage()                  = 0;
        virtual void                          BeginFrame()                         = 0;
        virtual void                          EndFrame()                           = 0;
        virtual Uref<RhiTaskSubmission>       GetSwapchainFrameReadyEventHandler() = 0;
        virtual Uref<RhiTaskSubmission>       GetSwapchainRenderDoneEventHandler() = 0;

        // Descriptor, these are deprecated.
        virtual RhiBindlessDescriptorRef*     createBindlessDescriptorRef()                                                     = 0;
        virtual Ref<Rhi::RhiDescHandleLegacy> RegisterUAVImage2(Rhi::RhiTexture* texture, Rhi::RhiImageSubResource subResource) = 0;
        virtual Ref<RhiDescHandleLegacy>      RegisterUniformBuffer(RhiMultiBuffer* buffer)                                     = 0;
        virtual Ref<RhiDescHandleLegacy>      RegisterStorageBufferShared(RhiMultiBuffer* buffer)                               = 0;
        virtual Ref<RhiDescHandleLegacy>      RegisterCombinedImageSampler(RhiTexture* texture, RhiSampler* sampler)            = 0;

        // Render target
        virtual Ref<RhiColorAttachment>       CreateRenderTarget(
                  RhiTexture*           renderTarget,
                  RhiClearValue         clearValue,
                  RhiRenderTargetLoadOp loadOp,
                  u32                   mip,
                  u32                   arrLayer) = 0;

        virtual Ref<RhiDepthStencilAttachment> CreateRenderTargetDepthStencil(
            RhiTexture*           renderTarget,
            RhiClearValue         clearValue,
            RhiRenderTargetLoadOp loadOp) = 0;

        virtual Ref<RhiRenderTargets>         CreateRenderTargets() = 0;

        // Vertex buffer
        virtual Ref<RhiVertexBufferView>      CreateVertexBufferView()                  = 0;
        virtual Ref<RhiVertexBufferView>      GetFullScreenQuadVertexBufferView() const = 0;

        virtual void                          SetCacheDirectory(const String& dir) = 0;
        virtual String                        GetCacheDir() const                  = 0;

        // Extensions
        virtual Uref<FSR2::RhiFsr2Processor>  CreateFsr2Processor() = 0;

        // Raytracing
        virtual Uref<RhiRTInstance>           CreateTLAS()               = 0;
        virtual Uref<RhiRTScene>              CreateBLAS()               = 0;
        virtual Uref<RhiRTShaderBindingTable> CreateShaderBindingTable() = 0;
        virtual Uref<RhiRTPass>               CreateRaytracingPass()     = 0;
    };

} // namespace Ifrit::Graphics::Rhi