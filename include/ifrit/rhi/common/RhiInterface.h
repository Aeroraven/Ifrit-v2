
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
#include "RhiDevice.h"
#include "RhiCommandList.h"
#include "RhiCommandListContext.h"

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <Windows.h>
#endif

namespace Ifrit::RHI
{

    // Structs
    struct RhiInitializeArguments
    {
        Fn<const char**(u32*)> mExtensionGetter;
        u32                    mSurfaceWidth                = -1;
        u32                    mSurfaceHeight               = -1;
        u32                    mExpectedSwapchainImageCount = 3;
        RhiCapabilityList      mDesiredCapabilities;

        ERhiVendor             mPreferredVendor            = ERhiVendor::Any;
        u32                    mPreferredGraphcisAdapterId = ~0u;

#ifdef _WIN32
        struct
        {
            HINSTANCE m_hInstance;
            HWND      m_hWnd;
        } mWin32;
#else
        struct
        {
            void* m_hInstance;
            void* m_hWnd;
        } mWin32;

#endif
    };

    // classes
    class IFRIT_APIDECL RhiBackendFactory
    {
    public:
        virtual ~RhiBackendFactory()                                                = default;
        virtual Owner<RhiBackend> CreateBackend(const RhiInitializeArguments& args) = 0;
    };

    class IFRIT_APIDECL RhiBackend
    {
    protected:
        RhiDevice* mContext;

    protected:
        virtual RhiCommandListAllocator* GetCommandListAllocator() = 0;

    public:
        virtual ~RhiBackend() = default;

        // ===== Core =====
        virtual void                     BeginFrame()                         = 0;
        virtual void                     EndFrame()                           = 0;
        virtual Owner<RhiTaskSubmission> GetSwapchainFrameReadyEventHandler() = 0;
        virtual Owner<RhiTaskSubmission> GetSwapchainRenderDoneEventHandler() = 0;
        virtual void                     WaitDeviceIdle()                     = 0;
        virtual RhiCapabilityList        GetCapabilities() const              = 0;

        virtual void                     SetCacheDirectory(const String& dir) = 0;
        virtual String                   GetCacheDir() const                  = 0;

        virtual RhiTexture*              GetSwapchainImage() = 0;

        // ===== Resource Creation =====
        virtual RhiBufferRef        CreateBuffer(const String& name, u32 size, u32 usage, bool hostVisible) const  = 0;
        virtual RhiTextureRef       CreateTexture(const String& name, u32 width, u32 height, u32 depth, u32 mipLevels,
                  ERhiImageFormat format, u32 accessFlags) const                                                   = 0;
        virtual RhiSamplerRef       CreateSampler(ERhiSamplerFilter filter, ERhiSamplerWrapMode addressMode) const = 0;
        virtual Ref<RhiMultiBuffer> CreateBufferCoherent(
            const String& name, u32 size, u32 usage, u32 numCopies = ~0u) const = 0;

        // ===== Resource Views =====
        virtual RhiUAVRef                  CreateUAV(RhiTexture* texture, RhiImageSubResource subResource) const = 0;
        virtual RhiUAVRef                  CreateUAV(RhiBuffer* buffer) const                                    = 0;
        virtual RhiSRVRef                  CreateSRV(RhiTexture* texture, RhiImageSubResource subResource) const = 0;
        virtual RhiSRVRef                  CreateSRV(RhiTexture* texture) const                                  = 0;
        virtual RhiSRVRef                  CreateSRV(RhiBuffer* buffer) const                                    = 0;
        virtual RhiCBVRef                  CreateCBV(RhiBuffer* buffer) const                                    = 0;

        // ===== Shader =====
        virtual Ref<RhiShaderCollection>   CreateShader(const String& name, const Vec<char>& code, const String& entry,
              ERhiShaderStage stage, ERhiShaderSourceType sourceType) = 0;

        // ===== Staged Buffer Creation =====
        virtual Ref<RhiStagedSingleBuffer> CreateStagedBuffer(RhiBuffer* target) = 0;

        // ===== Pipeline Creation =====
        virtual Owner<RhiComputePass>      CreateComputePass()  = 0;
        virtual Owner<RhiGraphicsPass>     CreateGraphicsPass() = 0;

        // ===== Render Targets =====
        virtual Ref<RhiColorAttachment>    CreateRenderTarget(RhiTexture* renderTarget, RhiClearValue2 clearValue,
               ERhiRenderTargetLoadOp loadOp, u32 mip, u32 arrLayer) = 0;
        virtual Ref<RhiDepthStencilAttachment> CreateRenderTargetDepthStencil(
            RhiTexture* renderTarget, RhiClearValue2 clearValue, ERhiRenderTargetLoadOp loadOp) = 0;
        virtual Ref<RhiRenderTargets>          CreateRenderTargets()                            = 0;

        // ===== Raytracing =====
        virtual Owner<RhiRTInstance>           CreateTLAS()               = 0;
        virtual Owner<RhiRTScene>              CreateBLAS()               = 0;
        virtual Owner<RhiRTShaderBindingTable> CreateShaderBindingTable() = 0;
        virtual Owner<RhiRTPass>               CreateRaytracingPass()     = 0;

        // ===== Utility =====
        virtual Ref<RhiDeviceTimer>            CreateDeviceTimer() = 0;

        // ===== Raw Handles =====
        virtual RhiRawHandle                   GetRawHandle_Instance() const      = 0;
        virtual RhiRawHandle                   GetRawHandle_ActiveAdapter() const = 0;
        virtual RhiRawHandle                   GetRawHandle_Device() const        = 0;

        // ===== Extension =====
        virtual Owner<FSR2::RhiFsr2Processor>  CreateFsr2Processor() = 0;

        // ===== Commands (Queues are planned to be removed) =====
        virtual RhiQueue*                      GetQueue(ERhiQueueCapability req)            = 0;
        virtual RhiCommandListBase*            AllocateCommandList(ERhiQueueCapability req) = 0;
    };

} // namespace Ifrit::RHI