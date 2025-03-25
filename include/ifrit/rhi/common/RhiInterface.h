
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

namespace Ifrit::GraphicsBackend::Rhi {

// Structs
struct RhiInitializeArguments {
  Fn<const char **(u32 *)> m_extensionGetter;
  bool m_enableValidationLayer = true;
  bool m_enableHardwareRayTracing = false;
  u32 m_surfaceWidth = -1;
  u32 m_surfaceHeight = -1;
  u32 m_expectedSwapchainImageCount = 3;
  u32 m_expectedGraphicsQueueCount = 1;
  u32 m_expectedComputeQueueCount = 1;
  u32 m_expectedTransferQueueCount = 1;
#ifdef _WIN32
  struct {
    HINSTANCE m_hInstance;
    HWND m_hWnd;
  } m_win32;
#else
  struct {
    void *m_hInstance;
    void *m_hWnd;
  } m_win32;

#endif
};

// classes
class IFRIT_APIDECL RhiBackendFactory {
public:
  virtual ~RhiBackendFactory() = default;
  virtual std::unique_ptr<RhiBackend> createBackend(const RhiInitializeArguments &args) = 0;
};

class IFRIT_APIDECL RhiBackend {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiBackend() = default;
  // Timer

  virtual std::shared_ptr<RhiDeviceTimer> createDeviceTimer() = 0;
  // Memory resource
  virtual void waitDeviceIdle() = 0;

  // Create a general buffer
  virtual RhiBufferRef createBuffer(const String &name, u32 size, u32 usage, bool hostVisible, bool addUAV) const = 0;
  virtual RhiBufferRef createBufferDevice(const String &name, u32 size, u32 usage, bool addUAV) const = 0;
  virtual std::shared_ptr<RhiMultiBuffer> createBufferCoherent(u32 size, u32 usage, u32 numCopies = ~0u) const = 0;
  virtual RhiBufferRef getFullScreenQuadVertexBuffer() const = 0;

  // Note that the texture created can only be accessed by the GPU
  virtual RhiTextureRef createDepthTexture(const String &name, u32 width, u32 height, bool addUAV) = 0;
  virtual RhiTextureRef createTexture2D(const String &name, u32 width, u32 height, RhiImageFormat format,
                                        u32 extraFlags, bool addUAV) = 0;
  virtual RhiTextureRef createTexture3D(const String &name, u32 width, u32 height, u32 depth, RhiImageFormat format,
                                        u32 extraFlags, bool addUAV) = 0;
  virtual RhiTextureRef createMipMapTexture(const String &name, u32 width, u32 height, u32 mips, RhiImageFormat format,
                                            u32 extraFlags, bool addUAV) = 0;
  virtual RhiSamplerRef createTrivialSampler() = 0;
  virtual RhiSamplerRef createTrivialBilinearSampler(bool repeat) = 0;
  virtual RhiSamplerRef createTrivialNearestSampler(bool repeat) = 0;

  virtual std::shared_ptr<RhiStagedSingleBuffer> createStagedSingleBuffer(RhiBuffer *target) = 0;

  // Command execution
  virtual RhiQueue *getQueue(RhiQueueCapability req) = 0;
  virtual RhiShader *createShader(const std::string &name, const std::vector<char> &code, const std::string &entry,
                                  RhiShaderStage stage, RhiShaderSourceType sourceType) = 0;

  // Pass execution
  virtual RhiComputePass *createComputePass() = 0;
  virtual RhiGraphicsPass *createGraphicsPass() = 0;

  // Swapchain
  virtual RhiTexture *getSwapchainImage() = 0;
  virtual void beginFrame() = 0;
  virtual void endFrame() = 0;
  virtual std::unique_ptr<RhiTaskSubmission> getSwapchainFrameReadyEventHandler() = 0;
  virtual std::unique_ptr<RhiTaskSubmission> getSwapchainRenderDoneEventHandler() = 0;

  // Descriptor, these are deprecated.
  virtual RhiBindlessDescriptorRef *createBindlessDescriptorRef() = 0;
  virtual std::shared_ptr<Rhi::RhiDescHandleLegacy> registerUAVImage2(Rhi::RhiTexture *texture,
                                                                      Rhi::RhiImageSubResource subResource) = 0;
  virtual std::shared_ptr<RhiDescHandleLegacy> registerUniformBuffer(RhiMultiBuffer *buffer) = 0;
  virtual std::shared_ptr<RhiDescHandleLegacy> registerStorageBufferShared(RhiMultiBuffer *buffer) = 0;
  virtual std::shared_ptr<RhiDescHandleLegacy> registerCombinedImageSampler(RhiTexture *texture,
                                                                            RhiSampler *sampler) = 0;

  // Render target
  virtual std::shared_ptr<RhiColorAttachment> createRenderTarget(RhiTexture *renderTarget, RhiClearValue clearValue,
                                                                 RhiRenderTargetLoadOp loadOp, u32 mip,
                                                                 u32 arrLayer) = 0;

  virtual std::shared_ptr<RhiDepthStencilAttachment>
  createRenderTargetDepthStencil(RhiTexture *renderTarget, RhiClearValue clearValue, RhiRenderTargetLoadOp loadOp) = 0;

  virtual std::shared_ptr<RhiRenderTargets> createRenderTargets() = 0;

  // Vertex buffer
  virtual std::shared_ptr<RhiVertexBufferView> createVertexBufferView() = 0;
  virtual std::shared_ptr<RhiVertexBufferView> getFullScreenQuadVertexBufferView() const = 0;

  virtual void setCacheDirectory(const std::string &dir) = 0;
  virtual std::string getCacheDirectory() const = 0;

  // Extensions
  virtual std::unique_ptr<FSR2::RhiFsr2Processor> createFsr2Processor() = 0;

  // Raytracing
  virtual std::unique_ptr<RhiRTInstance> createTLAS() = 0;
  virtual std::unique_ptr<RhiRTScene> createBLAS() = 0;
  virtual std::unique_ptr<RhiRTShaderBindingTable> createShaderBindingTable() = 0;
  virtual std::unique_ptr<RhiRTPass> createRaytracingPass() = 0;
};

} // namespace Ifrit::GraphicsBackend::Rhi