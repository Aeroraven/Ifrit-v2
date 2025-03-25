
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <memory>

namespace Ifrit::GraphicsBackend::VulkanGraphics {

struct RhiVulkanBackendImplDetails;
class IFRIT_APIDECL RhiVulkanBackend : public Rhi::RhiBackend {
protected:
  // Note that destructor order matters here
  // https://isocpp.org/wiki/faq/dtors#order-dtors-for-members
  Uref<Rhi::RhiDevice> m_device;
  Uref<Rhi::RhiSwapchain> m_swapChain;
  RhiVulkanBackendImplDetails *m_implDetails;

public:
  RhiVulkanBackend(const Rhi::RhiInitializeArguments &args);
  ~RhiVulkanBackend();

  void waitDeviceIdle() override;
  Ref<Rhi::RhiDeviceTimer> createDeviceTimer() override;
  Rhi::RhiBufferRef createBuffer(const String &name, u32 size, u32 usage, bool hostVisible, bool addUAV) const override;
  Rhi::RhiBufferRef createBufferDevice(const String &name, u32 size, u32 usage, bool addUAV) const override;
  Ref<Rhi::RhiMultiBuffer> createBufferCoherent(u32 size, u32 usage, u32 numCopies = ~0u) const override;
  Ref<Rhi::RhiStagedSingleBuffer> createStagedSingleBuffer(Rhi::RhiBuffer *target) override;
  Rhi::RhiBufferRef getFullScreenQuadVertexBuffer() const override;

  // Command execution
  Rhi::RhiQueue *getQueue(Rhi::RhiQueueCapability req) override;

  // Shader
  Rhi::RhiShader *createShader(const std::string &name, const std::vector<char> &code, const std::string &entry,
                               Rhi::RhiShaderStage stage, Rhi::RhiShaderSourceType sourceType) override;

  // Texture
  Rhi::RhiTextureRef createTexture2D(const String &name, u32 width, u32 height, Rhi::RhiImageFormat format,
                                     u32 extraFlags, bool addUAV) override;
  Rhi::RhiTextureRef createDepthTexture(const String &name, u32 width, u32 height, bool addUAV) override;
  Rhi::RhiTextureRef createTexture3D(const String &name, u32 width, u32 height, u32 depth, Rhi::RhiImageFormat format,
                                     u32 extraFlags, bool addUAV) override;
  Rhi::RhiSamplerRef createTrivialSampler() override;
  Rhi::RhiSamplerRef createTrivialBilinearSampler(bool repeat) override;
  Rhi::RhiSamplerRef createTrivialNearestSampler(bool repeat) override;

  Rhi::RhiTextureRef createMipMapTexture(const String &name, u32 width, u32 height, u32 mips,
                                         Rhi::RhiImageFormat format, u32 extraFlags, bool addUAV) override;

  // Pass
  Rhi::RhiComputePass *createComputePass() override;
  Rhi::RhiGraphicsPass *createGraphicsPass() override;

  // Swapchain
  Rhi::RhiTexture *getSwapchainImage() override;
  void beginFrame() override;
  void endFrame() override;
  Uref<Rhi::RhiTaskSubmission> getSwapchainFrameReadyEventHandler() override;
  Uref<Rhi::RhiTaskSubmission> getSwapchainRenderDoneEventHandler() override;

  // Descriptor
  virtual Rhi::RhiBindlessDescriptorRef *createBindlessDescriptorRef() override;
  virtual Ref<Rhi::RhiDescHandleLegacy> registerUAVImage2(Rhi::RhiTexture *texture,
                                                          Rhi::RhiImageSubResource subResource) override;
  virtual Ref<Rhi::RhiDescHandleLegacy> registerUniformBuffer(Rhi::RhiMultiBuffer *buffer) override;
  virtual Ref<Rhi::RhiDescHandleLegacy> registerStorageBufferShared(Rhi::RhiMultiBuffer *buffer) override;
  virtual Ref<Rhi::RhiDescHandleLegacy> registerCombinedImageSampler(Rhi::RhiTexture *texture,
                                                                     Rhi::RhiSampler *sampler) override;

  // Render targets
  virtual Ref<Rhi::RhiColorAttachment> createRenderTarget(Rhi::RhiTexture *renderTarget, Rhi::RhiClearValue clearValue,
                                                          Rhi::RhiRenderTargetLoadOp loadOp, u32 mips,
                                                          u32 layers) override;

  virtual Ref<Rhi::RhiDepthStencilAttachment>
  createRenderTargetDepthStencil(Rhi::RhiTexture *renderTarget, Rhi::RhiClearValue clearValue,
                                 Rhi::RhiRenderTargetLoadOp loadOp) override;

  virtual Ref<Rhi::RhiRenderTargets> createRenderTargets() override;

  // Vertex buffer
  virtual Ref<Rhi::RhiVertexBufferView> createVertexBufferView() override;
  virtual Ref<Rhi::RhiVertexBufferView> getFullScreenQuadVertexBufferView() const override;

  // Cache
  virtual void setCacheDirectory(const std::string &dir) override;
  virtual std::string getCacheDirectory() const override;

  // Extension
  virtual Uref<Rhi::FSR2::RhiFsr2Processor> createFsr2Processor() override;

  // Raytracing
  virtual Uref<Rhi::RhiRTInstance> createTLAS() { return nullptr; }
  virtual Uref<Rhi::RhiRTScene> createBLAS() { return nullptr; }
  virtual Uref<Rhi::RhiRTShaderBindingTable> createShaderBindingTable() { return nullptr; }

  virtual Uref<Rhi::RhiRTPass> createRaytracingPass() { return nullptr; }
};

class IFRIT_APIDECL RhiVulkanBackendBuilder : public Rhi::RhiBackendFactory, public Common::Utility::NonCopyable {
public:
  Uref<Rhi::RhiBackend> createBackend(const Rhi::RhiInitializeArguments &args) override;
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics