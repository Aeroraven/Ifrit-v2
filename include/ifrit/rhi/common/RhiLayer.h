
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
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
#include "RhiForwardingTypes.h"
#include "RhiFsr2Processor.h"
#include "ifrit/common/util/ApiConv.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// TODO: Raw pointers are used in the interface, this is not good?
// New interfaces added here will use smart pointers, however, the old
// interfaces are still remained unchanged. This should be fixed in the future.

namespace Ifrit::GraphicsBackend::Rhi {

// Structs
struct RhiInitializeArguments {
  std::function<const char **(uint32_t *)> m_extensionGetter;
  bool m_enableValidationLayer = true;
  uint32_t m_surfaceWidth = -1;
  uint32_t m_surfaceHeight = -1;
  uint32_t m_expectedSwapchainImageCount = 3;
  uint32_t m_expectedGraphicsQueueCount = 1;
  uint32_t m_expectedComputeQueueCount = 1;
  uint32_t m_expectedTransferQueueCount = 1;
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

struct RhiAttachmentBlendInfo {
  bool m_blendEnable = false;
  RhiBlendFactor m_srcColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE;
  RhiBlendFactor m_dstColorBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ZERO;
  RhiBlendOp m_colorBlendOp = RhiBlendOp::RHI_BLEND_OP_ADD;
  RhiBlendFactor m_srcAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ONE;
  RhiBlendFactor m_dstAlphaBlendFactor = RhiBlendFactor::RHI_BLEND_FACTOR_ZERO;
  RhiBlendOp m_alphaBlendOp = RhiBlendOp::RHI_BLEND_OP_ADD;
};

struct RhiClearValue {
  float m_color[4];
  float m_depth;
  uint32_t m_stencil;
};

struct RhiViewport {
  float x;
  float y;
  float width;
  float height;
  float minDepth;
  float maxDepth;
};

struct RhiScissor {
  int32_t x;
  int32_t y;
  uint32_t width;
  uint32_t height;
};

struct RhiImageSubResource {
  uint32_t mipLevel;
  uint32_t arrayLayer;
  uint32_t mipCount = 1;
  uint32_t layerCount = 1;
};

struct RhiBindlessIdRef {
  uint32_t activeFrame;
  std::vector<uint32_t> ids;

  inline uint32_t getActiveId() const { return ids[activeFrame]; }

  inline void setFromId(uint32_t frame) { activeFrame = frame; }
};

enum class RhiBarrierType { UAVAccess, Transition };

struct RhiUAVBarrier {
  RhiResourceType m_type;
  union {
    RhiBuffer *m_buffer;
    RhiTexture *m_texture;
  };
};

struct RhiTransitionBarrier {
  RhiResourceType m_type;
  union {
    RhiBuffer *m_buffer = nullptr;
    RhiTexture *m_texture;
  };
  RhiImageSubResource m_subResource = {0, 0, 1, 1};
  RhiResourceState m_srcState = RhiResourceState::Undefined;
  RhiResourceState m_dstState = RhiResourceState::Undefined;

  RhiTransitionBarrier() { m_texture = nullptr; }
};

struct RhiResourceBarrier {
  RhiBarrierType m_type = RhiBarrierType::UAVAccess;
  union {
    RhiUAVBarrier m_uav;
    RhiTransitionBarrier m_transition;
  };

  RhiResourceBarrier() { m_uav = {}; }
};

// classes
class IFRIT_APIDECL RhiBackendFactory {
public:
  virtual ~RhiBackendFactory() = default;
  virtual std::unique_ptr<Rhi::RhiBackend>
  createBackend(const RhiInitializeArguments &args) = 0;
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
  virtual std::shared_ptr<RhiBuffer> createBuffer(uint32_t size, uint32_t usage,
                                                  bool hostVisible) const = 0;
  virtual RhiBuffer *createIndirectMeshDrawBufferDevice(uint32_t drawCalls,
                                                        uint32_t usage) = 0;
  virtual RhiBuffer *createStorageBufferDevice(uint32_t size,
                                               uint32_t usage) = 0;
  virtual RhiMultiBuffer *createMultiBuffer(uint32_t size, uint32_t usage,
                                            uint32_t numCopies) = 0;
  virtual RhiMultiBuffer *createUniformBufferShared(uint32_t size,
                                                    bool hostVisible,
                                                    uint32_t extraFlags) = 0;
  virtual RhiMultiBuffer *createStorageBufferShared(uint32_t size,
                                                    bool hostVisible,
                                                    uint32_t extraFlags) = 0;
  virtual std::shared_ptr<Rhi::RhiTexture>
  createDepthRenderTexture(uint32_t width, uint32_t height) = 0;

  virtual std::shared_ptr<RhiBuffer> getFullScreenQuadVertexBuffer() const = 0;

  // Note that the texture created can only be accessed by the GPU
  virtual std::shared_ptr<RhiTexture> createTexture2D(uint32_t width,
                                                      uint32_t height,
                                                      RhiImageFormat format,
                                                      uint32_t extraFlags) = 0;
  virtual std::shared_ptr<RhiTexture>
  createRenderTargetTexture(uint32_t width, uint32_t height,
                            RhiImageFormat format, uint32_t extraFlags) = 0;

  virtual std::shared_ptr<RhiTexture>
  createRenderTargetTexture3D(uint32_t width, uint32_t height, uint32_t depth,
                              RhiImageFormat format, uint32_t extraFlags) = 0;

  virtual std::shared_ptr<RhiTexture>
  createRenderTargetMipTexture(uint32_t width, uint32_t height, uint32_t mips,
                               RhiImageFormat format, uint32_t extraFlags) = 0;

  virtual std::shared_ptr<RhiSampler> createTrivialSampler() = 0;
  virtual std::shared_ptr<RhiSampler>
  createTrivialBilinearSampler(bool repeat) = 0;

  virtual std::shared_ptr<RhiSampler>
  createTrivialNearestSampler(bool repeat) = 0;

  virtual std::shared_ptr<Rhi::RhiStagedSingleBuffer>
  createStagedSingleBuffer(RhiBuffer *target) = 0;

  // Command execution
  virtual RhiQueue *getQueue(RhiQueueCapability req) = 0;
  virtual RhiShader *createShader(const std::string &name,
                                  const std::vector<char> &code,
                                  const std::string &entry,
                                  Rhi::RhiShaderStage stage,
                                  Rhi::RhiShaderSourceType sourceType) = 0;

  // Pass execution
  virtual RhiComputePass *createComputePass() = 0;
  virtual RhiGraphicsPass *createGraphicsPass() = 0;

  // Swapchain
  virtual RhiTexture *getSwapchainImage() = 0;
  virtual void beginFrame() = 0;
  virtual void endFrame() = 0;
  virtual std::unique_ptr<RhiTaskSubmission>
  getSwapchainFrameReadyEventHandler() = 0;
  virtual std::unique_ptr<RhiTaskSubmission>
  getSwapchainRenderDoneEventHandler() = 0;

  // Descriptor
  virtual RhiBindlessDescriptorRef *createBindlessDescriptorRef() = 0;
  virtual std::shared_ptr<RhiBindlessIdRef>
  registerUniformBuffer(RhiMultiBuffer *buffer) = 0;

  virtual std::shared_ptr<RhiBindlessIdRef>
  registerStorageBuffer(RhiBuffer *buffer) = 0;

  virtual std::shared_ptr<RhiBindlessIdRef>
  registerStorageBufferShared(RhiMultiBuffer *buffer) = 0;

  virtual std::shared_ptr<Rhi::RhiBindlessIdRef>
  registerUAVImage(Rhi::RhiTexture *texture,
                   Rhi::RhiImageSubResource subResource) = 0;

  virtual std::shared_ptr<Rhi::RhiBindlessIdRef>
  registerCombinedImageSampler(Rhi::RhiTexture *texture,
                               Rhi::RhiSampler *sampler) = 0;

  // Render target
  virtual std::shared_ptr<RhiColorAttachment>
  createRenderTarget(RhiTexture *renderTarget, RhiClearValue clearValue,
                     RhiRenderTargetLoadOp loadOp, uint32_t mip,
                     uint32_t arrLayer) = 0;

  virtual std::shared_ptr<RhiDepthStencilAttachment>
  createRenderTargetDepthStencil(RhiTexture *renderTarget,
                                 RhiClearValue clearValue,
                                 RhiRenderTargetLoadOp loadOp) = 0;

  virtual std::shared_ptr<RhiRenderTargets> createRenderTargets() = 0;

  // Vertex buffer
  virtual std::shared_ptr<RhiVertexBufferView> createVertexBufferView() = 0;
  virtual std::shared_ptr<RhiVertexBufferView>
  getFullScreenQuadVertexBufferView() const = 0;

  virtual void setCacheDirectory(const std::string &dir) = 0;
  virtual std::string getCacheDirectory() const = 0;

  // Extensions
  virtual std::unique_ptr<FSR2::RhiFsr2Processor> createFsr2Processor() = 0;
};

// RHI device

class IFRIT_APIDECL RhiDevice {
protected:
  virtual int _polymorphismPlaceHolder() { return 0; }
};

class IFRIT_APIDECL RhiSwapchain {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiSwapchain() = default;
  virtual void present() = 0;
  virtual uint32_t acquireNextImage() = 0;
  virtual uint32_t getNumBackbuffers() const = 0;
  virtual uint32_t getCurrentFrameIndex() const = 0;
  virtual uint32_t getCurrentImageIndex() const = 0;
};

// RHI memory resource

class IFRIT_APIDECL RhiBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiBuffer() = default;
  virtual void map() = 0;
  virtual void unmap() = 0;
  virtual void flush() = 0;
  virtual void readBuffer(void *data, uint32_t size, uint32_t offset) = 0;
  virtual void writeBuffer(const void *data, uint32_t size,
                           uint32_t offset) = 0;
};

class IFRIT_APIDECL RhiMultiBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual Rhi::RhiBuffer *getActiveBuffer() = 0;
  virtual Rhi::RhiBuffer *getActiveBufferRelative(uint32_t deltaFrame) = 0;
  virtual ~RhiMultiBuffer() = default;
};

class IFRIT_APIDECL RhiStagedSingleBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiStagedSingleBuffer() = default;
  virtual void cmdCopyToDevice(const RhiCommandBuffer *cmd, const void *data,
                               uint32_t size, uint32_t localOffset) = 0;
};

class RhiStagedMultiBuffer {};

// RHI command

class RhiTaskSubmission {
protected:
  virtual int _polymorphismPlaceHolder() { return 0; }
};

class RhiHostBarrier {};

class IFRIT_APIDECL RhiCommandBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual void copyBuffer(const RhiBuffer *srcBuffer,
                          const RhiBuffer *dstBuffer, uint32_t size,
                          uint32_t srcOffset, uint32_t dstOffset) const = 0;
  virtual void dispatch(uint32_t groupCountX, uint32_t groupCountY,
                        uint32_t groupCountZ) const = 0;
  virtual void
  setViewports(const std::vector<Rhi::RhiViewport> &viewport) const = 0;
  virtual void
  setScissors(const std::vector<Rhi::RhiScissor> &scissor) const = 0;
  virtual void drawMeshTasksIndirect(const Rhi::RhiBuffer *buffer,
                                     uint32_t offset, uint32_t drawCount,
                                     uint32_t stride) const = 0;

  virtual void imageBarrier(const RhiTexture *texture, RhiResourceState src,
                            RhiResourceState dst,
                            RhiImageSubResource subResource) const = 0;

  virtual void uavBufferBarrier(const RhiBuffer *buffer) const = 0;

  // Clear UAV storage buffer, considered as a transfer operation, typically
  // need a barrier for sync.
  virtual void uavBufferClear(const RhiBuffer *buffer, uint32_t val) const = 0;

  virtual void
  attachBindlessReferenceGraphics(Rhi::RhiGraphicsPass *pass, uint32_t setId,
                                  RhiBindlessDescriptorRef *ref) const = 0;

  virtual void
  attachBindlessReferenceCompute(Rhi::RhiComputePass *pass, uint32_t setId,
                                 RhiBindlessDescriptorRef *ref) const = 0;

  virtual void
  attachVertexBufferView(const RhiVertexBufferView &view) const = 0;

  virtual void
  attachVertexBuffers(uint32_t firstSlot,
                      const std::vector<RhiBuffer *> &buffers) const = 0;

  virtual void drawInstanced(uint32_t vertexCount, uint32_t instanceCount,
                             uint32_t firstVertex,
                             uint32_t firstInstance) const = 0;

  virtual void dispatchIndirect(const RhiBuffer *buffer,
                                uint32_t offset) const = 0;

  virtual void setPushConst(Rhi::RhiComputePass *pass, uint32_t offset,
                            uint32_t size, const void *data) const = 0;
  virtual void setPushConst(Rhi::RhiGraphicsPass *pass, uint32_t offset,
                            uint32_t size, const void *data) const = 0;

  virtual void clearUAVImageFloat(const RhiTexture *texture,
                                  RhiImageSubResource subResource,
                                  const std::array<float, 4> &val) const = 0;
  virtual void
  resourceBarrier(const std::vector<RhiResourceBarrier> &barriers) const = 0;

  virtual void globalMemoryBarrier() const = 0;

  virtual void beginScope(const std::string &name) const = 0;
  virtual void endScope() const = 0;

  virtual void copyImage(const RhiTexture *src, RhiImageSubResource srcSub,
                         const RhiTexture *dst,
                         RhiImageSubResource dstSub) const = 0;

  virtual void copyBufferToImage(const RhiBuffer *src, const RhiTexture *dst,
                                 RhiImageSubResource dstSub) const = 0;

  virtual void setCullMode(RhiCullMode mode) const = 0;
};

class IFRIT_APIDECL RhiQueue {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiQueue() = default;

  // Runs a command buffer, with CPU waiting
  // the GPU to finish
  virtual void
  runSyncCommand(std::function<void(const RhiCommandBuffer *)> func) = 0;

  // Runs a command buffer, with CPU not
  // waiting the GPU to finish
  virtual std::unique_ptr<RhiTaskSubmission>
  runAsyncCommand(std::function<void(const RhiCommandBuffer *)> func,
                  const std::vector<RhiTaskSubmission *> &waitOn,
                  const std::vector<RhiTaskSubmission *> &toIssue) = 0;

  // Host sync
  virtual void hostWaitEvent(RhiTaskSubmission *event) = 0;
};

// RHI shader

class IFRIT_APIDECL RhiShader {
public:
  virtual RhiShaderStage getStage() const = 0;
  virtual uint32_t getNumDescriptorSets() const = 0;
};

// RHI imaging

class IFRIT_APIDECL RhiTexture {
protected:
  RhiDevice *m_context;
  bool m_rhiSwapchainImage = false;

public:
  virtual ~RhiTexture() = default;
  virtual uint32_t getHeight() const = 0;
  virtual uint32_t getWidth() const = 0;
  virtual bool isDepthTexture() const = 0;
};

// RHI pipeline

struct IFRIT_APIDECL RhiRenderPassContext {
  const RhiCommandBuffer *m_cmd;
  uint32_t m_frame;
};

class IFRIT_APIDECL RhiGeneralPassBase {};

class IFRIT_APIDECL RhiComputePass : public RhiGeneralPassBase {

public:
  virtual ~RhiComputePass() = default;
  virtual void setComputeShader(RhiShader *shader) = 0;
  virtual void
  setShaderBindingLayout(const std::vector<RhiDescriptorType> &layout) = 0;
  virtual void addShaderStorageBuffer(RhiBuffer *buffer, uint32_t position,
                                      RhiResourceAccessType access) = 0;
  virtual void addUniformBuffer(RhiMultiBuffer *buffer, uint32_t position) = 0;
  virtual void setExecutionFunction(
      std::function<void(Rhi::RhiRenderPassContext *)> func) = 0;
  virtual void
  setRecordFunction(std::function<void(Rhi::RhiRenderPassContext *)> func) = 0;

  virtual void run(const RhiCommandBuffer *cmd, uint32_t frameId) = 0;
  virtual void setNumBindlessDescriptorSets(uint32_t num) = 0;
  virtual void setPushConstSize(uint32_t size) = 0;
};

class IFRIT_APIDECL RhiGraphicsPass : public RhiGeneralPassBase {

public:
  virtual ~RhiGraphicsPass() = default;
  virtual void setTaskShader(RhiShader *shader) = 0;
  virtual void setMeshShader(RhiShader *shader) = 0;
  virtual void setVertexShader(RhiShader *shader) = 0;
  virtual void setPixelShader(RhiShader *shader) = 0;
  virtual void setRasterizerTopology(RhiRasterizerTopology topology) = 0;
  virtual void setRenderArea(uint32_t x, uint32_t y, uint32_t width,
                             uint32_t height) = 0;
  virtual void setDepthWrite(bool write) = 0;
  virtual void setDepthTestEnable(bool enable) = 0;
  virtual void setDepthCompareOp(Rhi::RhiCompareOp compareOp) = 0;

  virtual void
  setRenderTargetFormat(const Rhi::RhiRenderTargetsFormat &format) = 0;
  virtual void
  setShaderBindingLayout(const std::vector<RhiDescriptorType> &layout) = 0;
  virtual void addShaderStorageBuffer(RhiBuffer *buffer, uint32_t position,
                                      RhiResourceAccessType access) = 0;
  virtual void addUniformBuffer(RhiMultiBuffer *buffer, uint32_t position) = 0;
  virtual void setExecutionFunction(
      std::function<void(Rhi::RhiRenderPassContext *)> func) = 0;
  virtual void
  setRecordFunction(std::function<void(Rhi::RhiRenderPassContext *)> func) = 0;
  virtual void setRecordFunctionPostRenderPass(
      std::function<void(Rhi::RhiRenderPassContext *)> func) = 0;

  virtual void run(const RhiCommandBuffer *cmd, RhiRenderTargets *renderTargets,
                   uint32_t frameId) = 0;
  virtual void setNumBindlessDescriptorSets(uint32_t num) = 0;
  virtual void setPushConstSize(uint32_t size) = 0;
};

class IFRIT_APIDECL RhiPassGraph {};

// Rhi Descriptors

class IFRIT_APIDECL RhiBindlessDescriptorRef {
public:
  virtual void addUniformBuffer(Rhi::RhiMultiBuffer *buffer, uint32_t loc) = 0;
  virtual void addStorageBuffer(Rhi::RhiMultiBuffer *buffer, uint32_t loc) = 0;
  virtual void addStorageBuffer(Rhi::RhiBuffer *buffer, uint32_t loc) = 0;
  virtual void addCombinedImageSampler(Rhi::RhiTexture *texture,
                                       Rhi::RhiSampler *sampler,
                                       uint32_t loc) = 0;
  virtual void addUAVImage(Rhi::RhiTexture *texture,
                           RhiImageSubResource subResource, uint32_t loc) = 0;
};

// Rhi RenderTargets
struct IFRIT_APIDECL RhiRenderTargetsFormat {
  RhiImageFormat m_depthFormat;
  std::vector<RhiImageFormat> m_colorFormats;
};

class IFRIT_APIDECL RhiRenderTargets {
public:
  virtual void setColorAttachments(
      const std::vector<Rhi::RhiColorAttachment *> &attachments) = 0;
  virtual void
  setDepthStencilAttachment(Rhi::RhiDepthStencilAttachment *attachment) = 0;
  virtual void
  beginRendering(const Rhi::RhiCommandBuffer *commandBuffer) const = 0;
  virtual void
  endRendering(const Rhi::RhiCommandBuffer *commandBuffer) const = 0;
  virtual void setRenderArea(Rhi::RhiScissor area) = 0;
  virtual RhiRenderTargetsFormat getFormat() const = 0;
  virtual Rhi::RhiScissor getRenderArea() const = 0;
  virtual RhiDepthStencilAttachment *getDepthStencilAttachment() const = 0;
  virtual RhiColorAttachment *getColorAttachment(uint32_t index) const = 0;
};

class IFRIT_APIDECL RhiColorAttachment {
public:
  virtual RhiTexture *getRenderTarget() const = 0;
  virtual void setBlendInfo(const RhiAttachmentBlendInfo &info) = 0;
};

class IFRIT_APIDECL RhiDepthStencilAttachment {
public:
  virtual RhiTexture *getTexture() const = 0;
};

class IFRIT_APIDECL RhiSampler {
protected:
  virtual int _polymorphismPlaceHolder() { return 0; }
};

class IFRIT_APIDECL RhiVertexBufferView {
protected:
  virtual void addBinding(
      std::vector<uint32_t> location, std::vector<Rhi::RhiImageFormat> format,
      std::vector<uint32_t> offset, uint32_t stride,
      Rhi::RhiVertexInputRate inputRate = Rhi::RhiVertexInputRate::Vertex) = 0;
};

class IFRIT_APIDECL RhiDeviceTimer {
public:
  virtual void start(const RhiCommandBuffer *cmd) = 0;
  virtual void stop(const RhiCommandBuffer *cmd) = 0;
  virtual float getElapsedMs() = 0;
};

} // namespace Ifrit::GraphicsBackend::Rhi