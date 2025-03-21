
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

using RhiDeviceAddr = u64;

// Structs
struct RhiInitializeArguments {
  std::function<const char **(u32 *)> m_extensionGetter;
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
  f32 m_color[4];
  f32 m_depth;
  u32 m_stencil;
};

struct RhiViewport {
  f32 x;
  f32 y;
  f32 width;
  f32 height;
  f32 minDepth;
  f32 maxDepth;
};

struct RhiScissor {
  int32_t x;
  int32_t y;
  u32 width;
  u32 height;
};

struct RhiImageSubResource {
  u32 mipLevel;
  u32 arrayLayer;
  u32 mipCount = 1;
  u32 layerCount = 1;
};

struct RhiBindlessIdRef {
  u32 activeFrame;
  std::vector<u32> ids;

  inline u32 getActiveId() const { return ids[activeFrame]; }

  inline void setFromId(u32 frame) { activeFrame = frame; }
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
  RhiResourceState2 m_srcState = RhiResourceState2::AutoTraced;
  RhiResourceState2 m_dstState = RhiResourceState2::AutoTraced;

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
  virtual std::shared_ptr<RhiBuffer> createBuffer(u32 size, u32 usage, bool hostVisible) const = 0;
  virtual std::shared_ptr<RhiBuffer> createBufferDevice(u32 size, u32 usage) const = 0;

  virtual std::shared_ptr<RhiMultiBuffer> createBufferCoherent(u32 size, u32 usage, u32 numCopies = ~0u) const = 0;

  virtual std::shared_ptr<RhiTexture> createDepthTexture(u32 width, u32 height) = 0;

  virtual std::shared_ptr<RhiBuffer> getFullScreenQuadVertexBuffer() const = 0;

  // Note that the texture created can only be accessed by the GPU
  virtual std::shared_ptr<RhiTexture> createTexture2D(u32 width, u32 height, RhiImageFormat format, u32 extraFlags) = 0;

  virtual std::shared_ptr<RhiTexture> createTexture3D(u32 width, u32 height, u32 depth, RhiImageFormat format,
                                                      u32 extraFlags) = 0;

  virtual std::shared_ptr<RhiTexture> createMipMapTexture(u32 width, u32 height, u32 mips, RhiImageFormat format,
                                                          u32 extraFlags) = 0;

  virtual std::shared_ptr<RhiSampler> createTrivialSampler() = 0;
  virtual std::shared_ptr<RhiSampler> createTrivialBilinearSampler(bool repeat) = 0;
  virtual std::shared_ptr<RhiSampler> createTrivialNearestSampler(bool repeat) = 0;

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

  // Descriptor
  virtual RhiBindlessDescriptorRef *createBindlessDescriptorRef() = 0;
  virtual std::shared_ptr<RhiBindlessIdRef> registerUniformBuffer(RhiMultiBuffer *buffer) = 0;

  virtual std::shared_ptr<RhiBindlessIdRef> registerStorageBuffer(RhiBuffer *buffer) = 0;

  virtual std::shared_ptr<RhiBindlessIdRef> registerStorageBufferShared(RhiMultiBuffer *buffer) = 0;

  virtual std::shared_ptr<RhiBindlessIdRef> registerUAVImage(RhiTexture *texture, RhiImageSubResource subResource) = 0;

  virtual std::shared_ptr<RhiBindlessIdRef> registerCombinedImageSampler(RhiTexture *texture, RhiSampler *sampler) = 0;

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
  virtual u32 acquireNextImage() = 0;
  virtual u32 getNumBackbuffers() const = 0;
  virtual u32 getCurrentFrameIndex() const = 0;
  virtual u32 getCurrentImageIndex() const = 0;
};

// RHI memory resource

class IFRIT_APIDECL RhiBuffer {
protected:
  RhiDevice *m_context;
  RhiResourceState2 m_state = RhiResourceState2::Undefined;

private:
  inline void setState(RhiResourceState2 state) { m_state = state; }

public:
  virtual ~RhiBuffer() = default;
  virtual void map() = 0;
  virtual void unmap() = 0;
  virtual void flush() = 0;
  virtual void readBuffer(void *data, u32 size, u32 offset) = 0;
  virtual void writeBuffer(const void *data, u32 size, u32 offset) = 0;
  virtual inline RhiResourceState2 getState() const { return m_state; }

  virtual RhiDeviceAddr getDeviceAddress() const = 0;

  friend class RhiCommandBuffer;
};

class IFRIT_APIDECL RhiMultiBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual RhiBuffer *getActiveBuffer() = 0;
  virtual RhiBuffer *getActiveBufferRelative(u32 deltaFrame) = 0;
  virtual ~RhiMultiBuffer() = default;
};

class IFRIT_APIDECL RhiStagedSingleBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiStagedSingleBuffer() = default;
  virtual void cmdCopyToDevice(const RhiCommandBuffer *cmd, const void *data, u32 size, u32 localOffset) = 0;
};

class RhiStagedMultiBuffer {};

// RHI imaging

class IFRIT_APIDECL RhiTexture {
protected:
  RhiDevice *m_context;
  RhiResourceState2 m_state = RhiResourceState2::Undefined;
  bool m_rhiSwapchainImage = false;

private:
  inline void setState(RhiResourceState2 state) { m_state = state; }

public:
  virtual ~RhiTexture() = default;
  virtual u32 getHeight() const = 0;
  virtual u32 getWidth() const = 0;
  virtual bool isDepthTexture() const = 0;
  virtual inline RhiResourceState2 getState() const { return m_state; }
  virtual void *getNativeHandle() const = 0;

  friend class RhiCommandBuffer;
};

// RHI command

class RhiTaskSubmission {
protected:
  virtual int _polymorphismPlaceHolder() { return 0; }
};

class RhiHostBarrier {};

class IFRIT_APIDECL RhiCommandBuffer {
protected:
  RhiDevice *m_context;

protected:
  inline void _setTextureState(RhiTexture *texture, RhiResourceState2 state) const { texture->setState(state); }
  inline void _setBufferState(RhiBuffer *buffer, RhiResourceState2 state) const { buffer->setState(state); }

public:
  virtual void copyBuffer(const RhiBuffer *srcBuffer, const RhiBuffer *dstBuffer, u32 size, u32 srcOffset,
                          u32 dstOffset) const = 0;
  virtual void dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const = 0;
  virtual void setViewports(const std::vector<RhiViewport> &viewport) const = 0;
  virtual void setScissors(const std::vector<RhiScissor> &scissor) const = 0;
  virtual void drawMeshTasksIndirect(const RhiBuffer *buffer, u32 offset, u32 drawCount, u32 stride) const = 0;

  // Clear UAV storage buffer, considered as a transfer operation, typically
  // need a barrier for sync.
  virtual void bufferClear(const RhiBuffer *buffer, u32 val) const = 0;

  virtual void attachBindlessReferenceGraphics(RhiGraphicsPass *pass, u32 setId,
                                               RhiBindlessDescriptorRef *ref) const = 0;

  virtual void attachBindlessReferenceCompute(RhiComputePass *pass, u32 setId, RhiBindlessDescriptorRef *ref) const = 0;

  virtual void attachVertexBufferView(const RhiVertexBufferView &view) const = 0;

  virtual void attachVertexBuffers(u32 firstSlot, const std::vector<RhiBuffer *> &buffers) const = 0;

  virtual void drawInstanced(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const = 0;

  virtual void dispatchIndirect(const RhiBuffer *buffer, u32 offset) const = 0;

  virtual void setPushConst(RhiComputePass *pass, u32 offset, u32 size, const void *data) const = 0;
  virtual void setPushConst(RhiGraphicsPass *pass, u32 offset, u32 size, const void *data) const = 0;

  virtual void clearUAVImageFloat(const RhiTexture *texture, RhiImageSubResource subResource,
                                  const std::array<f32, 4> &val) const = 0;
  virtual void resourceBarrier(const std::vector<RhiResourceBarrier> &barriers) const = 0;

  virtual void globalMemoryBarrier() const = 0;

  virtual void beginScope(const std::string &name) const = 0;
  virtual void endScope() const = 0;

  virtual void copyImage(const RhiTexture *src, RhiImageSubResource srcSub, const RhiTexture *dst,
                         RhiImageSubResource dstSub) const = 0;

  virtual void copyBufferToImage(const RhiBuffer *src, const RhiTexture *dst, RhiImageSubResource dstSub) const = 0;

  virtual void setCullMode(RhiCullMode mode) const = 0;
};

class IFRIT_APIDECL RhiQueue {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiQueue() = default;

  // Runs a command buffer, with CPU waiting
  // the GPU to finish
  virtual void runSyncCommand(std::function<void(const RhiCommandBuffer *)> func) = 0;

  // Runs a command buffer, with CPU not
  // waiting the GPU to finish
  virtual std::unique_ptr<RhiTaskSubmission> runAsyncCommand(std::function<void(const RhiCommandBuffer *)> func,
                                                             const std::vector<RhiTaskSubmission *> &waitOn,
                                                             const std::vector<RhiTaskSubmission *> &toIssue) = 0;

  // Host sync
  virtual void hostWaitEvent(RhiTaskSubmission *event) = 0;
};

// RHI shader

class IFRIT_APIDECL RhiShader {
public:
  virtual RhiShaderStage getStage() const = 0;
  virtual u32 getNumDescriptorSets() const = 0;
};

// RHI pipeline

struct IFRIT_APIDECL RhiRenderPassContext {
  const RhiCommandBuffer *m_cmd;
  u32 m_frame;
};

class IFRIT_APIDECL RhiGeneralPassBase {};

class IFRIT_APIDECL RhiComputePass : public RhiGeneralPassBase {

public:
  virtual ~RhiComputePass() = default;
  virtual void setComputeShader(RhiShader *shader) = 0;
  virtual void setShaderBindingLayout(const std::vector<RhiDescriptorType> &layout) = 0;
  virtual void addShaderStorageBuffer(RhiBuffer *buffer, u32 position, RhiResourceAccessType access) = 0;
  virtual void addUniformBuffer(RhiMultiBuffer *buffer, u32 position) = 0;
  virtual void setExecutionFunction(std::function<void(RhiRenderPassContext *)> func) = 0;
  virtual void setRecordFunction(std::function<void(RhiRenderPassContext *)> func) = 0;

  virtual void run(const RhiCommandBuffer *cmd, u32 frameId) = 0;
  virtual void setNumBindlessDescriptorSets(u32 num) = 0;
  virtual void setPushConstSize(u32 size) = 0;
};

class IFRIT_APIDECL RhiGraphicsPass : public RhiGeneralPassBase {

public:
  virtual ~RhiGraphicsPass() = default;
  virtual void setTaskShader(RhiShader *shader) = 0;
  virtual void setMeshShader(RhiShader *shader) = 0;
  virtual void setVertexShader(RhiShader *shader) = 0;
  virtual void setPixelShader(RhiShader *shader) = 0;
  virtual void setRasterizerTopology(RhiRasterizerTopology topology) = 0;
  virtual void setRenderArea(u32 x, u32 y, u32 width, u32 height) = 0;
  virtual void setDepthWrite(bool write) = 0;
  virtual void setDepthTestEnable(bool enable) = 0;
  virtual void setDepthCompareOp(RhiCompareOp compareOp) = 0;

  virtual void setRenderTargetFormat(const RhiRenderTargetsFormat &format) = 0;
  virtual void setShaderBindingLayout(const std::vector<RhiDescriptorType> &layout) = 0;
  virtual void addShaderStorageBuffer(RhiBuffer *buffer, u32 position, RhiResourceAccessType access) = 0;
  virtual void addUniformBuffer(RhiMultiBuffer *buffer, u32 position) = 0;
  virtual void setExecutionFunction(std::function<void(RhiRenderPassContext *)> func) = 0;
  virtual void setRecordFunction(std::function<void(RhiRenderPassContext *)> func) = 0;
  virtual void setRecordFunctionPostRenderPass(std::function<void(RhiRenderPassContext *)> func) = 0;

  virtual void run(const RhiCommandBuffer *cmd, RhiRenderTargets *renderTargets, u32 frameId) = 0;
  virtual void setNumBindlessDescriptorSets(u32 num) = 0;
  virtual void setPushConstSize(u32 size) = 0;
};

class IFRIT_APIDECL RhiPassGraph {};

// Rhi Descriptors

class IFRIT_APIDECL RhiBindlessDescriptorRef {
public:
  virtual void addUniformBuffer(RhiMultiBuffer *buffer, u32 loc) = 0;
  virtual void addStorageBuffer(RhiMultiBuffer *buffer, u32 loc) = 0;
  virtual void addStorageBuffer(RhiBuffer *buffer, u32 loc) = 0;
  virtual void addCombinedImageSampler(RhiTexture *texture, RhiSampler *sampler, u32 loc) = 0;
  virtual void addUAVImage(RhiTexture *texture, RhiImageSubResource subResource, u32 loc) = 0;
};

// Rhi RenderTargets
struct IFRIT_APIDECL RhiRenderTargetsFormat {
  RhiImageFormat m_depthFormat;
  std::vector<RhiImageFormat> m_colorFormats;
};

class IFRIT_APIDECL RhiRenderTargets {
public:
  virtual void setColorAttachments(const std::vector<RhiColorAttachment *> &attachments) = 0;
  virtual void setDepthStencilAttachment(RhiDepthStencilAttachment *attachment) = 0;
  virtual void beginRendering(const RhiCommandBuffer *commandBuffer) const = 0;
  virtual void endRendering(const RhiCommandBuffer *commandBuffer) const = 0;
  virtual void setRenderArea(RhiScissor area) = 0;
  virtual RhiRenderTargetsFormat getFormat() const = 0;
  virtual RhiScissor getRenderArea() const = 0;
  virtual RhiDepthStencilAttachment *getDepthStencilAttachment() const = 0;
  virtual RhiColorAttachment *getColorAttachment(u32 index) const = 0;
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
  virtual void addBinding(std::vector<u32> location, std::vector<RhiImageFormat> format, std::vector<u32> offset,
                          u32 stride, RhiVertexInputRate inputRate = RhiVertexInputRate::Vertex) = 0;
};

class IFRIT_APIDECL RhiDeviceTimer {
public:
  virtual void start(const RhiCommandBuffer *cmd) = 0;
  virtual void stop(const RhiCommandBuffer *cmd) = 0;
  virtual f32 getElapsedMs() = 0;
};

// Raytracing types
struct IFRIT_APIDECL RhiRTGeometryReference {
  RhiDeviceAddr m_vertex;
  RhiDeviceAddr m_index;
  RhiDeviceAddr m_transform;
  u32 m_numVertices;
  u32 m_numIndices;
  u32 m_vertexComponents = 3;
  u32 m_vertexStride = 12;
};

struct IFRIT_APIDECL RhiRTShaderGroup {
  RhiShader *m_generalShader = nullptr;
  RhiShader *m_closestHitShader = nullptr;
  RhiShader *m_anyHitShader = nullptr;
  RhiShader *m_intersectionShader = nullptr;
};

class IFRIT_APIDECL RhiRTInstance {
public:
  virtual RhiDeviceAddr getDeviceAddress() const = 0;
};

class IFRIT_APIDECL RhiRTScene {
public:
  virtual RhiDeviceAddr getDeviceAddress() const = 0;
};

class IFRIT_APIDECL RhiRTShaderBindingTable {
public:
  virtual void _polymorphismPlaceHolder() {}
};

class IFRIT_APIDECL RhiRTPipeline {
public:
  virtual void _polymorphismPlaceHolder() {}
};

class IFRIT_APIDECL RhiRTPass : public RhiGeneralPassBase {
public:
  virtual void _polymorphismPlaceHolder() {}
};

} // namespace Ifrit::GraphicsBackend::Rhi