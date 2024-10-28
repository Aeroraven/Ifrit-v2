#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace Ifrit::Engine::GraphicsBackend::Rhi {
class RhiBackend;
class RhiContext;
class RhiBuffer;
class RhiTexture;
class RhiSampler;
class RhiShader;
class RhiPipeline;
class RhiCommandBuffer;
class RhiSwapchain;
class RhiDevice;
class RhiMultiBuffer;
class RhiStagedSingleBuffer;

// I don't think the abstraction should be on RHI level
// however, I am too lazy to refactor the whole thing
class RhiComputePass;
class RhiGraphicsPass;
class RhiPassGraph;

class RhiQueue;
class RhiHostBarrier;
class RhiTaskSubmission;
class RhiGraphicsQueue;
class RhiComputeQueue;
class RhiTransferQueue;

// Enums
enum RhiBufferUsage {
  RHI_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001,
  RHI_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002,
  RHI_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT = 0x00000004,
  RHI_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT = 0x00000008,
  RHI_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010,
  RHI_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020,
  RHI_BUFFER_USAGE_INDEX_BUFFER_BIT = 0x00000040,
  RHI_BUFFER_USAGE_VERTEX_BUFFER_BIT = 0x00000080,
  RHI_BUFFER_USAGE_INDIRECT_BUFFER_BIT = 0x00000100,
  RHI_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x00020000,
};

enum RhiQueueCapability {
  RHI_QUEUE_GRAPHICS_BIT = 0x00000001,
  RHI_QUEUE_COMPUTE_BIT = 0x00000002,
  RHI_QUEUE_TRANSFER_BIT = 0x00000004,
};

enum class RhiShaderStage { Vertex, Fragment, Compute, Mesh, Task };

enum class RhiDescriptorType {
  UniformBuffer,
  StorageBuffer,
  CombinedImageSampler,
  StorageImage,
  MaxEnum
};

enum class RhiResourceAccessType {
  Read,
  Write,
  ReadOrWrite,
  ReadAndWrite,
};

enum class RhiRenderTargetLoadOp { Load, Clear, DontCare };
enum class RhiCullMode { None, Front, Back };
enum class RhiRasterizerTopology { TriangleList, Line, Point };
enum class RhiGeometryGenerationType { Conventional, Mesh };

enum class RhiCompareOp {
  Never,
  Less,
  Equal,
  LessOrEqual,
  Greater,
  NotEqual,
  GreaterOrEqual,
  Always
};

enum class RhiResourceState { Undefined, RenderTarget, Present };

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

// classes
class RhiBackendFactory {
public:
  virtual ~RhiBackendFactory() = default;
  virtual std::unique_ptr<Rhi::RhiBackend>
  createBackend(const RhiInitializeArguments &args) = 0;
};

class RhiBackend {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiBackend() = default;

  // Memory resource

  virtual void waitDeviceIdle() = 0;

  // Create a general buffer
  virtual RhiBuffer *createBuffer(uint32_t size, uint32_t usage,
                                  bool hostVisible) const = 0;
  virtual RhiBuffer *createIndirectMeshDrawBufferDevice(uint32_t drawCalls) = 0;
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

  virtual RhiTexture *createDepthRenderTexture(uint32_t width,
                                               uint32_t height) = 0;

  virtual RhiStagedSingleBuffer *
  createStagedSingleBuffer(RhiBuffer *target) = 0;

  // Command execution
  virtual RhiQueue *getQueue(RhiQueueCapability req) = 0;

  virtual RhiShader *createShader(const std::vector<char> &code,
                                  std::string entry,
                                  Rhi::RhiShaderStage stage) = 0;

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
};

// RHI device

class RhiDevice {
protected:
  virtual int _polymorphismPlaceHolder() { return 0; }
};

class RhiSwapchain {
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

class RhiBuffer {
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

class RhiMultiBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual Rhi::RhiBuffer *getActiveBuffer() = 0;
  virtual ~RhiMultiBuffer() = default;
};

class RhiStagedSingleBuffer {
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

class RhiCommandBuffer {
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
                            RhiResourceState dst) const = 0;
};

class RhiQueue {
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

class RhiShader {
protected:
  virtual int _polymorphismPlaceHolder() { return 0; }
};

// RHI imaging

class RhiTexture {
protected:
  RhiDevice *m_context;
  bool m_rhiSwapchainImage = false;

public:
  virtual ~RhiTexture() = default;
};

// RHI pipeline

struct RhiRenderPassContext {
  const RhiCommandBuffer *m_cmd;
  uint32_t m_frame;
};

class RhiGeneralPassBase {};

class RhiComputePass {

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
};

class RhiGraphicsPass {

public:
  virtual ~RhiGraphicsPass() = default;
  virtual void setMeshShader(RhiShader *shader) = 0;
  virtual void setPixelShader(RhiShader *shader) = 0;
  virtual void setRasterizerTopology(RhiRasterizerTopology topology) = 0;
  virtual void setRenderArea(uint32_t x, uint32_t y, uint32_t width,
                             uint32_t height) = 0;
  virtual void setDepthWrite(bool write) = 0;
  virtual void setDepthTestEnable(bool enable) = 0;
  virtual void setDepthCompareOp(Rhi::RhiCompareOp compareOp) = 0;
  virtual void addColorAttachment(RhiTexture *texture, RhiRenderTargetLoadOp op,
                                  RhiClearValue clearValue) = 0;
  virtual void setDepthAttachment(RhiTexture *texture, RhiRenderTargetLoadOp op,
                                  RhiClearValue clearValue) = 0;
  virtual void
  setShaderBindingLayout(const std::vector<RhiDescriptorType> &layout) = 0;
  virtual void addShaderStorageBuffer(RhiBuffer *buffer, uint32_t position,
                                      RhiResourceAccessType access) = 0;
  virtual void addUniformBuffer(RhiMultiBuffer *buffer, uint32_t position) = 0;
  virtual void setExecutionFunction(
      std::function<void(Rhi::RhiRenderPassContext *)> func) = 0;
  virtual void
  setRecordFunction(std::function<void(Rhi::RhiRenderPassContext *)> func) = 0;
  virtual void
  setRecordFunctionPostRenderPass(std::function<void(Rhi::RhiRenderPassContext *)> func) = 0;

  virtual void run(const RhiCommandBuffer *cmd, uint32_t frameId) = 0;
};

class RhiPassGraph {};

} // namespace Ifrit::Engine::GraphicsBackend::Rhi