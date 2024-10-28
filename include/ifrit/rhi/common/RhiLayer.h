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

// Enums
enum RhiBufferUsage {
  RHI_BUFFER_USAGE_VERTEX_BUFFER = 0x00000001,
  RHI_BUFFER_USAGE_INDEX_BUFFER = 0x00000002,
  RHI_BUFFER_USAGE_UNIFORM_BUFFER = 0x00000004,
  RHI_BUFFER_USAGE_STORAGE_BUFFER = 0x00000008,
  RHI_BUFFER_USAGE_TRANSFER_SRC = 0x00000010,
  RHI_BUFFER_USAGE_TRANSFER_DST = 0x00000020,
  RHI_BUFFER_USAGE_INDIRECT_BUFFER = 0x00000040,
  RHI_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER = 0x00000080,
  RHI_BUFFER_USAGE_STORAGE_TEXEL_BUFFER = 0x00000100,
  RHI_BUFFER_USAGE_VERTEX_BUFFER_DYNAMIC = 0x00000200,
  RHI_BUFFER_USAGE_INDEX_BUFFER_DYNAMIC = 0x00000400,
  RHI_BUFFER_USAGE_UNIFORM_BUFFER_DYNAMIC = 0x00000800,
  RHI_BUFFER_USAGE_STORAGE_BUFFER_DYNAMIC = 0x00001000,
  RHI_BUFFER_USAGE_TRANSFER_SRC_DYNAMIC = 0x00002000,
  RHI_BUFFER_USAGE_TRANSFER_DST_DYNAMIC = 0x00004000,
  RHI_BUFFER_USAGE_INDIRECT_BUFFER_DYNAMIC = 0x00008000,
  RHI_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_DYNAMIC = 0x00010000,
  RHI_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_DYNAMIC = 0x00020000,
};

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

  virtual RhiStagedSingleBuffer *
  createStagedSingleBuffer(RhiBuffer *target) = 0;
};

class RhiDevice {};

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

class RhiBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiBuffer() = default;
};

class RhiMultiBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiMultiBuffer() = default;
};

class RhiStagedSingleBuffer {
protected:
  RhiDevice *m_context;

public:
  virtual ~RhiStagedSingleBuffer() = default;
};

class RhiStagedMultiBuffer {};

} // namespace Ifrit::Engine::GraphicsBackend::Rhi