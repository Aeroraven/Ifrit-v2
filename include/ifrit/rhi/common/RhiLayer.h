#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
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

// Note here 'passes' are in fact 'pipeline references'
// If two pass hold similar pipeline CI, they are the same
class RhiComputePass;
class RhiGraphicsPass;
class RhiPassGraph;

class RhiQueue;
class RhiHostBarrier;
class RhiTaskSubmission;
class RhiGraphicsQueue;
class RhiComputeQueue;
class RhiTransferQueue;

class RhiBindlessDescriptorRef;

class RhiRenderTargets;
struct RhiRenderTargetsFormat;
class RhiColorAttachment;
class RhiDepthStencilAttachment;

class RhiVertexBufferView;

struct RhiImageSubResource;

class RhiDeviceTimer;

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

enum RhiImageUsage {
  RHI_IMAGE_USAGE_TRANSFER_SRC_BIT = 1,
  RHI_IMAGE_USAGE_TRANSFER_DST_BIT = 2,
  RHI_IMAGE_USAGE_SAMPLED_BIT = 4,
  RHI_IMAGE_USAGE_STORAGE_BIT = 8,
  RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT = 16,
  RHI_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 32,
  RHI_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT = 64,
  RHI_IMAGE_USAGE_INPUT_ATTACHMENT_BIT = 128,
};

enum RhiQueueCapability {
  RHI_QUEUE_GRAPHICS_BIT = 0x00000001,
  RHI_QUEUE_COMPUTE_BIT = 0x00000002,
  RHI_QUEUE_TRANSFER_BIT = 0x00000004,
};

// These are just mapped from vulkan spec
enum RhiImageFormat {
  RHI_FORMAT_UNDEFINED = 0,
  RHI_FORMAT_R4G4_UNORM_PACK8 = 1,
  RHI_FORMAT_R4G4B4A4_UNORM_PACK16 = 2,
  RHI_FORMAT_B4G4R4A4_UNORM_PACK16 = 3,
  RHI_FORMAT_R5G6B5_UNORM_PACK16 = 4,
  RHI_FORMAT_B5G6R5_UNORM_PACK16 = 5,
  RHI_FORMAT_R5G5B5A1_UNORM_PACK16 = 6,
  RHI_FORMAT_B5G5R5A1_UNORM_PACK16 = 7,
  RHI_FORMAT_A1R5G5B5_UNORM_PACK16 = 8,
  RHI_FORMAT_R8_UNORM = 9,
  RHI_FORMAT_R8_SNORM = 10,
  RHI_FORMAT_R8_USCALED = 11,
  RHI_FORMAT_R8_SSCALED = 12,
  RHI_FORMAT_R8_UINT = 13,
  RHI_FORMAT_R8_SINT = 14,
  RHI_FORMAT_R8_SRGB = 15,
  RHI_FORMAT_R8G8_UNORM = 16,
  RHI_FORMAT_R8G8_SNORM = 17,
  RHI_FORMAT_R8G8_USCALED = 18,
  RHI_FORMAT_R8G8_SSCALED = 19,
  RHI_FORMAT_R8G8_UINT = 20,
  RHI_FORMAT_R8G8_SINT = 21,
  RHI_FORMAT_R8G8_SRGB = 22,
  RHI_FORMAT_R8G8B8_UNORM = 23,
  RHI_FORMAT_R8G8B8_SNORM = 24,
  RHI_FORMAT_R8G8B8_USCALED = 25,
  RHI_FORMAT_R8G8B8_SSCALED = 26,
  RHI_FORMAT_R8G8B8_UINT = 27,
  RHI_FORMAT_R8G8B8_SINT = 28,
  RHI_FORMAT_R8G8B8_SRGB = 29,
  RHI_FORMAT_B8G8R8_UNORM = 30,
  RHI_FORMAT_B8G8R8_SNORM = 31,
  RHI_FORMAT_B8G8R8_USCALED = 32,
  RHI_FORMAT_B8G8R8_SSCALED = 33,
  RHI_FORMAT_B8G8R8_UINT = 34,
  RHI_FORMAT_B8G8R8_SINT = 35,
  RHI_FORMAT_B8G8R8_SRGB = 36,
  RHI_FORMAT_R8G8B8A8_UNORM = 37,
  RHI_FORMAT_R8G8B8A8_SNORM = 38,
  RHI_FORMAT_R8G8B8A8_USCALED = 39,
  RHI_FORMAT_R8G8B8A8_SSCALED = 40,
  RHI_FORMAT_R8G8B8A8_UINT = 41,
  RHI_FORMAT_R8G8B8A8_SINT = 42,
  RHI_FORMAT_R8G8B8A8_SRGB = 43,
  RHI_FORMAT_B8G8R8A8_UNORM = 44,
  RHI_FORMAT_B8G8R8A8_SNORM = 45,
  RHI_FORMAT_B8G8R8A8_USCALED = 46,
  RHI_FORMAT_B8G8R8A8_SSCALED = 47,
  RHI_FORMAT_B8G8R8A8_UINT = 48,
  RHI_FORMAT_B8G8R8A8_SINT = 49,
  RHI_FORMAT_B8G8R8A8_SRGB = 50,
  RHI_FORMAT_A8B8G8R8_UNORM_PACK32 = 51,
  RHI_FORMAT_A8B8G8R8_SNORM_PACK32 = 52,
  RHI_FORMAT_A8B8G8R8_USCALED_PACK32 = 53,
  RHI_FORMAT_A8B8G8R8_SSCALED_PACK32 = 54,
  RHI_FORMAT_A8B8G8R8_UINT_PACK32 = 55,
  RHI_FORMAT_A8B8G8R8_SINT_PACK32 = 56,
  RHI_FORMAT_A8B8G8R8_SRGB_PACK32 = 57,
  RHI_FORMAT_A2R10G10B10_UNORM_PACK32 = 58,
  RHI_FORMAT_A2R10G10B10_SNORM_PACK32 = 59,
  RHI_FORMAT_A2R10G10B10_USCALED_PACK32 = 60,
  RHI_FORMAT_A2R10G10B10_SSCALED_PACK32 = 61,
  RHI_FORMAT_A2R10G10B10_UINT_PACK32 = 62,
  RHI_FORMAT_A2R10G10B10_SINT_PACK32 = 63,
  RHI_FORMAT_A2B10G10R10_UNORM_PACK32 = 64,
  RHI_FORMAT_A2B10G10R10_SNORM_PACK32 = 65,
  RHI_FORMAT_A2B10G10R10_USCALED_PACK32 = 66,
  RHI_FORMAT_A2B10G10R10_SSCALED_PACK32 = 67,
  RHI_FORMAT_A2B10G10R10_UINT_PACK32 = 68,
  RHI_FORMAT_A2B10G10R10_SINT_PACK32 = 69,
  RHI_FORMAT_R16_UNORM = 70,
  RHI_FORMAT_R16_SNORM = 71,
  RHI_FORMAT_R16_USCALED = 72,
  RHI_FORMAT_R16_SSCALED = 73,
  RHI_FORMAT_R16_UINT = 74,
  RHI_FORMAT_R16_SINT = 75,
  RHI_FORMAT_R16_SFLOAT = 76,
  RHI_FORMAT_R16G16_UNORM = 77,
  RHI_FORMAT_R16G16_SNORM = 78,
  RHI_FORMAT_R16G16_USCALED = 79,
  RHI_FORMAT_R16G16_SSCALED = 80,
  RHI_FORMAT_R16G16_UINT = 81,
  RHI_FORMAT_R16G16_SINT = 82,
  RHI_FORMAT_R16G16_SFLOAT = 83,
  RHI_FORMAT_R16G16B16_UNORM = 84,
  RHI_FORMAT_R16G16B16_SNORM = 85,
  RHI_FORMAT_R16G16B16_USCALED = 86,
  RHI_FORMAT_R16G16B16_SSCALED = 87,
  RHI_FORMAT_R16G16B16_UINT = 88,
  RHI_FORMAT_R16G16B16_SINT = 89,
  RHI_FORMAT_R16G16B16_SFLOAT = 90,
  RHI_FORMAT_R16G16B16A16_UNORM = 91,
  RHI_FORMAT_R16G16B16A16_SNORM = 92,
  RHI_FORMAT_R16G16B16A16_USCALED = 93,
  RHI_FORMAT_R16G16B16A16_SSCALED = 94,
  RHI_FORMAT_R16G16B16A16_UINT = 95,
  RHI_FORMAT_R16G16B16A16_SINT = 96,
  RHI_FORMAT_R16G16B16A16_SFLOAT = 97,
  RHI_FORMAT_R32_UINT = 98,
  RHI_FORMAT_R32_SINT = 99,
  RHI_FORMAT_R32_SFLOAT = 100,
  RHI_FORMAT_R32G32_UINT = 101,
  RHI_FORMAT_R32G32_SINT = 102,
  RHI_FORMAT_R32G32_SFLOAT = 103,
  RHI_FORMAT_R32G32B32_UINT = 104,
  RHI_FORMAT_R32G32B32_SINT = 105,
  RHI_FORMAT_R32G32B32_SFLOAT = 106,
  RHI_FORMAT_R32G32B32A32_UINT = 107,
  RHI_FORMAT_R32G32B32A32_SINT = 108,
  RHI_FORMAT_R32G32B32A32_SFLOAT = 109,
  RHI_FORMAT_R64_UINT = 110,
  RHI_FORMAT_R64_SINT = 111,
  RHI_FORMAT_R64_SFLOAT = 112,
  RHI_FORMAT_R64G64_UINT = 113,
  RHI_FORMAT_R64G64_SINT = 114,
  RHI_FORMAT_R64G64_SFLOAT = 115,
  RHI_FORMAT_R64G64B64_UINT = 116,
  RHI_FORMAT_R64G64B64_SINT = 117,
  RHI_FORMAT_R64G64B64_SFLOAT = 118,
  RHI_FORMAT_R64G64B64A64_UINT = 119,
  RHI_FORMAT_R64G64B64A64_SINT = 120,
  RHI_FORMAT_R64G64B64A64_SFLOAT = 121,
  RHI_FORMAT_B10G11R11_UFLOAT_PACK32 = 122,
  RHI_FORMAT_E5B9G9R9_UFLOAT_PACK32 = 123,
  RHI_FORMAT_D16_UNORM = 124,
  RHI_FORMAT_X8_D24_UNORM_PACK32 = 125,
  RHI_FORMAT_D32_SFLOAT = 126,
  RHI_FORMAT_S8_UINT = 127,
  RHI_FORMAT_D16_UNORM_S8_UINT = 128,
  RHI_FORMAT_D24_UNORM_S8_UINT = 129,
  RHI_FORMAT_D32_SFLOAT_S8_UINT = 130,
  RHI_FORMAT_BC1_RGB_UNORM_BLOCK = 131,
  RHI_FORMAT_BC1_RGB_SRGB_BLOCK = 132,
  RHI_FORMAT_BC1_RGBA_UNORM_BLOCK = 133,
  RHI_FORMAT_BC1_RGBA_SRGB_BLOCK = 134,
  RHI_FORMAT_BC2_UNORM_BLOCK = 135,
  RHI_FORMAT_BC2_SRGB_BLOCK = 136,
  RHI_FORMAT_BC3_UNORM_BLOCK = 137,
  RHI_FORMAT_BC3_SRGB_BLOCK = 138,
  RHI_FORMAT_BC4_UNORM_BLOCK = 139,
  RHI_FORMAT_BC4_SNORM_BLOCK = 140,
  RHI_FORMAT_BC5_UNORM_BLOCK = 141,
  RHI_FORMAT_BC5_SNORM_BLOCK = 142,
  RHI_FORMAT_BC6H_UFLOAT_BLOCK = 143,
  RHI_FORMAT_BC6H_SFLOAT_BLOCK = 144,
  RHI_FORMAT_BC7_UNORM_BLOCK = 145,
  RHI_FORMAT_BC7_SRGB_BLOCK = 146,
  RHI_FORMAT_ETC2_R8G8B8_UNORM_BLOCK = 147,
  RHI_FORMAT_ETC2_R8G8B8_SRGB_BLOCK = 148,
  RHI_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK = 149,
  RHI_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK = 150,
  RHI_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK = 151,
  RHI_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK = 152,
  RHI_FORMAT_EAC_R11_UNORM_BLOCK = 153,
  RHI_FORMAT_EAC_R11_SNORM_BLOCK = 154,
  RHI_FORMAT_EAC_R11G11_UNORM_BLOCK = 155,
  RHI_FORMAT_EAC_R11G11_SNORM_BLOCK = 156,
  RHI_FORMAT_ASTC_4x4_UNORM_BLOCK = 157,
  RHI_FORMAT_ASTC_4x4_SRGB_BLOCK = 158,
  RHI_FORMAT_ASTC_5x4_UNORM_BLOCK = 159,
  RHI_FORMAT_ASTC_5x4_SRGB_BLOCK = 160,
  RHI_FORMAT_ASTC_5x5_UNORM_BLOCK = 161,
  RHI_FORMAT_ASTC_5x5_SRGB_BLOCK = 162,
  RHI_FORMAT_ASTC_6x5_UNORM_BLOCK = 163,
  RHI_FORMAT_ASTC_6x5_SRGB_BLOCK = 164,
  RHI_FORMAT_ASTC_6x6_UNORM_BLOCK = 165,
  RHI_FORMAT_ASTC_6x6_SRGB_BLOCK = 166,
  RHI_FORMAT_ASTC_8x5_UNORM_BLOCK = 167,
  RHI_FORMAT_ASTC_8x5_SRGB_BLOCK = 168,
  RHI_FORMAT_ASTC_8x6_UNORM_BLOCK = 169,
  RHI_FORMAT_ASTC_8x6_SRGB_BLOCK = 170,
  RHI_FORMAT_ASTC_8x8_UNORM_BLOCK = 171,
  RHI_FORMAT_ASTC_8x8_SRGB_BLOCK = 172,
  RHI_FORMAT_ASTC_10x5_UNORM_BLOCK = 173,
  RHI_FORMAT_ASTC_10x5_SRGB_BLOCK = 174,
  RHI_FORMAT_ASTC_10x6_UNORM_BLOCK = 175,
  RHI_FORMAT_ASTC_10x6_SRGB_BLOCK = 176,
  RHI_FORMAT_ASTC_10x8_UNORM_BLOCK = 177,
  RHI_FORMAT_ASTC_10x8_SRGB_BLOCK = 178,
  RHI_FORMAT_ASTC_10x10_UNORM_BLOCK = 179,
  RHI_FORMAT_ASTC_10x10_SRGB_BLOCK = 180,
  RHI_FORMAT_ASTC_12x10_UNORM_BLOCK = 181,
  RHI_FORMAT_ASTC_12x10_SRGB_BLOCK = 182,
  RHI_FORMAT_ASTC_12x12_UNORM_BLOCK = 183,
  RHI_FORMAT_ASTC_12x12_SRGB_BLOCK = 184
};

enum class RhiShaderStage { Vertex, Fragment, Compute, Mesh, Task };
enum class RhiShaderSourceType { GLSLCode, Binary };
enum class RhiVertexInputRate { Vertex, Instance };

enum class RhiDescriptorBindPoint { Compute, Graphics };

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

enum class RhiResourceType { Buffer, Texture };
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

enum class RhiResourceState {
  Undefined,
  Common,
  RenderTarget,
  DepthStencilRenderTarget,
  Present,
  UAVStorageImage,
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
    RhiBuffer *m_buffer;
    RhiTexture *m_texture;
  };
  RhiResourceState m_srcState;
  RhiResourceState m_dstState;
};

struct RhiResourceBarrier {
  RhiBarrierType m_type;
  union {
    RhiUAVBarrier m_uav;
    RhiTransitionBarrier m_transition;
  };
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
  virtual RhiBuffer *createBuffer(uint32_t size, uint32_t usage,
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
  virtual RhiTexture *createDepthRenderTexture(uint32_t width,
                                               uint32_t height) = 0;

  virtual std::shared_ptr<RhiBuffer> getFullScreenQuadVertexBuffer() const = 0;

  // Note that the texture created can only be accessed by the GPU
  virtual std::shared_ptr<RhiTexture>
  createRenderTargetTexture(uint32_t width, uint32_t height,
                            RhiImageFormat format, uint32_t extraFlags) = 0;

  virtual std::shared_ptr<RhiTexture>
  createRenderTargetMipTexture(uint32_t width, uint32_t height, uint32_t mips,
                               RhiImageFormat format, uint32_t extraFlags) = 0;

  virtual std::shared_ptr<RhiSampler> createTrivialSampler() = 0;

  virtual std::shared_ptr<Rhi::RhiStagedSingleBuffer>
  createStagedSingleBuffer(RhiBuffer *target) = 0;

  // Command execution
  virtual RhiQueue *getQueue(RhiQueueCapability req) = 0;
  virtual RhiShader *createShader(const std::vector<char> &code,
                                  std::string entry, Rhi::RhiShaderStage stage,
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

  virtual std::shared_ptr<Rhi::RhiBindlessIdRef>
  registerUAVImage(Rhi::RhiTexture *texture,
                   Rhi::RhiImageSubResource subResource) = 0;

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
  virtual uint32_t getHeight() = 0;
  virtual uint32_t getWidth() = 0;
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

struct RhiImageSubResource {
  uint32_t mipLevel;
  uint32_t arrayLayer;
  uint32_t mipCount = 1;
  uint32_t layerCount = 1;
};

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
};

class IFRIT_APIDECL RhiColorAttachment {
protected:
  virtual int _polymorphismPlaceHolder() { return 0; }
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