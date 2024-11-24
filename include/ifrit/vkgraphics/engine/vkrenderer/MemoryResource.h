#pragma once
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include <memory>
#include <unordered_map>

namespace Ifrit::GraphicsBackend::VulkanGraphics {
enum class BufferMemoryType { DeviceLocal, HostLocal };

struct BufferCreateInfo {
  uint32_t size;
  VkFlags usage;
  bool hostVisible = false;
};

class IFRIT_APIDECL SingleBuffer : public Rhi::RhiBuffer {
protected:
  VkBuffer m_buffer;
  EngineContext *m_context;
  VmaAllocation m_allocation;
  VmaAllocationInfo m_allocInfo;
  BufferCreateInfo m_createInfo;
  char *m_mappedMemory = nullptr;
  bool m_created = false;

protected:
  void init();

public:
  SingleBuffer() : m_created(false) {}
  SingleBuffer(EngineContext *ctx, const BufferCreateInfo &ci)
      : m_context(ctx), m_createInfo(ci), m_created(true) {
    init();
  }
  virtual ~SingleBuffer();
  virtual void map() override;
  virtual void unmap() override;
  virtual void readBuffer(void *data, uint32_t size, uint32_t offset) override;
  virtual void writeBuffer(const void *data, uint32_t size,
                           uint32_t offset) override;
  virtual void flush() override;
  inline VkBuffer getBuffer() const { return m_buffer; }
  inline VkFlags getUsage() const { return m_createInfo.usage; }
  inline uint32_t getSize() const { return m_createInfo.size; }
  inline BufferCreateInfo getCreateInfo() const { return m_createInfo; }
};

class IFRIT_APIDECL MultiBuffer : public Rhi::RhiMultiBuffer {
protected:
  std::vector<std::unique_ptr<SingleBuffer>> m_buffersOwning;
  std::vector<SingleBuffer *> m_buffers;
  BufferCreateInfo m_createInfo;
  EngineContext *m_context;
  uint32_t m_activeFrame = 0;

public:
  MultiBuffer(EngineContext *ctx, const BufferCreateInfo &ci,
              uint32_t numCopies);
  MultiBuffer(const MultiBuffer &p) = delete;
  MultiBuffer(EngineContext *ctx, const std::vector<SingleBuffer *> &buffers);
  MultiBuffer &operator=(const MultiBuffer &p) = delete;
  SingleBuffer *getBuffer(uint32_t index);
  inline Rhi::RhiBuffer *getActiveBuffer() override {
    return m_buffers[m_activeFrame];
  }
  inline void advanceFrame() {
    m_activeFrame++;
    m_activeFrame %= m_buffers.size();
  }
  inline uint32_t getActiveFrame() { return m_activeFrame; }
  inline void setActiveFrame(uint32_t frame) { m_activeFrame = frame; }
  inline uint32_t getBufferCount() {
    using namespace Ifrit::Common::Utility;
    return size_cast<int>(m_buffers.size());
  }
  inline Rhi::RhiBuffer *getActiveBufferRelative(uint32_t deltaFrame) override {
    return m_buffers[(m_activeFrame + deltaFrame) % m_buffers.size()];
  }
};

enum class ImageType { Image2D, Image3D, ImageCube };
enum class ImageAspect { Color, Depth, Stencil };

struct ImageCreateInfo {
  VkFormat format;
  VkImageUsageFlags usage;
  ImageAspect aspect = ImageAspect::Color;
  ImageType type = ImageType::Image2D;
  uint32_t width;
  uint32_t height;
  uint32_t depth = 1;
  uint32_t mipLevels = 1;
  uint32_t arrayLayers = 1;
  bool hostVisible = false;
};

class IFRIT_APIDECL SingleDeviceImage : public Rhi::RhiTexture {
protected:
  EngineContext *m_context;
  VkFormat m_format;
  VkImage m_image;
  VkImageView m_imageView = nullptr;
  VmaAllocation m_allocation;
  VmaAllocationInfo m_allocInfo;
  ImageCreateInfo m_createInfo{};
  bool m_isSwapchainImage = false;
  bool m_created = false;

  std::unordered_map<uint64_t, VkImageView> m_managedImageViews;

public:
  SingleDeviceImage() {}
  virtual ~SingleDeviceImage();
  SingleDeviceImage(EngineContext *ctx, const ImageCreateInfo &ci);

  virtual VkFormat getFormat() const;
  virtual VkImage getImage() const;
  virtual VkImageView getImageView();
  virtual VkImageView getImageViewMipLayer(uint32_t mipLevel, uint32_t layer,
                                           uint32_t mipRange,
                                           uint32_t layerRange);
  inline bool getIsSwapchainImage() { return m_isSwapchainImage; }
  uint32_t getSize();
  inline uint32_t getWidth() const override { return m_createInfo.width; }
  inline uint32_t getHeight() const override { return m_createInfo.height; }
  inline uint32_t getDepth() const { return m_createInfo.depth; }
  inline VkImageAspectFlags getAspect() const {
    if (m_createInfo.aspect == ImageAspect::Color) {
      return VK_IMAGE_ASPECT_COLOR_BIT;
    } else if (m_createInfo.aspect == ImageAspect::Depth) {
      return VK_IMAGE_ASPECT_DEPTH_BIT;
    } else if (m_createInfo.aspect == ImageAspect::Stencil) {
      return VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
  }
  inline uint32_t getMipLevels() { return m_createInfo.mipLevels; }
  inline uint32_t getArrayLayers() { return m_createInfo.arrayLayers; }
};

struct SamplerCreateInfo {
  VkFilter magFilter = VK_FILTER_LINEAR;
  VkFilter minFilter = VK_FILTER_LINEAR;
  VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  VkSamplerAddressMode addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  VkSamplerAddressMode addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  VkSamplerAddressMode addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  float mipLodBias = 0.0f;
  VkBool32 anisotropyEnable = VK_FALSE;
  float maxAnisotropy = -1;
  VkBool32 compareEnable = VK_FALSE;
  VkCompareOp compareOp = VK_COMPARE_OP_ALWAYS;
  float minLod = 0.0f;
  float maxLod = 0.0f;
  VkBorderColor borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  VkBool32 unnormalizedCoordinates = VK_FALSE;
};

class IFRIT_APIDECL Sampler : public Rhi::RhiSampler {
protected:
  EngineContext *m_context;
  VkSampler m_sampler;
  SamplerCreateInfo m_createInfo;

public:
  Sampler(EngineContext *ctx, const SamplerCreateInfo &ci);
  ~Sampler();
  inline VkSampler getSampler() { return m_sampler; }
};

class IFRIT_APIDECL ResourceManager {
protected:
  EngineContext *m_context;
  std::vector<std::unique_ptr<SingleBuffer>> m_simpleBuffer;
  std::vector<std::unique_ptr<MultiBuffer>> m_multiBuffer;
  std::vector<uint32_t> m_multiBufferTraced;
  std::vector<std::unique_ptr<SingleDeviceImage>> m_simpleImage;
  std::vector<std::unique_ptr<Sampler>> m_samplers;
  int32_t m_defaultCopies = -1;

public:
  ResourceManager(EngineContext *ctx) : m_context(ctx) {}
  virtual ~ResourceManager() {}
  ResourceManager(const ResourceManager &p) = delete;
  ResourceManager &operator=(const ResourceManager &p) = delete;

  MultiBuffer *createMultipleBuffer(const BufferCreateInfo &ci,
                                    uint32_t numCopies = UINT32_MAX);
  MultiBuffer *createTracedMultipleBuffer(const BufferCreateInfo &ci,
                                          uint32_t numCopies = UINT32_MAX);
  SingleBuffer *createSimpleBuffer(const BufferCreateInfo &ci);
  SingleDeviceImage *createSimpleImage(const ImageCreateInfo &ci);
  Sampler *createSampler(const SamplerCreateInfo &ci);

  void setActiveFrame(uint32_t frame);
  inline void setDefaultCopies(int32_t copies) { m_defaultCopies = copies; }

  // Previous design is TOO UGLY, it saves UNUSED buffer and image
  // for new interfaces, following methods will be used to create mem resources
  // TODO: refactor the code
  std::shared_ptr<SingleDeviceImage>
  createSimpleImageUnmanaged(const ImageCreateInfo &ci);

  std::shared_ptr<SingleBuffer>
  createSimpleBufferUnmanaged(const BufferCreateInfo &ci);

  // Quick shortcuts

  // Create a storage buffer that used to transfer data from host side
  // EVERY FRAME. It accepts data from staging buffer or host. Double
  // buffering will be used in this buffer.
  MultiBuffer *createStorageBufferShared(uint32_t size, bool hostVisible,
                                         VkFlags extraFlags = 0);

  // Create a storage buffer that only works on GPU side.
  SingleBuffer *createStorageBufferDevice(uint32_t size,
                                          VkFlags extraFlags = 0);

  // Create a indirect mesh draw buffer and the target for compute shader
  SingleBuffer *createIndirectMeshDrawBufferDevice(uint32_t numDrawCalls,
                                                   VkFlags extraFlags = 0);

  // Create a uniform buffer that will be used to transfer data from host
  MultiBuffer *createUniformBufferShared(uint32_t size, bool hostVisible,
                                         VkFlags extraFlags = 0);

  // Create a vertex buffer.
  SingleBuffer *createVertexBufferDevice(uint32_t size, VkFlags extraFlags = 0);

  // Create an index buffer.
  SingleBuffer *createIndexBufferDevice(uint32_t size, VkFlags extraFlags = 0);

  // Create an proxy multi buffer
  MultiBuffer *
  createProxyMultiBuffer(const std::vector<SingleBuffer *> &buffers);

  // Create a depth attachment
  SingleDeviceImage *
  createDepthAttachment(uint32_t width, uint32_t height,
                        VkFormat format = VK_FORMAT_D32_SFLOAT,
                        VkImageUsageFlags extraUsage = 0);

  // Create a readable render target texture
  std::shared_ptr<SingleDeviceImage>
  createRenderTargetTexture(uint32_t width, uint32_t height, VkFormat format,
                            VkImageUsageFlags extraUsage = 0);

  // Create a readable render target texture with mipLevels
  std::shared_ptr<SingleDeviceImage>
  createRenderTargetMipTexture(uint32_t width, uint32_t height, uint32_t mips,
                               VkFormat format,
                               VkImageUsageFlags extraUsage = 0);

  // Create a device only texture, intended for shader read.
  SingleDeviceImage *createTexture2DDevice(uint32_t width, uint32_t height,
                                           VkFormat format,
                                           VkImageUsageFlags extraUsage = 0);

  // Create a simple sampler
  std::shared_ptr<Sampler> createTrivialRenderTargetSampler();

  // Create a bilinear sampler
  std::shared_ptr<Sampler> createTrivialBilinearSampler();
};

} // namespace Ifrit::GraphicsBackend::VulkanGraphics