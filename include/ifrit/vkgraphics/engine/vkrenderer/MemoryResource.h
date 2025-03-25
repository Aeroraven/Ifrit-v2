
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
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include <memory>
#include <unordered_map>

namespace Ifrit::GraphicsBackend::VulkanGraphics {
enum class BufferMemoryType { DeviceLocal, HostLocal };

struct BufferCreateInfo {
  u32 size;
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
  u64 m_deviceAddress = 0;

protected:
  void init();

public:
  SingleBuffer(EngineContext *ctx, const BufferCreateInfo &ci)
      : Rhi::RhiBuffer(ctx->getDeleteQueue()), m_context(ctx), m_createInfo(ci), m_created(true) {
    init();
  }
  virtual ~SingleBuffer();
  virtual void map() override;
  virtual void unmap() override;
  virtual void readBuffer(void *data, u32 size, u32 offset) override;
  virtual void writeBuffer(const void *data, u32 size, u32 offset) override;
  virtual void flush() override;
  inline VkBuffer getBuffer() const { return m_buffer; }
  inline VkFlags getUsage() const { return m_createInfo.usage; }
  inline u32 getSize() const { return m_createInfo.size; }
  inline BufferCreateInfo getCreateInfo() const { return m_createInfo; }

  virtual Rhi::RhiDeviceAddr getDeviceAddress() const;
};

class IFRIT_APIDECL MultiBuffer : public Rhi::RhiMultiBuffer {
protected:
  Vec<Rhi::RhiBufferRef> m_buffersOwning;
  Vec<SingleBuffer *> m_buffers;
  BufferCreateInfo m_createInfo;
  EngineContext *m_context;
  u32 m_activeFrame = 0;

public:
  MultiBuffer(EngineContext *ctx, const BufferCreateInfo &ci, u32 numCopies);
  MultiBuffer(const MultiBuffer &p) = delete;
  MultiBuffer &operator=(const MultiBuffer &p) = delete;
  SingleBuffer *getBuffer(u32 index);
  inline Rhi::RhiBuffer *getActiveBuffer() override { return m_buffers[m_activeFrame]; }
  inline void advanceFrame() {
    m_activeFrame++;
    m_activeFrame %= m_buffers.size();
  }
  inline u32 getActiveFrame() { return m_activeFrame; }
  inline void setActiveFrame(u32 frame) { m_activeFrame = frame; }
  inline u32 getBufferCount() {
    using namespace Ifrit::Common::Utility;
    return size_cast<int>(m_buffers.size());
  }
  inline Rhi::RhiBuffer *getActiveBufferRelative(u32 deltaFrame) override {
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
  u32 width;
  u32 height;
  u32 depth = 1;
  u32 mipLevels = 1;
  u32 arrayLayers = 1;
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

  std::unordered_map<u64, VkImageView> m_managedImageViews;

public:
  virtual ~SingleDeviceImage();
  explicit SingleDeviceImage(nullptr_t v) : Rhi::RhiTexture(nullptr) {}
  SingleDeviceImage(EngineContext *ctx, const ImageCreateInfo &ci);

  virtual VkFormat getFormat() const;
  virtual VkImage getImage() const;
  virtual VkImageView getImageView();
  virtual VkImageView getImageViewMipLayer(u32 mipLevel, u32 layer, u32 mipRange, u32 layerRange);
  inline bool getIsSwapchainImage() { return m_isSwapchainImage; }
  u32 getSize();
  inline u32 getWidth() const override { return m_createInfo.width; }
  inline u32 getHeight() const override { return m_createInfo.height; }
  inline u32 getDepth() const { return m_createInfo.depth; }
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
  inline u32 getMipLevels() { return m_createInfo.mipLevels; }
  inline u32 getArrayLayers() { return m_createInfo.arrayLayers; }
  inline bool isDepthTexture() const override { return m_createInfo.aspect == ImageAspect::Depth; }
  inline void *getNativeHandle() const override { return m_image; }
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
  Vec<Ref<MultiBuffer>> m_multiBuffer;
  Vec<u32> m_multiBufferTraced;
  i32 m_defaultCopies = -1;

public:
  ResourceManager(EngineContext *ctx) : m_context(ctx) {}
  virtual ~ResourceManager() {}
  ResourceManager(const ResourceManager &p) = delete;
  ResourceManager &operator=(const ResourceManager &p) = delete;

  Ref<MultiBuffer> createMultipleBuffer(const BufferCreateInfo &ci, u32 numCopies = UINT32_MAX);
  Ref<MultiBuffer> createTracedMultipleBuffer(const BufferCreateInfo &ci, u32 numCopies = UINT32_MAX);

  void setActiveFrame(u32 frame);
  inline void setDefaultCopies(i32 copies) { m_defaultCopies = copies; }

  // Previous design is TOO UGLY, it saves UNUSED buffer and image
  // for new interfaces, following methods will be used to create mem resources
  // TODO: refactor the code
  Rhi::RhiTextureRef createSimpleImageUnmanaged(const ImageCreateInfo &ci);
  Rhi::RhiBufferRef createSimpleBufferUnmanaged(const BufferCreateInfo &ci);
  Rhi::RhiTextureRef createDepthAttachment(u32 width, u32 height, VkFormat format = VK_FORMAT_D32_SFLOAT,
                                           VkImageUsageFlags extraUsage = 0);

  Rhi::RhiTextureRef createTexture2DDeviceUnmanaged(u32 width, u32 height, VkFormat format,
                                                    VkImageUsageFlags extraUsage = 0);

  Rhi::RhiTextureRef createRenderTargetTexture(u32 width, u32 height, VkFormat format,
                                               VkImageUsageFlags extraUsage = 0);
  Rhi::RhiTextureRef createTexture3D(u32 width, u32 height, u32 depth, VkFormat format,
                                     VkImageUsageFlags extraUsage = 0);
  Rhi::RhiTextureRef createMipTexture(u32 width, u32 height, u32 mips, VkFormat format,
                                      VkImageUsageFlags extraUsage = 0);
  Rhi::RhiSamplerRef createTrivialRenderTargetSampler();
  Rhi::RhiSamplerRef createTrivialBilinearSampler(bool repeat);
  Rhi::RhiSamplerRef createTrivialNearestSampler(bool repeat);
};

} // namespace Ifrit::GraphicsBackend::VulkanGraphics