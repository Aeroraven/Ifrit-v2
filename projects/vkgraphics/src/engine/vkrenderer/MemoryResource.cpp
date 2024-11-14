#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/utility/Logger.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::GraphicsBackend::VulkanGraphics {
IFRIT_APIDECL void SingleBuffer::init() {
  VkBufferCreateInfo bufferCI{};
  bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCI.size = m_createInfo.size;
  bufferCI.usage = m_createInfo.usage;
  bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = VMA_MEMORY_USAGE_AUTO;
  if (m_createInfo.hostVisible) {
    allocCI.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  vkrVulkanAssert(vmaCreateBuffer(m_context->getAllocator(), &bufferCI,
                                  &allocCI, &m_buffer, &m_allocation,
                                  &m_allocInfo),
                  "Failed to create buffer");
}
IFRIT_APIDECL SingleBuffer::~SingleBuffer() {
  vmaDestroyBuffer(m_context->getAllocator(), m_buffer, m_allocation);
}
IFRIT_APIDECL void SingleBuffer::map() {
  vkrAssert(m_createInfo.hostVisible, "Buffer must be host visible to map");
  vmaMapMemory(m_context->getAllocator(), m_allocation,
               (void **)&m_mappedMemory);
}
IFRIT_APIDECL void SingleBuffer::unmap() {
  vkrAssert(m_mappedMemory, "Buffer must be mapped to unmap");
  vmaUnmapMemory(m_context->getAllocator(), m_allocation);
  m_mappedMemory = nullptr;
}
IFRIT_APIDECL void SingleBuffer::readBuffer(void *data, uint32_t size,
                                            uint32_t offset) {
  vkrAssert(m_mappedMemory, "Buffer must be mapped to copy data");
  memcpy(m_mappedMemory + offset, data, size);
}
IFRIT_APIDECL void SingleBuffer::writeBuffer(const void *data, uint32_t size,
                                             uint32_t offset) {
  vkrAssert(m_mappedMemory, "Buffer must be mapped to copy data");
  memcpy((void *)(m_mappedMemory + offset), data, size);
}
IFRIT_APIDECL void SingleBuffer::flush() {
  vmaFlushAllocation(m_context->getAllocator(), m_allocation, 0, VK_WHOLE_SIZE);
}

// Class: MultiBuffer
IFRIT_APIDECL MultiBuffer::MultiBuffer(EngineContext *ctx,
                                       const BufferCreateInfo &ci,
                                       uint32_t numCopies)
    : m_context(ctx), m_createInfo(ci) {
  for (uint32_t i = 0; i < numCopies; i++) {
    m_buffersOwning.push_back(std::make_unique<SingleBuffer>(ctx, ci));
    m_buffers.push_back(m_buffersOwning.back().get());
  }
}

IFRIT_APIDECL
MultiBuffer::MultiBuffer(EngineContext *ctx,
                         const std::vector<SingleBuffer *> &buffers) {
  // Assert all buffers have the same create info
  m_context = ctx;
  m_buffers = buffers;
  m_createInfo = buffers[0]->getCreateInfo();
  for (auto &buffer : buffers) {
    vkrAssert(buffer->getCreateInfo().size == m_createInfo.size,
              "Buffer size must be the same");
    vkrAssert(buffer->getCreateInfo().usage == m_createInfo.usage,
              "Buffer usage must be the same");
  }
}
IFRIT_APIDECL SingleBuffer *MultiBuffer::getBuffer(uint32_t index) {
  return m_buffers[index];
}

// Class: Image
IFRIT_APIDECL VkFormat SingleDeviceImage::getFormat() const { return m_format; }
IFRIT_APIDECL VkImage SingleDeviceImage::getImage() const { return m_image; }
IFRIT_APIDECL VkImageView SingleDeviceImage::getImageView() {
  return m_imageView;
}

IFRIT_APIDECL VkImageView SingleDeviceImage::getImageViewMipLayer(
    uint32_t mipLevel, uint32_t layer, uint32_t mipRange, uint32_t layerRange) {
  auto key = (static_cast<uint64_t>(mipLevel) << 16) |
             (static_cast<uint64_t>(layer) << 32) |
             (static_cast<uint64_t>(mipRange) << 48) |
             (static_cast<uint64_t>(layerRange) << 56);
  if (m_managedImageViews.count(key)) {
    return m_managedImageViews[key];
  }

  // create image view
  auto &ci = m_createInfo;
  VkImageViewCreateInfo imageViewCI{};
  imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  imageViewCI.image = m_image;
  if (ci.type == ImageType::ImageCube) {
    imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  } else if (ci.type == ImageType::Image2D) {
    imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
  } else if (ci.type == ImageType::Image3D) {
    imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_3D;
  }
  imageViewCI.format = ci.format;
  if (ci.aspect == ImageAspect::Color) {
    imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  } else if (ci.aspect == ImageAspect::Depth) {
    imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  } else if (ci.aspect == ImageAspect::Stencil) {
    imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
  }
  imageViewCI.subresourceRange.baseMipLevel = mipLevel;
  imageViewCI.subresourceRange.levelCount = mipRange;
  imageViewCI.subresourceRange.baseArrayLayer = layer;
  imageViewCI.subresourceRange.layerCount = layerRange;

  VkImageView imageView;
  vkrVulkanAssert(vkCreateImageView(m_context->getDevice(), &imageViewCI,
                                    nullptr, &imageView),
                  "Failed to create image view");
  m_managedImageViews[key] = imageView;
  return imageView;
}

IFRIT_APIDECL SingleDeviceImage::~SingleDeviceImage() {
  if (m_created) {
    vmaDestroyImage(m_context->getAllocator(), m_image, m_allocation);
    for (auto &[k, v] : m_managedImageViews) {
      vkDestroyImageView(m_context->getDevice(), v, nullptr);
    }
  }
}
IFRIT_APIDECL SingleDeviceImage::SingleDeviceImage(EngineContext *ctx,
                                                   const ImageCreateInfo &ci) {
  m_context = ctx;
  m_createInfo = ci;
  VkImageCreateInfo imageCI{};
  imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

  if (ci.type == ImageType::Image2D) {
    imageCI.imageType = VK_IMAGE_TYPE_2D;
  } else if (ci.type == ImageType::Image3D) {
    imageCI.imageType = VK_IMAGE_TYPE_3D;
  } else if (ci.type == ImageType::ImageCube) {
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  }

  imageCI.format = ci.format;
  imageCI.extent.width = ci.width;
  imageCI.extent.height = ci.height;
  imageCI.extent.depth = ci.depth;
  imageCI.mipLevels = ci.mipLevels;
  imageCI.arrayLayers = ci.arrayLayers;
  imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
  imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageCI.usage = ci.usage;
  imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageCI.flags = 0;
  m_format = ci.format;
  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = VMA_MEMORY_USAGE_AUTO;
  if (ci.hostVisible) {
    allocCI.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  vkrVulkanAssert(vmaCreateImage(ctx->getAllocator(), &imageCI, &allocCI,
                                 &m_image, &m_allocation, &m_allocInfo),
                  "Failed to create image");
  m_imageView = this->getImageViewMipLayer(0, 0, ci.mipLevels, ci.arrayLayers);
  m_created = true;
}

IFRIT_APIDECL uint32_t SingleDeviceImage::getSize() {
  auto pixelNums =
      m_createInfo.width * m_createInfo.height * m_createInfo.depth;
  auto pixelSize = 0;

#define FORMAT_MAP(format, pxSize)                                             \
  case format:                                                                 \
    pixelSize = pxSize;                                                        \
    break;
  switch (m_createInfo.format) {
    FORMAT_MAP(VK_FORMAT_R8G8B8A8_UNORM, 4)
    FORMAT_MAP(VK_FORMAT_R8G8B8A8_SRGB, 4)
    FORMAT_MAP(VK_FORMAT_R32G32B32A32_SFLOAT, 16)
    FORMAT_MAP(VK_FORMAT_R32G32B32_SFLOAT, 12)
    FORMAT_MAP(VK_FORMAT_R32G32_SFLOAT, 8)
    FORMAT_MAP(VK_FORMAT_R32_SFLOAT, 4)
    FORMAT_MAP(VK_FORMAT_R32_UINT, 4)
    FORMAT_MAP(VK_FORMAT_R32G32B32A32_UINT, 16)
    FORMAT_MAP(VK_FORMAT_R32G32B32_UINT, 12)
    FORMAT_MAP(VK_FORMAT_R32G32_UINT, 8)
    FORMAT_MAP(VK_FORMAT_R32G32B32A32_SINT, 16)
    FORMAT_MAP(VK_FORMAT_R32G32B32_SINT, 12)
    FORMAT_MAP(VK_FORMAT_R32G32_SINT, 8)
    FORMAT_MAP(VK_FORMAT_R32_SINT, 4)

  default:
    vkrError("Unsupported format");
  }
#undef FORMAT_MAP
  return pixelNums * pixelSize;
}

// Class: Sampler
IFRIT_APIDECL Sampler::Sampler(EngineContext *ctx,
                               const SamplerCreateInfo &ci) {
  m_context = ctx;
  VkSamplerCreateInfo samplerCI{};
  samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCI.magFilter = ci.magFilter;
  samplerCI.minFilter = ci.minFilter;
  samplerCI.addressModeU = ci.addressModeU;
  samplerCI.addressModeV = ci.addressModeV;
  samplerCI.addressModeW = ci.addressModeW;
  if (ctx->getPhysicalDeviceProperties().limits.maxSamplerAnisotropy < 1.0f) {
    samplerCI.anisotropyEnable = VK_FALSE;
    samplerCI.maxAnisotropy = 1.0f;
    if (ci.anisotropyEnable) {
      vkrLog("Anisotropy not supported. Feature disabled");
    }
  } else {
    samplerCI.anisotropyEnable = VK_TRUE;
    samplerCI.maxAnisotropy = ci.maxAnisotropy;
    if (samplerCI.maxAnisotropy < 0.0f) {
      samplerCI.maxAnisotropy =
          ctx->getPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
    }
  }

  samplerCI.borderColor = ci.borderColor;
  samplerCI.unnormalizedCoordinates = ci.unnormalizedCoordinates;
  samplerCI.compareEnable = ci.compareEnable;
  samplerCI.compareOp = ci.compareOp;
  samplerCI.mipmapMode = ci.mipmapMode;
  samplerCI.mipLodBias = ci.mipLodBias;
  samplerCI.minLod = ci.minLod;
  samplerCI.maxLod = ci.maxLod;

  vkrVulkanAssert(
      vkCreateSampler(ctx->getDevice(), &samplerCI, nullptr, &m_sampler),
      "Failed to create sampler");
}

IFRIT_APIDECL Sampler::~Sampler() {
  vkDestroySampler(m_context->getDevice(), m_sampler, nullptr);
}

// Class: ResourceManager
IFRIT_APIDECL MultiBuffer *
ResourceManager::createMultipleBuffer(const BufferCreateInfo &ci,
                                      uint32_t numCopies) {
  if (numCopies == UINT32_MAX) {
    if (m_defaultCopies == -1) {
      vkrError("No default copies set for multiple buffer");
    }
    numCopies = m_defaultCopies;
  }
  auto buffer = std::make_unique<MultiBuffer>(m_context, ci, numCopies);
  auto ptr = buffer.get();
  m_multiBuffer.push_back(std::move(buffer));
  return ptr;
}

IFRIT_APIDECL SingleBuffer *
ResourceManager::createSimpleBuffer(const BufferCreateInfo &ci) {
  auto buffer = std::make_unique<SingleBuffer>(m_context, ci);
  auto ptr = buffer.get();
  m_simpleBuffer.push_back(std::move(buffer));
  return ptr;
}
IFRIT_APIDECL SingleDeviceImage *
ResourceManager::createSimpleImage(const ImageCreateInfo &ci) {
  auto image = std::make_unique<SingleDeviceImage>(m_context, ci);
  auto ptr = image.get();
  m_simpleImage.push_back(std::move(image));
  return ptr;
}

IFRIT_APIDECL std::shared_ptr<SingleDeviceImage>
ResourceManager::createSimpleImageUnmanaged(const ImageCreateInfo &ci) {
  auto image = std::make_shared<SingleDeviceImage>(m_context, ci);
  return image;
}

IFRIT_APIDECL std::shared_ptr<SingleBuffer>
ResourceManager::createSimpleBufferUnmanaged(const BufferCreateInfo &ci) {
  auto buffer = std::make_shared<SingleBuffer>(m_context, ci);
  return buffer;
}

IFRIT_APIDECL Sampler *
ResourceManager::createSampler(const SamplerCreateInfo &ci) {
  auto sampler = std::make_unique<Sampler>(m_context, ci);
  auto ptr = sampler.get();
  m_samplers.push_back(std::move(sampler));
  return ptr;
}

IFRIT_APIDECL MultiBuffer *
ResourceManager::createTracedMultipleBuffer(const BufferCreateInfo &ci,
                                            uint32_t numCopies) {
  if (numCopies == UINT32_MAX) {
    if (m_defaultCopies == -1) {
      vkrError("No default copies set for multiple buffer");
    }
    numCopies = m_defaultCopies;
  }
  auto buffer = std::make_unique<MultiBuffer>(m_context, ci, numCopies);
  auto ptr = buffer.get();
  m_multiBuffer.push_back(std::move(buffer));
  m_multiBufferTraced.push_back(size_cast<int>(m_multiBuffer.size()) - 1);
  return ptr;
}

IFRIT_APIDECL void ResourceManager::setActiveFrame(uint32_t frame) {
  for (int i = 0; i < m_multiBufferTraced.size(); i++) {
    m_multiBuffer[m_multiBufferTraced[i]]->setActiveFrame(frame);
  }
}

// Shortcut methods
IFRIT_APIDECL MultiBuffer *
ResourceManager::createStorageBufferShared(uint32_t size, bool hostVisible,
                                           VkFlags extraFlags) {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extraFlags;
  ci.hostVisible = hostVisible;
  return createTracedMultipleBuffer(ci);
}

IFRIT_APIDECL SingleBuffer *
ResourceManager::createStorageBufferDevice(uint32_t size, VkFlags extraFlags) {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extraFlags;
  ci.hostVisible = false;
  return createSimpleBuffer(ci);
}

IFRIT_APIDECL SingleBuffer *
ResourceManager::createIndirectMeshDrawBufferDevice(uint32_t numDrawCalls,
                                                    VkFlags extraFlags) {
  BufferCreateInfo ci{};
  ci.size = numDrawCalls * sizeof(VkDrawMeshTasksIndirectCommandEXT);
  ci.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extraFlags;
  ci.hostVisible = false;
  return createSimpleBuffer(ci);
}

IFRIT_APIDECL MultiBuffer *
ResourceManager::createUniformBufferShared(uint32_t size, bool hostVisible,
                                           VkFlags extraFlags) {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | extraFlags;
  ci.hostVisible = hostVisible;
  return createTracedMultipleBuffer(ci);
}

IFRIT_APIDECL SingleBuffer *
ResourceManager::createVertexBufferDevice(uint32_t size, VkFlags extraFlags) {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | extraFlags;
  ci.hostVisible = false;
  return createSimpleBuffer(ci);
}

IFRIT_APIDECL SingleBuffer *
ResourceManager::createIndexBufferDevice(uint32_t size, VkFlags extraFlags) {
  BufferCreateInfo ci{};
  ci.size = size;
  ci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | extraFlags;
  ci.hostVisible = false;
  return createSimpleBuffer(ci);
}

IFRIT_APIDECL
MultiBuffer *ResourceManager::createProxyMultiBuffer(
    const std::vector<SingleBuffer *> &buffers) {
  auto tmp = new MultiBuffer(m_context, buffers);
  auto buffer = std::make_unique<MultiBuffer>(m_context, buffers);
  auto ptr = buffer.get();
  m_multiBuffer.push_back(std::move(buffer));
  return ptr;
}

IFRIT_APIDECL SingleDeviceImage *
ResourceManager::createDepthAttachment(uint32_t width, uint32_t height,
                                       VkFormat format,
                                       VkImageUsageFlags extraUsage) {
  ImageCreateInfo ci{};
  ci.aspect = ImageAspect::Depth;
  ci.format = format;
  ci.width = width;
  ci.height = height;
  ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
             VK_IMAGE_USAGE_SAMPLED_BIT | extraUsage;
  ci.hostVisible = false;
  return createSimpleImage(ci);
}

IFRIT_APIDECL std::shared_ptr<SingleDeviceImage>
ResourceManager::createRenderTargetTexture(uint32_t width, uint32_t height,
                                           VkFormat format,
                                           VkImageUsageFlags extraUsage) {
  ImageCreateInfo ci{};
  ci.aspect = ImageAspect::Color;
  ci.format = format;
  ci.width = width;
  ci.height = height;
  ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
             extraUsage;
  ci.hostVisible = false;
  return createSimpleImageUnmanaged(ci);
}

std::shared_ptr<SingleDeviceImage>
ResourceManager::createRenderTargetMipTexture(uint32_t width, uint32_t height,
                                              uint32_t mips, VkFormat format,
                                              VkImageUsageFlags extraUsage) {
  ImageCreateInfo ci{};
  ci.aspect = ImageAspect::Color;
  ci.format = format;
  ci.width = width;
  ci.height = height;
  ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
             extraUsage;
  ci.hostVisible = false;
  ci.mipLevels = mips;
  return createSimpleImageUnmanaged(ci);
}

IFRIT_APIDECL SingleDeviceImage *ResourceManager::createTexture2DDevice(
    uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage) {
  ImageCreateInfo ci{};
  ci.aspect = ImageAspect::Color;
  ci.format = format;
  ci.width = width;
  ci.height = height;
  ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | usage;
  ci.hostVisible = false;
  return createSimpleImage(ci);
}

IFRIT_APIDECL std::shared_ptr<Sampler>
ResourceManager::createTrivialRenderTargetSampler() {
  SamplerCreateInfo ci{};
  ci.magFilter = VK_FILTER_LINEAR;
  ci.minFilter = VK_FILTER_LINEAR;
  ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ci.anisotropyEnable = false;
  ci.maxAnisotropy = 1.0f;
  ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
  ci.unnormalizedCoordinates = VK_FALSE;
  ci.compareEnable = VK_FALSE;
  ci.compareOp = VK_COMPARE_OP_ALWAYS;
  ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  ci.mipLodBias = 0.0f;
  ci.minLod = 0.0f;
  ci.maxLod = 0.0f;
  auto sampler = std::make_shared<Sampler>(m_context, ci);
  return sampler;
}

} // namespace Ifrit::GraphicsBackend::VulkanGraphics