#include <vkrenderer/include/engine/vkrenderer/MemoryResource.h>
#include <vkrenderer/include/utility/Logger.h>

namespace Ifrit::Engine::VkRenderer {
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
IFRIT_APIDECL void SingleBuffer::map(size_t size) {
  vkrAssert(m_createInfo.hostVisible, "Buffer must be host visible to map");
  vmaMapMemory(m_context->getAllocator(), m_allocation,
               (void **)&m_mappedMemory);
}
IFRIT_APIDECL void SingleBuffer::unmap() {
  vkrAssert(m_mappedMemory, "Buffer must be mapped to unmap");
  vmaUnmapMemory(m_context->getAllocator(), m_allocation);
  m_mappedMemory = nullptr;
}
IFRIT_APIDECL void SingleBuffer::copyFromBuffer(void *data, uint32_t size,
                                                uint32_t offset) {
  vkrAssert(m_mappedMemory, "Buffer must be mapped to copy data");
  memcpy(m_mappedMemory + offset, data, size);
}
IFRIT_APIDECL void SingleBuffer::copyToBuffer(const void *data, uint32_t size,
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
  for (int i = 0; i < numCopies; i++) {
    m_buffers.push_back(std::make_unique<SingleBuffer>(ctx, ci));
  }
}

IFRIT_APIDECL SingleBuffer *MultiBuffer::getBuffer(uint32_t index) {
  return m_buffers[index].get();
}

// Class: Image
IFRIT_APIDECL VkFormat SingleDeviceImage::getFormat() { return m_format; }
IFRIT_APIDECL VkImage SingleDeviceImage::getImage() { return m_image; }
IFRIT_APIDECL VkImageView SingleDeviceImage::getImageView() {
  return m_imageView;
}

IFRIT_APIDECL SingleDeviceImage::~SingleDeviceImage() {
  if (m_created) {
    vkDestroyImageView(m_context->getDevice(), m_imageView, nullptr);
    vmaDestroyImage(m_context->getAllocator(), m_image, m_allocation);
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

  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = VMA_MEMORY_USAGE_AUTO;
  if (ci.hostVisible) {
    allocCI.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  vkrVulkanAssert(vmaCreateImage(ctx->getAllocator(), &imageCI, &allocCI,
                                 &m_image, &m_allocation, &m_allocInfo),
                  "Failed to create image");

  // create image view
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
  imageViewCI.subresourceRange.baseMipLevel = 0;
  imageViewCI.subresourceRange.levelCount = ci.mipLevels;
  imageViewCI.subresourceRange.baseArrayLayer = 0;
  imageViewCI.subresourceRange.layerCount = ci.arrayLayers;

  vkrVulkanAssert(
      vkCreateImageView(ctx->getDevice(), &imageViewCI, nullptr, &m_imageView),
      "Failed to create image view");
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
  m_multiBufferTraced.push_back(m_multiBuffer.size() - 1);
  return ptr;
}

IFRIT_APIDECL void ResourceManager::setActiveFrame(uint32_t frame) {
  for (int i = 0; i < m_multiBufferTraced.size(); i++) {
    m_multiBuffer[m_multiBufferTraced[i]]->setActiveFrame(frame);
  }
}

} // namespace Ifrit::Engine::VkRenderer