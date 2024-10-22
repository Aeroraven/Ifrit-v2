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
IFRIT_APIDECL VkFormat DeviceImage::getFormat() { return m_format; }
IFRIT_APIDECL VkImage DeviceImage::getImage() { return m_image; }
IFRIT_APIDECL VkImageView DeviceImage::getImageView() { return m_imageView; }

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