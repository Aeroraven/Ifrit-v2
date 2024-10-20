#include <vkrenderer/include/engine/vkrenderer/MemoryResource.h>
#include <vkrenderer/include/utility/Logger.h>

namespace Ifrit::Engine::VkRenderer {
IFRIT_APIDECL void Buffer::init() {
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
IFRIT_APIDECL Buffer::~Buffer() {
  vmaDestroyBuffer(m_context->getAllocator(), m_buffer, m_allocation);
}
IFRIT_APIDECL void Buffer::map(size_t size) {
  vkrAssert(m_createInfo.hostVisible, "Buffer must be host visible to map");
  vmaMapMemory(m_context->getAllocator(), m_allocation,
               (void **)&m_mappedMemory);
}
IFRIT_APIDECL void Buffer::unmap() {
  vkrAssert(m_mappedMemory, "Buffer must be mapped to unmap");
  vmaUnmapMemory(m_context->getAllocator(), m_allocation);
  m_mappedMemory = nullptr;
}
IFRIT_APIDECL void Buffer::copyFromBuffer(void *data, uint32_t size,
                                          uint32_t offset) {
  vkrAssert(m_mappedMemory, "Buffer must be mapped to copy data");
  memcpy(m_mappedMemory + offset, data, size);
}
IFRIT_APIDECL void Buffer::copyToBuffer(const void *data, uint32_t size,
                                        uint32_t offset) {
  vkrAssert(m_mappedMemory, "Buffer must be mapped to copy data");
  memcpy((void *)(m_mappedMemory + offset), data, size);
}
IFRIT_APIDECL void Buffer::flush() {
  vmaFlushAllocation(m_context->getAllocator(), m_allocation, 0, VK_WHOLE_SIZE);
}

// Class: Image

IFRIT_APIDECL VkFormat DeviceImage::getFormat() { return m_format; }
IFRIT_APIDECL VkImage DeviceImage::getImage() { return m_image; }
IFRIT_APIDECL VkImageView DeviceImage::getImageView() { return m_imageView; }
} // namespace Ifrit::Engine::VkRenderer