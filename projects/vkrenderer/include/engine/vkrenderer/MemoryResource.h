#pragma once
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
namespace Ifrit::Engine::VkRenderer {
enum class BufferMemoryType { DeviceLocal, HostLocal };

struct BufferCreateInfo {
  uint32_t size;
  VkFlags usage;
  bool hostVisible = false;
};

class IFRIT_APIDECL Buffer {
private:
  VkBuffer m_buffer;
  EngineContext *m_context;
  VmaAllocation m_allocation;
  VmaAllocationInfo m_allocInfo;
  BufferCreateInfo m_createInfo;
  char *m_mappedMemory = nullptr;

protected:
  void init();

public:
  Buffer(EngineContext *ctx, const BufferCreateInfo &ci)
      : m_context(ctx), m_createInfo(ci) {
    init();
  }
  virtual ~Buffer();
  virtual void map(size_t size = VK_WHOLE_SIZE);
  virtual void unmap();
  virtual void copyFromBuffer(void *data, uint32_t size, uint32_t offset);
  virtual void copyToBuffer(const void *data, uint32_t size, uint32_t offset);
  virtual void flush();
  inline VkBuffer getBuffer() const { return m_buffer; }
};

class IFRIT_APIDECL DeviceImage {
protected:
  VkFormat m_format;
  VkImage m_image;
  VkImageView m_imageView;

public:
  DeviceImage() {}
  virtual VkFormat getFormat();
  virtual VkImage getImage();
  virtual VkImageView getImageView();
};
} // namespace Ifrit::Engine::VkRenderer