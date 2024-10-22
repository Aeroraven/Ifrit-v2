#pragma once
#include <memory>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>

namespace Ifrit::Engine::VkRenderer {
enum class BufferMemoryType { DeviceLocal, HostLocal };

struct BufferCreateInfo {
  uint32_t size;
  VkFlags usage;
  bool hostVisible = false;
};

class IFRIT_APIDECL SingleBuffer {
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
  virtual void map(size_t size = VK_WHOLE_SIZE);
  virtual void unmap();
  virtual void copyFromBuffer(void *data, uint32_t size, uint32_t offset);
  virtual void copyToBuffer(const void *data, uint32_t size, uint32_t offset);
  virtual void flush();
  inline VkBuffer getBuffer() const { return m_buffer; }
  inline VkFlags getUsage() const { return m_createInfo.usage; }
  inline uint32_t getSize() const { return m_createInfo.size; }
};

class IFRIT_APIDECL MultiBuffer {
protected:
  std::vector<std::unique_ptr<SingleBuffer>> m_buffers;
  BufferCreateInfo m_createInfo;
  EngineContext *m_context;
  uint32_t m_activeFrame = 0;

public:
  MultiBuffer(EngineContext *ctx, const BufferCreateInfo &ci,
              uint32_t numCopies);
  MultiBuffer(const MultiBuffer &p) = delete;
  MultiBuffer &operator=(const MultiBuffer &p) = delete;
  SingleBuffer *getBuffer(uint32_t index);
  inline SingleBuffer* getActiveBuffer() {
    return m_buffers[m_activeFrame].get();
  }
  inline void advanceFrame() {
    m_activeFrame++;
    m_activeFrame %= m_buffers.size();
  }
  inline uint32_t getActiveFrame() { return m_activeFrame; }
  inline void setActiveFrame(uint32_t frame) { m_activeFrame = frame; }
  inline uint32_t getBufferCount() { return m_buffers.size(); }
};

class IFRIT_APIDECL DeviceImage {
protected:
  VkFormat m_format;
  VkImage m_image;
  VkImageView m_imageView;
  bool m_isSwapchainImage = false;

public:
  DeviceImage() {}
  virtual VkFormat getFormat();
  virtual VkImage getImage();
  virtual VkImageView getImageView();
  inline bool getIsSwapchainImage() { return m_isSwapchainImage; }
};

class IFRIT_APIDECL ResourceManager {
protected:
  EngineContext *m_context;
  std::vector<std::unique_ptr<SingleBuffer>> m_simpleBuffer;
  std::vector<std::unique_ptr<MultiBuffer>> m_multiBuffer;
  std::vector<uint32_t> m_multiBufferTraced;
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

  void setActiveFrame(uint32_t frame);
  inline void setDefaultCopies(int32_t copies) { m_defaultCopies = copies; }
};

} // namespace Ifrit::Engine::VkRenderer