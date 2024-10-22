#pragma once
#include <vkrenderer/include/engine/vkrenderer/Command.h>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vkrenderer/include/engine/vkrenderer/MemoryResource.h>

namespace Ifrit::Engine::VkRenderer {
class IFRIT_APIDECL StagedSingleBuffer {
protected:
  std::unique_ptr<SingleBuffer> m_bufferUnique;
  SingleBuffer *m_buffer;
  std::unique_ptr<SingleBuffer> m_stagingBuffer;
  EngineContext *m_context;

public:
  StagedSingleBuffer(EngineContext *ctx, SingleBuffer *buffer);
  StagedSingleBuffer(EngineContext *ctx, const BufferCreateInfo &ci);
  StagedSingleBuffer(const StagedSingleBuffer &p) = delete;
  StagedSingleBuffer &operator=(const StagedSingleBuffer &p) = delete;

  virtual ~StagedSingleBuffer() {}
  inline SingleBuffer *getBuffer() const { return m_buffer; }
  inline SingleBuffer *getStagingBuffer() const {
    return m_stagingBuffer.get();
  }
  void cmdCopyToDevice(CommandBuffer *cmd, const void *data, uint32_t size,
                       uint32_t localOffset);
};
} // namespace Ifrit::Engine::VkRenderer