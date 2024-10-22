#include <vkrenderer/include/engine/vkrenderer/StagedMemoryResource.h>

namespace Ifrit::Engine::VkRenderer {

IFRIT_APIDECL StagedSingleBuffer::StagedSingleBuffer(EngineContext *ctx,
                                                     SingleBuffer *buffer) {
  m_context = ctx;
  m_buffer = buffer;
  BufferCreateInfo stagingCI{};
  stagingCI.size = buffer->getSize();
  stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  stagingCI.hostVisible = true;
  m_stagingBuffer = std::make_unique<SingleBuffer>(ctx, stagingCI);
}
IFRIT_APIDECL StagedSingleBuffer::StagedSingleBuffer(EngineContext *ctx,
                                                     const BufferCreateInfo &ci)
    : m_context(ctx) {
  BufferCreateInfo ci2 = ci;
  ci2.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  m_bufferUnique = std::make_unique<SingleBuffer>(ctx, ci2);
  m_buffer = m_bufferUnique.get();

  BufferCreateInfo stagingCI = ci;
  stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  stagingCI.hostVisible = true;
  m_stagingBuffer = std::make_unique<SingleBuffer>(ctx, stagingCI);
}

IFRIT_APIDECL void StagedSingleBuffer::cmdCopyToDevice(CommandBuffer *cmd,
                                                       const void *data,
                                                       uint32_t size,
                                                       uint32_t localOffset) {
  m_stagingBuffer->map();
  m_stagingBuffer->copyFromBuffer((void *)data, size, 0);
  m_stagingBuffer->flush();
  m_stagingBuffer->unmap();
  cmd->copyBuffer(m_stagingBuffer.get()->getBuffer(), m_buffer->getBuffer(),
                  size, 0, localOffset);
}

} // namespace Ifrit::Engine::VkRenderer