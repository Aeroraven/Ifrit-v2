#pragma once
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"

namespace Ifrit::GraphicsBackend::VulkanGraphics {
class IFRIT_APIDECL StagedSingleBuffer : public Rhi::RhiStagedSingleBuffer {
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
  void cmdCopyToDevice(const Rhi::RhiCommandBuffer *cmd, const void *data,
                       uint32_t size, uint32_t localOffset) override;
};

class IFRIT_APIDECL StagedSingleImage {
protected:
  std::unique_ptr<SingleDeviceImage> m_imageUnique;
  SingleDeviceImage *m_image;
  std::unique_ptr<SingleBuffer> m_stagingBuffer;
  EngineContext *m_context;

public:
  StagedSingleImage(EngineContext *ctx, SingleDeviceImage *image);
  StagedSingleImage(EngineContext *ctx, const ImageCreateInfo &ci);

  StagedSingleImage(const StagedSingleImage &p) = delete;
  StagedSingleImage &operator=(const StagedSingleImage &p) = delete;

  virtual ~StagedSingleImage() {}
  inline SingleDeviceImage *getImage() const { return m_image; }
  inline SingleBuffer *getStagingBuffer() const {
    return m_stagingBuffer.get();
  }
  void cmdCopyToDevice(CommandBuffer *cmd, const void *data,
                       VkImageLayout srcLayout, VkImageLayout dstlayout,
                       VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics