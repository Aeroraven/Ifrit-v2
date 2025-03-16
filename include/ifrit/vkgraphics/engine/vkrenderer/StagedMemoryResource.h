
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
  inline SingleBuffer *getStagingBuffer() const { return m_stagingBuffer.get(); }
  void cmdCopyToDevice(const Rhi::RhiCommandBuffer *cmd, const void *data, u32 size, u32 localOffset) override;
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
  inline SingleBuffer *getStagingBuffer() const { return m_stagingBuffer.get(); }
  void cmdCopyToDevice(CommandBuffer *cmd, const void *data, VkImageLayout srcLayout, VkImageLayout dstlayout,
                       VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics