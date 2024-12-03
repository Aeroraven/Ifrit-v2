
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


#include "ifrit/vkgraphics/engine/vkrenderer/StagedMemoryResource.h"

namespace Ifrit::GraphicsBackend::VulkanGraphics {

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

IFRIT_APIDECL void
StagedSingleBuffer::cmdCopyToDevice(const Rhi::RhiCommandBuffer *cmd,
                                    const void *data, uint32_t size,
                                    uint32_t localOffset) {
  m_stagingBuffer->map();
  m_stagingBuffer->writeBuffer((void *)data, size, 0);
  m_stagingBuffer->flush();
  m_stagingBuffer->unmap();
  cmd->copyBuffer(m_stagingBuffer.get(), m_buffer, size, 0, localOffset);
}

IFRIT_APIDECL StagedSingleImage::StagedSingleImage(EngineContext *ctx,
                                                   SingleDeviceImage *image)
    : m_context(ctx), m_image(image) {
  BufferCreateInfo stagingCI{};
  stagingCI.size = image->getSize();
  stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  stagingCI.hostVisible = true;
  m_stagingBuffer = std::make_unique<SingleBuffer>(ctx, stagingCI);
}

IFRIT_APIDECL StagedSingleImage::StagedSingleImage(EngineContext *ctx,
                                                   const ImageCreateInfo &ci)
    : m_context(ctx) {
  ImageCreateInfo ci2 = ci;
  ci2.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  m_imageUnique = std::make_unique<SingleDeviceImage>(ctx, ci2);
  m_image = m_imageUnique.get();

  BufferCreateInfo stagingCI{};
  stagingCI.size = m_image->getSize();
  stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  stagingCI.hostVisible = true;
  m_stagingBuffer = std::make_unique<SingleBuffer>(ctx, stagingCI);
}

IFRIT_APIDECL void StagedSingleImage::cmdCopyToDevice(
    CommandBuffer *cmd, const void *data, VkImageLayout srcLayout,
    VkImageLayout dstLayout, VkPipelineStageFlags dstStage,
    VkAccessFlags dstAccess) {
  m_stagingBuffer->map();
  m_stagingBuffer->writeBuffer((void *)data, m_image->getSize(), 0);
  m_stagingBuffer->flush();
  m_stagingBuffer->unmap();
  PipelineBarrier barrier(m_context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0);

  VkImageMemoryBarrier imageBarrier{};
  imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imageBarrier.oldLayout = srcLayout;
  imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  imageBarrier.image = m_image->getImage();
  imageBarrier.subresourceRange.aspectMask = m_image->getAspect();
  imageBarrier.subresourceRange.levelCount = 1;
  imageBarrier.subresourceRange.layerCount = 1;
  imageBarrier.srcAccessMask = 0;
  imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.addImageMemoryBarrier(imageBarrier);

  cmd->pipelineBarrier(barrier);
  cmd->copyBufferToImageAll(m_stagingBuffer.get(), m_image->getImage(),
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            m_image->getWidth(), m_image->getHeight(),
                            m_image->getDepth());

  imageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  imageBarrier.newLayout = dstLayout;
  imageBarrier.srcAccessMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
  imageBarrier.dstAccessMask = dstAccess;
  PipelineBarrier barrier2(m_context, VK_ACCESS_TRANSFER_WRITE_BIT, dstStage,
                           0);
  barrier2.addImageMemoryBarrier(imageBarrier);
  cmd->pipelineBarrier(barrier2);
}

} // namespace Ifrit::GraphicsBackend::VulkanGraphics