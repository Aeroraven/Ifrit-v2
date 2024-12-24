
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

#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Binding.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderGraph.h"
#include "ifrit/vkgraphics/utility/Logger.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::GraphicsBackend::VulkanGraphics {
IFRIT_APIDECL TimelineSemaphore::TimelineSemaphore(EngineContext *ctx)
    : m_context(ctx) {
  VkSemaphoreTypeCreateInfo timelineCI{};
  timelineCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timelineCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timelineCI.initialValue = 0;
  VkSemaphoreCreateInfo semaphoreCI{};
  semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphoreCI.pNext = &timelineCI;
  vkrVulkanAssert(vkCreateSemaphore(m_context->getDevice(), &semaphoreCI,
                                    nullptr, &m_semaphore),
                  "Failed to create timeline semaphore");
}

IFRIT_APIDECL TimelineSemaphore::~TimelineSemaphore() {
  vkDestroySemaphore(m_context->getDevice(), m_semaphore, nullptr);
}

IFRIT_APIDECL void CommandPool::init() {
  VkCommandPoolCreateInfo poolCI{};
  poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolCI.queueFamilyIndex = m_queueFamily;
  poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkrVulkanAssert(vkCreateCommandPool(m_context->getDevice(), &poolCI, nullptr,
                                      &m_commandPool),
                  "Failed to create command pool");
}

IFRIT_APIDECL CommandPool::~CommandPool() {
  vkDestroyCommandPool(m_context->getDevice(), m_commandPool, nullptr);
}

IFRIT_APIDECL std::shared_ptr<CommandBuffer>
CommandPool::allocateCommandBuffer() {
  VkCommandBufferAllocateInfo bufferAI{};
  bufferAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  bufferAI.commandPool = m_commandPool;
  bufferAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  bufferAI.commandBufferCount = 1;

  VkCommandBuffer buffer;
  vkrVulkanAssert(
      vkAllocateCommandBuffers(m_context->getDevice(), &bufferAI, &buffer),
      "Failed to allocate command buffer");
  return std::make_shared<CommandBuffer>(m_context, buffer, m_queueFamily);
}

IFRIT_APIDECL std::unique_ptr<CommandBuffer>
CommandPool::allocateCommandBufferUnique() {
  VkCommandBufferAllocateInfo bufferAI{};
  bufferAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  bufferAI.commandPool = m_commandPool;
  bufferAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  bufferAI.commandBufferCount = 1;

  VkCommandBuffer buffer;
  vkrVulkanAssert(
      vkAllocateCommandBuffers(m_context->getDevice(), &bufferAI, &buffer),
      "Failed to allocate command buffer");
  return std::make_unique<CommandBuffer>(m_context, buffer, m_queueFamily);
}

// Class: Pipeline Barrier
IFRIT_APIDECL void PipelineBarrier::addMemoryBarrier(VkMemoryBarrier barrier) {
  m_memoryBarriers.push_back(barrier);
}

IFRIT_APIDECL void
PipelineBarrier::addBufferMemoryBarrier(VkBufferMemoryBarrier barrier) {
  m_bufferMemoryBarriers.push_back(barrier);
}

IFRIT_APIDECL void
PipelineBarrier::addImageMemoryBarrier(VkImageMemoryBarrier barrier) {
  m_imageMemoryBarriers.push_back(barrier);
}

// Class: CommandBuffer
IFRIT_APIDECL void CommandBuffer::beginRecord() {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkrVulkanAssert(vkBeginCommandBuffer(m_commandBuffer, &beginInfo),
                  "Failed to begin command buffer");
}

IFRIT_APIDECL void CommandBuffer::endRecord() {
  vkrVulkanAssert(vkEndCommandBuffer(m_commandBuffer),
                  "Failed to end command buffer");
}

IFRIT_APIDECL void CommandBuffer::reset() {
  vkrVulkanAssert(
      vkResetCommandBuffer(m_commandBuffer,
                           VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT),
      "Failed to reset command buffer");
}

IFRIT_APIDECL void
CommandBuffer::pipelineBarrier(const PipelineBarrier &barrier) const {
  vkCmdPipelineBarrier(m_commandBuffer, barrier.m_srcStage, barrier.m_dstStage,
                       barrier.m_dependencyFlags,
                       size_cast<int>(barrier.m_memoryBarriers.size()),
                       barrier.m_memoryBarriers.data(),
                       size_cast<int>(barrier.m_bufferMemoryBarriers.size()),
                       barrier.m_bufferMemoryBarriers.data(),
                       size_cast<int>(barrier.m_imageMemoryBarriers.size()),
                       barrier.m_imageMemoryBarriers.data());
}

IFRIT_APIDECL void CommandBuffer::setViewports(
    const std::vector<Rhi::RhiViewport> &viewport) const {
  std::vector<VkViewport> vps;
  for (int i = 0; i < viewport.size(); i++) {
    VkViewport s = {viewport[i].x,        viewport[i].y,
                    viewport[i].width,    viewport[i].height,
                    viewport[i].minDepth, viewport[i].maxDepth};
    vps.push_back(s);
  }
  vkCmdSetViewport(m_commandBuffer, 0, size_cast<int>(vps.size()), vps.data());
}

IFRIT_APIDECL void
CommandBuffer::setScissors(const std::vector<Rhi::RhiScissor> &scissor) const {
  std::vector<VkRect2D> scs;
  for (int i = 0; i < scissor.size(); i++) {
    VkRect2D s = {{scissor[i].x, scissor[i].y},
                  {scissor[i].width, scissor[i].height}};
    scs.push_back(s);
  }
  vkCmdSetScissor(m_commandBuffer, 0, size_cast<int>(scs.size()), scs.data());
}

IFRIT_APIDECL void CommandBuffer::draw(uint32_t vertexCount,
                                       uint32_t instanceCount,
                                       uint32_t firstVertex,
                                       uint32_t firstInstance) const {
  vkCmdDraw(m_commandBuffer, vertexCount, instanceCount, firstVertex,
            firstInstance);
}

IFRIT_APIDECL void CommandBuffer::drawIndexed(uint32_t indexCount,
                                              uint32_t instanceCount,
                                              uint32_t firstIndex,
                                              int32_t vertexOffset,
                                              uint32_t firstInstance) const {
  vkCmdDrawIndexed(m_commandBuffer, indexCount, instanceCount, firstIndex,
                   vertexOffset, firstInstance);
}

IFRIT_APIDECL void CommandBuffer::dispatch(uint32_t groupCountX,
                                           uint32_t groupCountY,
                                           uint32_t groupCountZ) const {
  vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, groupCountZ);
}

IFRIT_APIDECL void CommandBuffer::drawMeshTasks(uint32_t groupCountX,
                                                uint32_t groupCountY,
                                                uint32_t groupCountZ) const {
  m_context->getExtensionFunction().p_vkCmdDrawMeshTasksEXT(
      m_commandBuffer, groupCountX, groupCountY, groupCountZ);
}

IFRIT_APIDECL void
CommandBuffer::drawMeshTasksIndirect(const Rhi::RhiBuffer *buffer,
                                     uint32_t offset, uint32_t drawCount,
                                     uint32_t stride) const {
  auto buf = checked_cast<SingleBuffer>(buffer)->getBuffer();
  m_context->getExtensionFunction().p_vkCmdDrawMeshTasksIndirectEXT(
      m_commandBuffer, buf, offset, drawCount, stride);
}

IFRIT_APIDECL void CommandBuffer::copyBuffer(const Rhi::RhiBuffer *srcBuffer,
                                             const Rhi::RhiBuffer *dstBuffer,
                                             uint32_t size, uint32_t srcOffset,
                                             uint32_t dstOffset) const {
  VkBufferCopy copyRegion{};
  copyRegion.srcOffset = srcOffset;
  copyRegion.dstOffset = dstOffset;
  copyRegion.size = size;
  auto src = checked_cast<SingleBuffer>(srcBuffer)->getBuffer();
  auto dst = checked_cast<SingleBuffer>(dstBuffer)->getBuffer();
  vkCmdCopyBuffer(m_commandBuffer, src, dst, 1, &copyRegion);
}

IFRIT_APIDECL void CommandBuffer::copyBufferToImageAllInternal(
    const Rhi::RhiBuffer *srcBuffer, VkImage dstImage, VkImageLayout dstLayout,
    uint32_t width, uint32_t height, uint32_t depth) const {
  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width, height, depth};

  auto src = checked_cast<SingleBuffer>(srcBuffer)->getBuffer();
  vkCmdCopyBufferToImage(m_commandBuffer, src, dstImage, dstLayout, 1, &region);
}

IFRIT_APIDECL void
CommandBuffer::copyBufferToImage(const Rhi::RhiBuffer *src,
                                 const Rhi::RhiTexture *dst,
                                 Rhi::RhiImageSubResource dstSub) const {
  auto image = checked_cast<SingleDeviceImage>(dst);
  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = dstSub.mipLevel;
  region.imageSubresource.baseArrayLayer = dstSub.arrayLayer;
  region.imageSubresource.layerCount = dstSub.layerCount;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {image->getWidth(), image->getHeight(), 1};

  auto buffer = checked_cast<SingleBuffer>(src)->getBuffer();
  vkCmdCopyBufferToImage(m_commandBuffer, buffer, image->getImage(),
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

IFRIT_APIDECL void CommandBuffer::globalMemoryBarrier() const {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);
}

// Rhi compatible
IFRIT_APIDECL void CommandBuffer::imageBarrier(
    const Rhi::RhiTexture *texture, Rhi::RhiResourceState src,
    Rhi::RhiResourceState dst, Rhi::RhiImageSubResource subResource) const {
  auto image = checked_cast<SingleDeviceImage>(texture);
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  // select old layout
  switch (src) {
  case Rhi::RhiResourceState::Undefined:
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.srcAccessMask =
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_HOST_READ_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    break;
  case Rhi::RhiResourceState::Common:
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    break;

  case Rhi::RhiResourceState::RenderTarget:
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    break;
  case Rhi::RhiResourceState::Present:
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    break;
  case Rhi::RhiResourceState::DepthStencilRenderTarget:
    barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    break;
  case Rhi::RhiResourceState::UAVStorageImage:
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    break;

  case Rhi::RhiResourceState::CopySource:
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    break;

  case Rhi::RhiResourceState::CopyDest:
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    break;

  default:
    vkrError("Invalid resource state");
  }
  // select new layout
  switch (dst) {
  case Rhi::RhiResourceState::Undefined:
    barrier.newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.dstAccessMask =
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_HOST_READ_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    break;
  case Rhi::RhiResourceState::Common:
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    break;
  case Rhi::RhiResourceState::RenderTarget:
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    break;
  case Rhi::RhiResourceState::DepthStencilRenderTarget:
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    break;
  case Rhi::RhiResourceState::Present:
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    break;
  case Rhi::RhiResourceState::UAVStorageImage:
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    break;

  case Rhi::RhiResourceState::CopySource:
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    break;

  case Rhi::RhiResourceState::CopyDest:
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    break;
  default:
    vkrError("Invalid resource state");
  }
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  auto p = image->getImage();
  barrier.image = p;
  barrier.subresourceRange.aspectMask = image->getAspect();
  barrier.subresourceRange.layerCount = subResource.layerCount;
  barrier.subresourceRange.levelCount = subResource.mipCount;
  barrier.subresourceRange.baseArrayLayer = subResource.arrayLayer;
  barrier.subresourceRange.baseMipLevel = subResource.mipLevel;

  vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

IFRIT_APIDECL void CommandBuffer::attachBindlessReferenceGraphics(
    Rhi::RhiGraphicsPass *pass, uint32_t setId,
    Rhi::RhiBindlessDescriptorRef *ref) const {

  auto bindless = checked_cast<DescriptorBindlessIndices>(ref);
  auto graphicsPass = checked_cast<GraphicsPass>(pass);
  auto set = bindless->getActiveRangeSet();
  auto offset = bindless->getActiveRangeOffset();

  vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          graphicsPass->getPipelineLayout(), setId, 1, &set, 1,
                          &offset);
}

IFRIT_APIDECL void CommandBuffer::attachBindlessReferenceCompute(
    Rhi::RhiComputePass *pass, uint32_t setId,
    Rhi::RhiBindlessDescriptorRef *ref) const {
  auto bindless = checked_cast<DescriptorBindlessIndices>(ref);
  auto computePass = checked_cast<ComputePass>(pass);
  auto set = bindless->getActiveRangeSet();
  auto offset = bindless->getActiveRangeOffset();

  vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computePass->getPipelineLayout(), setId, 1, &set, 1,
                          &offset);
}

IFRIT_APIDECL void CommandBuffer::attachVertexBufferView(
    const Rhi::RhiVertexBufferView &view) const {
  auto vxDesc = checked_cast<VertexBufferDescriptor>(&view);
  auto exfun = m_context->getExtensionFunction();
  exfun.p_vkCmdSetVertexInputEXT(
      m_commandBuffer, size_cast<uint32_t>(vxDesc->m_bindings.size()),
      vxDesc->m_bindings.data(),
      size_cast<uint32_t>(vxDesc->m_attributes.size()),
      vxDesc->m_attributes.data());
}

IFRIT_APIDECL void CommandBuffer::attachVertexBuffers(
    uint32_t firstSlot, const std::vector<Rhi::RhiBuffer *> &buffers) const {
  std::vector<VkBuffer> vxbuffers;
  std::vector<VkDeviceSize> offsets;
  for (int i = 0; i < buffers.size(); i++) {
    auto buffer = checked_cast<SingleBuffer>(buffers[i]);
    vxbuffers.push_back(buffer->getBuffer());
    offsets.push_back(0);
  }
  vkCmdBindVertexBuffers(m_commandBuffer, firstSlot,
                         size_cast<uint32_t>(buffers.size()), vxbuffers.data(),
                         offsets.data());
}

IFRIT_APIDECL void CommandBuffer::drawInstanced(uint32_t vertexCount,
                                                uint32_t instanceCount,
                                                uint32_t firstVertex,
                                                uint32_t firstInstance) const {
  vkCmdDraw(m_commandBuffer, vertexCount, instanceCount, firstVertex,
            firstInstance);
}

IFRIT_APIDECL void
CommandBuffer::uavBufferBarrier(const Rhi::RhiBuffer *buffer) const {
  auto buf = checked_cast<SingleBuffer>(buffer)->getBuffer();
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.buffer = buf;
  barrier.size = VK_WHOLE_SIZE;
  barrier.srcAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1,
                       &barrier, 0, nullptr);
}

IFRIT_APIDECL void CommandBuffer::uavBufferClear(const Rhi::RhiBuffer *buffer,
                                                 uint32_t val) const {
  auto buf = checked_cast<SingleBuffer>(buffer)->getBuffer();
  vkCmdFillBuffer(m_commandBuffer, buf, 0, VK_WHOLE_SIZE, val);
}

IFRIT_APIDECL void CommandBuffer::dispatchIndirect(const Rhi::RhiBuffer *buffer,
                                                   uint32_t offset) const {
  auto buf = checked_cast<SingleBuffer>(buffer)->getBuffer();
  vkCmdDispatchIndirect(m_commandBuffer, buf, offset);
}

IFRIT_APIDECL void CommandBuffer::setPushConst(Rhi::RhiComputePass *pass,
                                               uint32_t offset, uint32_t size,
                                               const void *data) const {
  auto computePass = checked_cast<ComputePass>(pass);
  vkCmdPushConstants(m_commandBuffer, computePass->getPipelineLayout(),
                     VK_SHADER_STAGE_ALL, offset, size, data);
};
IFRIT_APIDECL void CommandBuffer::setPushConst(Rhi::RhiGraphicsPass *pass,
                                               uint32_t offset, uint32_t size,
                                               const void *data) const {
  auto graphicsPass = checked_cast<GraphicsPass>(pass);
  vkCmdPushConstants(m_commandBuffer, graphicsPass->getPipelineLayout(),
                     VK_SHADER_STAGE_ALL, offset, size, data);
};

IFRIT_APIDECL void
CommandBuffer::clearUAVImageFloat(const Rhi::RhiTexture *texture,
                                  Rhi::RhiImageSubResource subResource,
                                  const std::array<float, 4> &val) const {
  auto image = checked_cast<SingleDeviceImage>(texture);
  VkClearColorValue clearColor;
  clearColor.float32[0] = val[0];
  clearColor.float32[1] = val[1];
  clearColor.float32[2] = val[2];
  clearColor.float32[3] = val[3];
  VkImageSubresourceRange range{};
  range.aspectMask = image->getAspect();
  range.baseMipLevel = subResource.mipLevel;
  range.levelCount = subResource.mipCount;
  range.baseArrayLayer = subResource.arrayLayer;
  range.layerCount = subResource.layerCount;
  vkCmdClearColorImage(m_commandBuffer, image->getImage(),
                       VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
}

IFRIT_APIDECL void CommandBuffer::copyImage(
    const Rhi::RhiTexture *src, Rhi::RhiImageSubResource srcSub,
    const Rhi::RhiTexture *dst, Rhi::RhiImageSubResource dstSub) const {
  auto srcImage = checked_cast<SingleDeviceImage>(src);
  auto dstImage = checked_cast<SingleDeviceImage>(dst);
  VkImageCopy region{};
  region.srcSubresource.aspectMask = srcImage->getAspect();
  region.srcSubresource.mipLevel = srcSub.mipLevel;
  region.srcSubresource.baseArrayLayer = srcSub.arrayLayer;
  region.srcSubresource.layerCount = srcSub.layerCount;
  region.srcOffset = {0, 0, 0};
  region.dstSubresource.aspectMask = dstImage->getAspect();
  region.dstSubresource.mipLevel = dstSub.mipLevel;
  region.dstSubresource.baseArrayLayer = dstSub.arrayLayer;
  region.dstSubresource.layerCount = dstSub.layerCount;
  region.dstOffset = {0, 0, 0};
  region.extent.width = srcImage->getWidth();
  region.extent.height = srcImage->getHeight();
  region.extent.depth = srcImage->getDepth();
  vkCmdCopyImage(m_commandBuffer, srcImage->getImage(),
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage->getImage(),
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
};

void _resourceStateToAccessMask(Rhi::RhiResourceState state,
                                VkAccessFlags &dstAccessMask) {
  switch (state) {
  case Rhi::RhiResourceState::Undefined:
    dstAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_HOST_READ_BIT |
                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    break;
  case Rhi::RhiResourceState::Common:
    dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    break;
  case Rhi::RhiResourceState::RenderTarget:
    dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    break;
  case Rhi::RhiResourceState::Present:
    dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    break;
  case Rhi::RhiResourceState::DepthStencilRenderTarget:
    dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    break;
  case Rhi::RhiResourceState::UAVStorageImage:
    dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    break;
  case Rhi::RhiResourceState::CopySource:
    dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    break;
  case Rhi::RhiResourceState::CopyDest:
    dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    break;
  case Rhi::RhiResourceState::UnorderedAccess:
    dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    break;
  case Rhi::RhiResourceState::PixelShaderResource:
    dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    break;
  default:
    vkrError("Invalid resource state");
  }
}

void _resourceStateToImageLayout(Rhi::RhiResourceState state,
                                 VkImageLayout &dstLayout) {
  switch (state) {
  case Rhi::RhiResourceState::Undefined:
    dstLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    break;
  case Rhi::RhiResourceState::Common:
    dstLayout = VK_IMAGE_LAYOUT_GENERAL;
    break;
  case Rhi::RhiResourceState::RenderTarget:
    dstLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    break;
  case Rhi::RhiResourceState::Present:
    dstLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    break;
  case Rhi::RhiResourceState::DepthStencilRenderTarget:
    dstLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    break;
  case Rhi::RhiResourceState::UAVStorageImage:
    dstLayout = VK_IMAGE_LAYOUT_GENERAL;
    break;
  case Rhi::RhiResourceState::CopySource:
    dstLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    break;
  case Rhi::RhiResourceState::CopyDest:
    dstLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    break;
  case Rhi::RhiResourceState::PixelShaderResource:
    dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    break;
  default:
    vkrError("Invalid resource state");
  }
}

IFRIT_APIDECL
void CommandBuffer::resourceBarrier(
    const std::vector<Rhi::RhiResourceBarrier> &barriers) const {
  std::vector<VkImageMemoryBarrier> imageBarriers;
  std::vector<VkBufferMemoryBarrier> bufferBarriers;
  for (int i = 0; i < barriers.size(); i++) {
    auto barrier = barriers[i];
    if (barrier.m_type == Rhi::RhiBarrierType::Transition) {
      auto resourceType = barrier.m_transition.m_type;
      auto srcState = barrier.m_transition.m_srcState;
      auto dstState = barrier.m_transition.m_dstState;
      VkAccessFlags srcAccessMask, dstAccessMask;
      _resourceStateToAccessMask(srcState, srcAccessMask);
      _resourceStateToAccessMask(dstState, dstAccessMask);

      if (resourceType == Rhi::RhiResourceType::Buffer) {
        VkBufferMemoryBarrier bufferBarrier{};
        bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bufferBarrier.buffer =
            checked_cast<SingleBuffer>(barrier.m_transition.m_buffer)
                ->getBuffer();
        bufferBarrier.offset = 0;
        bufferBarrier.size = VK_WHOLE_SIZE;
        bufferBarrier.srcAccessMask = srcAccessMask;
        bufferBarrier.dstAccessMask = dstAccessMask;
        bufferBarriers.push_back(bufferBarrier);
      } else if (resourceType == Rhi::RhiResourceType::Texture) {
        // WARNING: subresource unspecified
        VkImageLayout srcLayout, dstLayout;
        auto subResource = barrier.m_transition.m_subResource;
        _resourceStateToImageLayout(srcState, srcLayout);
        _resourceStateToImageLayout(dstState, dstLayout);
        auto image =
            checked_cast<SingleDeviceImage>(barrier.m_transition.m_texture);
        auto aspect = image->getAspect();
        VkImageMemoryBarrier imageBarrier{};
        imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageBarrier.image =
            checked_cast<SingleDeviceImage>(barrier.m_transition.m_texture)
                ->getImage();
        imageBarrier.subresourceRange.aspectMask = aspect;
        imageBarrier.subresourceRange.baseMipLevel = subResource.mipLevel;
        imageBarrier.subresourceRange.levelCount = subResource.mipCount;
        imageBarrier.subresourceRange.baseArrayLayer = subResource.arrayLayer;
        imageBarrier.subresourceRange.layerCount = subResource.layerCount;
        imageBarrier.srcAccessMask = srcAccessMask;
        imageBarrier.dstAccessMask = dstAccessMask;
        imageBarrier.oldLayout = srcLayout;
        imageBarrier.newLayout = dstLayout;
        imageBarriers.push_back(imageBarrier);
      }

    } else if (barrier.m_type == Rhi::RhiBarrierType::UAVAccess) {
      auto resourceType = barrier.m_uav.m_type;
      if (resourceType == Rhi::RhiResourceType::Buffer) {
        VkBufferMemoryBarrier bufferBarrier{};
        bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bufferBarrier.buffer =
            checked_cast<SingleBuffer>(barrier.m_uav.m_buffer)->getBuffer();
        bufferBarrier.offset = 0;
        bufferBarrier.size = VK_WHOLE_SIZE;
        bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bufferBarriers.push_back(bufferBarrier);
      } else if (resourceType == Rhi::RhiResourceType::Texture) {
        // WARNING: subresource unspecified
        VkImageMemoryBarrier imageBarrier{};
        imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageBarrier.image =
            checked_cast<SingleDeviceImage>(barrier.m_uav.m_texture)
                ->getImage();
        imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarrier.subresourceRange.baseMipLevel = 0;
        imageBarrier.subresourceRange.levelCount = 1;
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.layerCount = 1;
        imageBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageBarriers.push_back(imageBarrier);
      }
    }
  }
  vkCmdPipelineBarrier(
      m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
      size_cast<int>(bufferBarriers.size()), bufferBarriers.data(),
      size_cast<int>(imageBarriers.size()), imageBarriers.data());
}

IFRIT_APIDECL void CommandBuffer::beginScope(const std::string &name) const {
  VkDebugUtilsLabelEXT label{};
  label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
  label.pLabelName = name.c_str();
  label.color[0] = 1.0f;
  label.color[1] = 1.0f;
  label.color[2] = 0.0f;
  label.color[3] = 1.0f;
  auto exfun = m_context->getExtensionFunction();
  exfun.p_vkCmdBeginDebugUtilsLabelEXT(m_commandBuffer, &label);
}
IFRIT_APIDECL void CommandBuffer::endScope() const {
  auto exfun = m_context->getExtensionFunction();
  exfun.p_vkCmdEndDebugUtilsLabelEXT(m_commandBuffer);
}

IFRIT_APIDECL void CommandBuffer::setCullMode(Rhi::RhiCullMode mode) const {
  VkCullModeFlags cullMode;
  switch (mode) {
  case Rhi::RhiCullMode::None:
    cullMode = VK_CULL_MODE_NONE;
    break;
  case Rhi::RhiCullMode::Front:
    cullMode = VK_CULL_MODE_FRONT_BIT;
    break;
  case Rhi::RhiCullMode::Back:
    cullMode = VK_CULL_MODE_BACK_BIT;
    break;
  default:
    vkrError("Invalid cull mode");
  }
  auto exfun = m_context->getExtensionFunction();
  vkCmdSetCullMode(m_commandBuffer, cullMode);
}
// Class: Queue
IFRIT_APIDECL Queue::Queue(EngineContext *ctx, VkQueue queue, uint32_t family,
                           VkQueueFlags capability)
    : m_context(ctx), m_queue(queue), m_queueFamily(family),
      m_capability(capability) {
  m_commandPool = std::make_unique<CommandPool>(ctx, family);
  m_timelineSemaphore = std::make_unique<TimelineSemaphore>(ctx);
}

IFRIT_APIDECL CommandBuffer *Queue::beginRecording() {
  if (m_cmdBufInUse.size() != 0) {
    vkrError("Command buffer still in use");
  }
  auto buffer = m_commandPool->allocateCommandBufferUnique();
  if (buffer == nullptr) {
    vkrError("Nullptr");
  }
  auto p = buffer.get();
  m_currentCommandBuffer = p;
  m_cmdBufInUse.push_back(std::move(buffer));
  p->beginRecord();
  return p;
}

IFRIT_APIDECL TimelineSemaphoreWait
Queue::submitCommand(const std::vector<TimelineSemaphoreWait> &waitSemaphores,
                     VkFence fence, VkSemaphore swapchainSemaphore) {
  m_recordedCounter++;

  std::vector<VkSemaphore> waitSemaphoreHandles;
  std::vector<uint64_t> waitValues;
  std::vector<VkPipelineStageFlags> waitStages;

  for (int i = 0; i < waitSemaphores.size(); i++) {
    waitSemaphoreHandles.push_back(waitSemaphores[i].m_semaphore);
    waitValues.push_back(waitSemaphores[i].m_value);
    waitStages.push_back(waitSemaphores[i].m_waitStage);
  }

  std::vector<uint64_t> signalValues;
  std::vector<VkSemaphore> signalSemaphores;
  signalValues.push_back(m_recordedCounter);
  signalSemaphores.push_back(m_timelineSemaphore->getSemaphore());

  if (swapchainSemaphore) {
    signalValues.push_back(0);
    signalSemaphores.push_back(swapchainSemaphore);
  }

  VkCommandBuffer commandBuffer = m_currentCommandBuffer->getCommandBuffer();
  m_currentCommandBuffer->endRecord();

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = size_cast<int>(waitSemaphoreHandles.size());
  submitInfo.pWaitSemaphores = waitSemaphoreHandles.data();
  submitInfo.pWaitDstStageMask = waitStages.data();
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  submitInfo.signalSemaphoreCount = size_cast<int>(signalSemaphores.size());
  submitInfo.pSignalSemaphores = signalSemaphores.data();

  VkTimelineSemaphoreSubmitInfo timelineInfo{};
  timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timelineInfo.waitSemaphoreValueCount = size_cast<int>(waitValues.size());
  timelineInfo.pWaitSemaphoreValues = waitValues.data();
  timelineInfo.signalSemaphoreValueCount = size_cast<int>(signalValues.size());
  timelineInfo.pSignalSemaphoreValues = signalValues.data();

  submitInfo.pNext = &timelineInfo;
  VkFence vfence = VK_NULL_HANDLE;
  if (fence) {
    vfence = fence;
  }
  vkrVulkanAssert(vkQueueSubmit(m_queue, 1, &submitInfo, vfence),
                  "Failed to submit command buffer");
  m_cmdBufInUse.pop_back();

  TimelineSemaphoreWait ret;
  ret.m_semaphore = m_timelineSemaphore.get()->getSemaphore();
  ret.m_value = m_recordedCounter;
  ret.m_waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  return ret;
}

IFRIT_APIDECL void Queue::waitIdle() { vkQueueWaitIdle(m_queue); }

IFRIT_APIDECL void Queue::counterReset() { m_recordedCounter = 0; }

// Class: CommandSubmissionList
IFRIT_APIDECL CommandSubmissionList::CommandSubmissionList(EngineContext *ctx)
    : m_context(ctx) {
  m_hostSyncSemaphore = std::make_unique<TimelineSemaphore>(ctx);
}

IFRIT_APIDECL void
CommandSubmissionList::addSubmission(const CommandSubmissionInfo &info) {
  m_submissions.push_back(info);
}

IFRIT_APIDECL void CommandSubmissionList::submit(bool hostSync) {
  for (int i = 0; i < m_submissions.size(); i++) {
    auto &submission = m_submissions[i];
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkPipelineStageFlags> waitStages;
    std::vector<VkSemaphore> signalSemaphores;
    std::vector<uint64_t> signalValues;
    std::vector<uint64_t> waitValues;

    for (int j = 0; j < submission.m_waitSemaphore.size(); j++) {
      waitSemaphores.push_back(submission.m_waitSemaphore[j]->getSemaphore());
      waitStages.push_back(submission.m_waitStages[j]);
      waitValues.push_back(submission.m_waitValues[j]);
    }
    for (int j = 0; j < submission.m_signalSemaphore.size(); j++) {
      signalSemaphores.push_back(
          submission.m_signalSemaphore[j]->getSemaphore());
      signalValues.push_back(submission.m_signalValues[j]);
    }

    if (hostSync) {
      waitSemaphores.push_back(m_hostSyncSemaphore->getSemaphore());
      waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
      waitValues.push_back(1);
    }

    VkSubmitInfo submitInfo{};
    const VkCommandBuffer commandBuffer =
        submission.m_commandBuffer->getCommandBuffer();
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = size_cast<int>(waitSemaphores.size());
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = size_cast<int>(signalSemaphores.size());
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount = size_cast<int>(waitValues.size());
    timelineInfo.pWaitSemaphoreValues = waitValues.data();
    timelineInfo.signalSemaphoreValueCount =
        size_cast<int>(signalValues.size());
    timelineInfo.pSignalSemaphoreValues = signalValues.data();

    submitInfo.pNext = &timelineInfo;

    vkrVulkanAssert(vkQueueSubmit(submission.m_queue->getQueue(), 1,
                                  &submitInfo, VK_NULL_HANDLE),
                    "Failed to submit command buffer");
  }
}

void Queue::runSyncCommand(
    std::function<void(const Rhi::RhiCommandBuffer *)> func) {
  auto cmd = beginRecording();
  func(cmd);
  submitCommand({}, VK_NULL_HANDLE);
  waitIdle();
}

std::unique_ptr<Rhi::RhiTaskSubmission>
Queue::runAsyncCommand(std::function<void(const Rhi::RhiCommandBuffer *)> func,
                       const std::vector<Rhi::RhiTaskSubmission *> &waitOn,
                       const std::vector<Rhi::RhiTaskSubmission *> &toIssue) {
  auto cmd = beginRecording();
  func(cmd);
  std::vector<TimelineSemaphoreWait> waitSemaphores;

  VkSemaphore swapchainSemaphore = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  if (toIssue.size() != 0) {
    vkrAssert(toIssue.size() == 1, "Only one semaphore can be issued");
    auto semaphore = checked_cast<TimelineSemaphoreWait>(toIssue[0]);
    vkrAssert(semaphore->m_isSwapchainSemaphore,
              "Only swapchain semaphore can be issued");
    swapchainSemaphore = semaphore->m_semaphore;
    fence = semaphore->m_fence;
  }
  for (int i = 0; i < waitOn.size(); i++) {
    auto semaphore = checked_cast<TimelineSemaphoreWait>(waitOn[i]);
    waitSemaphores.push_back(*semaphore);
  }
  return std::make_unique<TimelineSemaphoreWait>(
      submitCommand(waitSemaphores, fence, swapchainSemaphore));
}

void Queue::hostWaitEvent(Rhi::RhiTaskSubmission *event) {
  VkSemaphoreWaitInfo waitInfo{};
  auto sev = checked_cast<TimelineSemaphoreWait>(event);
  waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  waitInfo.semaphoreCount = 1;
  waitInfo.pSemaphores = &sev->m_semaphore;
  waitInfo.pValues = &sev->m_value;
  vkrVulkanAssert(vkWaitSemaphores(m_context->getDevice(), &waitInfo,
                                   std::numeric_limits<uint64_t>::max()),
                  "Failed to wait for semaphore");
}

// Queue Collections
IFRIT_APIDECL void QueueCollections::loadQueues() {
  auto &queueData = m_context->getQueueInfo();
  for (int i = 0; i < queueData.m_allQueues.size(); i++) {
    auto queue = queueData.m_allQueues[i];
    auto queueCapability =
        queueData.m_queueFamilies[queue.m_familyIndex].m_capability;
    m_queues.push_back(std::make_unique<Queue>(
        m_context, queue.m_queue, queue.m_familyIndex, queueCapability));
  }
}
IFRIT_APIDECL std::vector<Queue *> QueueCollections::getGraphicsQueues() {
  std::vector<Queue *> graphicsQueues;
  for (int i = 0; i < m_queues.size(); i++) {
    if (m_queues[i]->getCapability() & VK_QUEUE_GRAPHICS_BIT) {
      graphicsQueues.push_back(m_queues[i].get());
    }
  }
  return graphicsQueues;
}

IFRIT_APIDECL std::vector<Queue *> QueueCollections::getComputeQueues() {
  std::vector<Queue *> computeQueues;
  for (int i = 0; i < m_queues.size(); i++) {
    if (m_queues[i]->getCapability() & VK_QUEUE_COMPUTE_BIT) {
      computeQueues.push_back(m_queues[i].get());
    }
  }
  return computeQueues;
}

IFRIT_APIDECL std::vector<Queue *> QueueCollections::getTransferQueues() {
  std::vector<Queue *> transferQueues;
  for (int i = 0; i < m_queues.size(); i++) {
    if (m_queues[i]->getCapability() & VK_QUEUE_TRANSFER_BIT) {
      transferQueues.push_back(m_queues[i].get());
    }
  }
  return transferQueues;
}

} // namespace Ifrit::GraphicsBackend::VulkanGraphics