#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/common/core/TypingUtil.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include "ifrit/vkgraphics/utility/Logger.h"

using namespace Ifrit::Common::Core;

namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics {
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
  vkCmdPipelineBarrier(
      m_commandBuffer, barrier.m_srcStage, barrier.m_dstStage,
      barrier.m_dependencyFlags, barrier.m_memoryBarriers.size(),
      barrier.m_memoryBarriers.data(), barrier.m_bufferMemoryBarriers.size(),
      barrier.m_bufferMemoryBarriers.data(),
      barrier.m_imageMemoryBarriers.size(),
      barrier.m_imageMemoryBarriers.data());
}

IFRIT_APIDECL void
CommandBuffer::setViewports(const std::vector<Viewport> &viewport) const {
  std::vector<VkViewport> vps;
  for (int i = 0; i < viewport.size(); i++) {
    VkViewport s = {viewport[i].x,        viewport[i].y,
                    viewport[i].width,    viewport[i].height,
                    viewport[i].minDepth, viewport[i].maxDepth};
    vps.push_back(s);
  }
  vkCmdSetViewport(m_commandBuffer, 0, vps.size(), vps.data());
}

IFRIT_APIDECL void
CommandBuffer::setScissors(const std::vector<Scissor> &scissor) const {
  std::vector<VkRect2D> scs;
  for (int i = 0; i < scissor.size(); i++) {
    VkRect2D s = {{scissor[i].x, scissor[i].y},
                  {scissor[i].width, scissor[i].height}};
    scs.push_back(s);
  }
  vkCmdSetScissor(m_commandBuffer, 0, scs.size(), scs.data());
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

IFRIT_APIDECL void CommandBuffer::copyBufferToImageAll(
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
  submitInfo.waitSemaphoreCount = waitSemaphoreHandles.size();
  submitInfo.pWaitSemaphores = waitSemaphoreHandles.data();
  submitInfo.pWaitDstStageMask = waitStages.data();
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  submitInfo.signalSemaphoreCount = signalSemaphores.size();
  submitInfo.pSignalSemaphores = signalSemaphores.data();

  VkTimelineSemaphoreSubmitInfo timelineInfo{};
  timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timelineInfo.waitSemaphoreValueCount = waitValues.size();
  timelineInfo.pWaitSemaphoreValues = waitValues.data();
  timelineInfo.signalSemaphoreValueCount = signalValues.size();
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
    submitInfo.waitSemaphoreCount = waitSemaphores.size();
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = signalSemaphores.size();
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount = waitValues.size();
    timelineInfo.pWaitSemaphoreValues = waitValues.data();
    timelineInfo.signalSemaphoreValueCount = signalValues.size();
    timelineInfo.pSignalSemaphoreValues = signalValues.data();

    submitInfo.pNext = &timelineInfo;

    vkrVulkanAssert(vkQueueSubmit(submission.m_queue->getQueue(), 1,
                                  &submitInfo, VK_NULL_HANDLE),
                    "Failed to submit command buffer");
  }
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

} // namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics