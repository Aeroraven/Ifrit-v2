
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
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include <memory>
#include <vector>

namespace Ifrit::GraphicsBackend::VulkanGraphics {
class CommandBuffer;

class IFRIT_APIDECL VertexBufferDescriptor : public Rhi::RhiVertexBufferView {
public:
  std::vector<VkVertexInputAttributeDescription2EXT> m_attributes;
  std::vector<VkVertexInputBindingDescription2EXT> m_bindings;
  inline void addBinding(std::vector<uint32_t> location,
                         std::vector<Rhi::RhiImageFormat> format,
                         std::vector<uint32_t> offset, uint32_t stride,
                         Rhi::RhiVertexInputRate inputRate =
                             Rhi::RhiVertexInputRate::Vertex) override {
    VkVertexInputBindingDescription2EXT binding{};
    binding.binding =
        Ifrit::Common::Utility::size_cast<uint32_t>(m_bindings.size());
    binding.stride = stride;
    binding.divisor = 1;
    binding.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT;

    if (inputRate == Rhi::RhiVertexInputRate::Vertex) {
      binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    } else {
      binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    }
    m_bindings.push_back(binding);

    for (int i = 0; i < location.size(); i++) {
      VkVertexInputAttributeDescription2EXT attribute{};
      attribute.sType =
          VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT;
      attribute.binding = binding.binding;
      attribute.format = static_cast<VkFormat>(format[i]);
      attribute.location = location[i];
      attribute.offset = offset[i];
      m_attributes.push_back(attribute);
    }
  }
};

class IFRIT_APIDECL TimelineSemaphore {
private:
  EngineContext *m_context;
  VkSemaphore m_semaphore;
  uint64_t m_recordedCounter = 0;

public:
  TimelineSemaphore(EngineContext *ctx);
  ~TimelineSemaphore();
  inline VkSemaphore getSemaphore() const { return m_semaphore; }
};

class TimelineSemaphoreWait : public Rhi::RhiTaskSubmission {
public:
  VkSemaphore m_semaphore;
  VkFence m_fence;
  uint64_t m_value;
  VkFlags m_waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  bool m_isSwapchainSemaphore = false;
  TimelineSemaphoreWait &operator=(const TimelineSemaphoreWait &other) {
    m_semaphore = other.m_semaphore;
    m_value = other.m_value;
    m_waitStage = other.m_waitStage;
    m_isSwapchainSemaphore = other.m_isSwapchainSemaphore;
    return *this;
  }
};

class IFRIT_APIDECL PipelineBarrier {
private:
  EngineContext *m_context;
  VkPipelineStageFlags m_srcStage;
  VkPipelineStageFlags m_dstStage;
  VkDependencyFlags m_dependencyFlags;
  std::vector<VkMemoryBarrier> m_memoryBarriers;
  std::vector<VkBufferMemoryBarrier> m_bufferMemoryBarriers;
  std::vector<VkImageMemoryBarrier> m_imageMemoryBarriers;

public:
  PipelineBarrier(EngineContext *ctx, VkPipelineStageFlags srcStage,
                  VkPipelineStageFlags dstStage,
                  VkDependencyFlags dependencyFlags)
      : m_context(ctx), m_srcStage(srcStage), m_dstStage(dstStage),
        m_dependencyFlags(dependencyFlags) {}

  void addMemoryBarrier(VkMemoryBarrier barrier);
  void addBufferMemoryBarrier(VkBufferMemoryBarrier barrier);
  void addImageMemoryBarrier(VkImageMemoryBarrier barrier);

  friend class CommandBuffer;
};

class IFRIT_APIDECL CommandBuffer : public Rhi::RhiCommandBuffer {
private:
  EngineContext *m_context;
  VkCommandBuffer m_commandBuffer;
  uint32_t m_queueFamily;

public:
  CommandBuffer(EngineContext *ctx, VkCommandBuffer buffer,
                uint32_t queueFamily)
      : m_context(ctx), m_commandBuffer(buffer), m_queueFamily(queueFamily) {}
  virtual ~CommandBuffer() {}

  inline uint32_t getQueueFamily() const { return m_queueFamily; }
  void beginRecord();
  void endRecord();
  void reset();
  inline VkCommandBuffer getCommandBuffer() const { return m_commandBuffer; }

  // Functionality
  void pipelineBarrier(const PipelineBarrier &barrier) const;

  void draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
            uint32_t firstInstance) const;

  void drawMeshTasks(uint32_t groupCountX, uint32_t groupCountY,
                     uint32_t groupCountZ) const;

  void drawIndexed(uint32_t indexCount, uint32_t instanceCount,
                   uint32_t firstIndex, int32_t vertexOffset,
                   uint32_t firstInstance) const;

  void copyBuffer(const Rhi::RhiBuffer *srcBuffer,
                  const Rhi::RhiBuffer *dstBuffer, uint32_t size,
                  uint32_t srcOffset = 0, uint32_t dstOffset = 0) const;

  void copyBufferToImageAllInternal(const Rhi::RhiBuffer *srcBuffer,
                                    VkImage dstImage, VkImageLayout dstLayout,
                                    uint32_t width, uint32_t height,
                                    uint32_t depth) const;

  // Rhi compatible
  void
  setViewports(const std::vector<Rhi::RhiViewport> &viewport) const override;
  void setScissors(const std::vector<Rhi::RhiScissor> &scissor) const override;
  void dispatch(uint32_t groupCountX, uint32_t groupCountY,
                uint32_t groupCountZ) const override;
  void drawMeshTasksIndirect(const Rhi::RhiBuffer *buffer, uint32_t offset,
                             uint32_t drawCount,
                             uint32_t stride) const override;
  void imageBarrier(const Rhi::RhiTexture *texture, Rhi::RhiResourceState src,
                    Rhi::RhiResourceState dst,
                    Rhi::RhiImageSubResource subResource) const override;

  void attachBindlessReferenceGraphics(
      Rhi::RhiGraphicsPass *pass, uint32_t setId,
      Rhi::RhiBindlessDescriptorRef *ref) const override;

  void attachBindlessReferenceCompute(
      Rhi::RhiComputePass *pass, uint32_t setId,
      Rhi::RhiBindlessDescriptorRef *ref) const override;

  virtual void
  attachVertexBufferView(const Rhi::RhiVertexBufferView &view) const override;

  virtual void attachVertexBuffers(
      uint32_t firstSlot,
      const std::vector<Rhi::RhiBuffer *> &buffers) const override;

  virtual void drawInstanced(uint32_t vertexCount, uint32_t instanceCount,
                             uint32_t firstVertex,
                             uint32_t firstInstance) const override;

  virtual void uavBufferBarrier(const Rhi::RhiBuffer *buffer) const override;

  virtual void uavBufferClear(const Rhi::RhiBuffer *buffer,
                              uint32_t val) const override;

  virtual void dispatchIndirect(const Rhi::RhiBuffer *buffer,
                                uint32_t offset) const override;

  virtual void setPushConst(Rhi::RhiComputePass *pass, uint32_t offset,
                            uint32_t size, const void *data) const override;
  virtual void setPushConst(Rhi::RhiGraphicsPass *pass, uint32_t offset,
                            uint32_t size, const void *data) const override;
  virtual void
  clearUAVImageFloat(const Rhi::RhiTexture *texture,
                     Rhi::RhiImageSubResource subResource,
                     const std::array<float, 4> &val) const override;

  virtual void resourceBarrier(
      const std::vector<Rhi::RhiResourceBarrier> &barriers) const override;

  virtual void globalMemoryBarrier() const override;

  virtual void beginScope(const std::string &name) const override;
  virtual void endScope() const override;

  virtual void copyImage(const Rhi::RhiTexture *src,
                         Rhi::RhiImageSubResource srcSub,
                         const Rhi::RhiTexture *dst,
                         Rhi::RhiImageSubResource dstSub) const override;

  virtual void
  copyBufferToImage(const Rhi::RhiBuffer *src, const Rhi::RhiTexture *dst,
                    Rhi::RhiImageSubResource dstSub) const override;
  virtual void setCullMode(Rhi::RhiCullMode mode) const override;
};

class IFRIT_APIDECL CommandPool {
private:
  EngineContext *m_context;
  uint32_t m_queueFamily;
  VkCommandPool m_commandPool;

protected:
  void init();

public:
  CommandPool(EngineContext *ctx, uint32_t chosenQueueFamily)
      : m_context(ctx), m_queueFamily(chosenQueueFamily) {
    init();
  }
  ~CommandPool();
  std::shared_ptr<CommandBuffer> allocateCommandBuffer();
  std::unique_ptr<CommandBuffer> allocateCommandBufferUnique();
};

class IFRIT_APIDECL Queue : public Rhi::RhiQueue {
private:
  EngineContext *m_context;
  VkQueue m_queue;
  uint32_t m_queueFamily;
  uint32_t m_capability;
  std::unique_ptr<CommandPool> m_commandPool;
  std::unique_ptr<TimelineSemaphore> m_timelineSemaphore;
  std::vector<std::unique_ptr<CommandBuffer>> m_cmdBufInUse;
  uint64_t m_recordedCounter = 0;
  CommandBuffer *m_currentCommandBuffer = nullptr;

public:
  Queue() { printf("Runtime Error:queue\n"); }
  Queue(EngineContext *ctx, VkQueue queue, uint32_t queueFamily,
        uint32_t capability);

  virtual ~Queue() {}
  inline VkQueue getQueue() const { return m_queue; }
  inline uint32_t getQueueFamily() const { return m_queueFamily; }
  inline uint32_t getCapability() const { return m_capability; }

  CommandBuffer *beginRecording();
  TimelineSemaphoreWait
  submitCommand(const std::vector<TimelineSemaphoreWait> &waitSemaphores,
                VkFence fence, VkSemaphore swapchainSemaphore = nullptr);
  void waitIdle();
  void counterReset();

  // for rhi layers override
  void runSyncCommand(
      std::function<void(const Rhi::RhiCommandBuffer *)> func) override;

  std::unique_ptr<Rhi::RhiTaskSubmission> runAsyncCommand(
      std::function<void(const Rhi::RhiCommandBuffer *)> func,
      const std::vector<Rhi::RhiTaskSubmission *> &waitOn,
      const std::vector<Rhi::RhiTaskSubmission *> &toIssue) override;

  void hostWaitEvent(Rhi::RhiTaskSubmission *event) override;
};

class IFRIT_APIDECL QueueCollections {
private:
  EngineContext *m_context;
  std::vector<std::unique_ptr<Queue>> m_queues;

public:
  QueueCollections(EngineContext *ctx) : m_context(ctx) {}
  QueueCollections(const QueueCollections &p) = delete; // copy constructor
  QueueCollections &operator=(const QueueCollections &p) = delete;

  void loadQueues();
  std::vector<Queue *> getGraphicsQueues();
  std::vector<Queue *> getComputeQueues();
  std::vector<Queue *> getTransferQueues();
};

struct CommandSubmissionInfo {
  CommandBuffer *m_commandBuffer;
  Queue *m_queue;
  std::vector<TimelineSemaphore *> m_waitSemaphore;
  std::vector<uint64_t> m_waitValues;
  std::vector<TimelineSemaphore *> m_signalSemaphore;
  std::vector<uint64_t> m_signalValues;
  std::vector<VkFlags> m_waitStages;
};

class IFRIT_APIDECL CommandSubmissionList {
private:
  EngineContext *m_context;
  std::vector<CommandSubmissionInfo> m_submissions;
  std::unique_ptr<TimelineSemaphore> m_hostSyncSemaphore = nullptr;

public:
  CommandSubmissionList(EngineContext *ctx);
  void addSubmission(const CommandSubmissionInfo &info);
  void submit(bool hostSync = false);
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics