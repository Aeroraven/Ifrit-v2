
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
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include <memory>
#include <vector>

namespace Ifrit::GraphicsBackend::VulkanGraphics {

class CommandBuffer;

class IFRIT_APIDECL VertexBufferDescriptor : public Rhi::RhiVertexBufferView {
public:
  Vec<VkVertexInputAttributeDescription2EXT> m_attributes;
  Vec<VkVertexInputBindingDescription2EXT> m_bindings;
  inline void addBinding(Vec<u32> location, Vec<Rhi::RhiImageFormat> format, Vec<u32> offset, u32 stride,
                         Rhi::RhiVertexInputRate inputRate = Rhi::RhiVertexInputRate::Vertex) override {
    VkVertexInputBindingDescription2EXT binding{};
    binding.binding = Ifrit::Common::Utility::size_cast<u32>(m_bindings.size());
    binding.stride = stride;
    binding.divisor = 1;
    binding.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT;

    if (inputRate == Rhi::RhiVertexInputRate::Vertex) {
      binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    } else {
      binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    }
    m_bindings.push_back(binding);

    for (i32 i = 0; i < location.size(); i++) {
      VkVertexInputAttributeDescription2EXT attribute{};
      attribute.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT;
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
  u64 m_recordedCounter = 0;

public:
  TimelineSemaphore(EngineContext *ctx);
  ~TimelineSemaphore();
  inline VkSemaphore getSemaphore() const { return m_semaphore; }
};

class TimelineSemaphoreWait : public Rhi::RhiTaskSubmission {
public:
  VkSemaphore m_semaphore;
  VkFence m_fence;
  u64 m_value;
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
  Vec<VkMemoryBarrier> m_memoryBarriers;
  Vec<VkBufferMemoryBarrier> m_bufferMemoryBarriers;
  Vec<VkImageMemoryBarrier> m_imageMemoryBarriers;

public:
  PipelineBarrier(EngineContext *ctx, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                  VkDependencyFlags dependencyFlags)
      : m_context(ctx), m_srcStage(srcStage), m_dstStage(dstStage), m_dependencyFlags(dependencyFlags) {}

  void addMemoryBarrier(VkMemoryBarrier barrier);
  void addBufferMemoryBarrier(VkBufferMemoryBarrier barrier);
  void addImageMemoryBarrier(VkImageMemoryBarrier barrier);

  friend class CommandBuffer;
};

class IFRIT_APIDECL CommandBuffer : public Rhi::RhiCommandList {
private:
  EngineContext *m_context;
  VkCommandBuffer m_commandBuffer;
  u32 m_queueFamily;

public:
  CommandBuffer(EngineContext *ctx, VkCommandBuffer buffer, u32 queueFamily)
      : m_context(ctx), m_commandBuffer(buffer), m_queueFamily(queueFamily) {}
  virtual ~CommandBuffer() {}

  inline u32 getQueueFamily() const { return m_queueFamily; }
  void beginRecord();
  void endRecord();
  void reset();
  inline VkCommandBuffer getCommandBuffer() const { return m_commandBuffer; }

  // Functionality
  void pipelineBarrier(const PipelineBarrier &barrier) const;

  void draw(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const;

  void drawMeshTasks(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const;

  void drawIndexed(u32 indexCount, u32 instanceCount, u32 firstIndex, int32_t vertexOffset, u32 firstInstance) const;

  void copyBuffer(const Rhi::RhiBuffer *srcBuffer, const Rhi::RhiBuffer *dstBuffer, u32 size, u32 srcOffset = 0,
                  u32 dstOffset = 0) const;

  void copyBufferToImageAllInternal(const Rhi::RhiBuffer *srcBuffer, VkImage dstImage, VkImageLayout dstLayout,
                                    u32 width, u32 height, u32 depth) const;

  // Rhi compatible
  void setViewports(const Vec<Rhi::RhiViewport> &viewport) const override;
  void setScissors(const Vec<Rhi::RhiScissor> &scissor) const override;
  void dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const override;
  void drawMeshTasksIndirect(const Rhi::RhiBuffer *buffer, u32 offset, u32 drawCount, u32 stride) const override;

  void imageBarrier(Rhi::RhiTexture *texture, Rhi::RhiResourceState src, Rhi::RhiResourceState dst,
                    Rhi::RhiImageSubResource subResource) const; // DEPRECATED

  void attachBindlessReferenceGraphics(Rhi::RhiGraphicsPass *pass, u32 setId,
                                       Rhi::RhiBindlessDescriptorRef *ref) const override;

  void attachBindlessReferenceCompute(Rhi::RhiComputePass *pass, u32 setId,
                                      Rhi::RhiBindlessDescriptorRef *ref) const override;

  void attachVertexBufferView(const Rhi::RhiVertexBufferView &view) const override;

  void attachVertexBuffers(u32 firstSlot, const Vec<Rhi::RhiBuffer *> &buffers) const override;

  void drawInstanced(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const override;

  void bufferClear(const Rhi::RhiBuffer *buffer, u32 val) const override;

  void dispatchIndirect(const Rhi::RhiBuffer *buffer, u32 offset) const override;

  void setPushConst(Rhi::RhiComputePass *pass, u32 offset, u32 size, const void *data) const override;
  void setPushConst(Rhi::RhiGraphicsPass *pass, u32 offset, u32 size, const void *data) const override;
  void clearUAVImageFloat(const Rhi::RhiTexture *texture, Rhi::RhiImageSubResource subResource,
                          const std::array<float, 4> &val) const override;

  void resourceBarrier(const Vec<Rhi::RhiResourceBarrier> &barriers) const override;

  void globalMemoryBarrier() const override;

  void beginScope(const std::string &name) const override;
  void endScope() const override;

  void copyImage(const Rhi::RhiTexture *src, Rhi::RhiImageSubResource srcSub, const Rhi::RhiTexture *dst,
                 Rhi::RhiImageSubResource dstSub) const override;

  void copyBufferToImage(const Rhi::RhiBuffer *src, const Rhi::RhiTexture *dst,
                         Rhi::RhiImageSubResource dstSub) const override;
  void setCullMode(Rhi::RhiCullMode mode) const override;
};

class IFRIT_APIDECL CommandPool {
private:
  EngineContext *m_context;
  u32 m_queueFamily;
  VkCommandPool m_commandPool;

protected:
  void init();

public:
  CommandPool(EngineContext *ctx, u32 chosenQueueFamily) : m_context(ctx), m_queueFamily(chosenQueueFamily) { init(); }
  ~CommandPool();
  Ref<CommandBuffer> allocateCommandBuffer();
  Uref<CommandBuffer> allocateCommandBufferUnique();
};

class IFRIT_APIDECL Queue : public Rhi::RhiQueue {
private:
  EngineContext *m_context;
  VkQueue m_queue;
  u32 m_queueFamily;
  u32 m_capability;
  Uref<CommandPool> m_commandPool;
  Uref<TimelineSemaphore> m_timelineSemaphore;
  Vec<Uref<CommandBuffer>> m_cmdBufInUse;
  u64 m_recordedCounter = 0;
  CommandBuffer *m_currentCommandBuffer = nullptr;

public:
  Queue() { printf("Runtime Error:queue\n"); }
  Queue(EngineContext *ctx, VkQueue queue, u32 queueFamily, u32 capability);

  virtual ~Queue() {}
  inline VkQueue getQueue() const { return m_queue; }
  inline u32 getQueueFamily() const { return m_queueFamily; }
  inline u32 getCapability() const { return m_capability; }

  CommandBuffer *beginRecording();
  TimelineSemaphoreWait submitCommand(const Vec<TimelineSemaphoreWait> &waitSemaphores, VkFence fence,
                                      VkSemaphore swapchainSemaphore = nullptr);
  void waitIdle();
  void counterReset();

  // for rhi layers override
  void runSyncCommand(std::function<void(const Rhi::RhiCommandList *)> func) override;

  Uref<Rhi::RhiTaskSubmission> runAsyncCommand(std::function<void(const Rhi::RhiCommandList *)> func,
                                               const Vec<Rhi::RhiTaskSubmission *> &waitOn,
                                               const Vec<Rhi::RhiTaskSubmission *> &toIssue) override;

  void hostWaitEvent(Rhi::RhiTaskSubmission *event) override;
};

class IFRIT_APIDECL QueueCollections {
private:
  EngineContext *m_context;
  Vec<Uref<Queue>> m_queues;

public:
  QueueCollections(EngineContext *ctx) : m_context(ctx) {}
  QueueCollections(const QueueCollections &p) = delete; // copy constructor
  QueueCollections &operator=(const QueueCollections &p) = delete;

  void loadQueues();
  Vec<Queue *> getGraphicsQueues();
  Vec<Queue *> getComputeQueues();
  Vec<Queue *> getTransferQueues();
};

struct CommandSubmissionInfo {
  CommandBuffer *m_commandBuffer;
  Queue *m_queue;
  Vec<TimelineSemaphore *> m_waitSemaphore;
  Vec<u64> m_waitValues;
  Vec<TimelineSemaphore *> m_signalSemaphore;
  Vec<u64> m_signalValues;
  Vec<VkFlags> m_waitStages;
};

class IFRIT_APIDECL CommandSubmissionList {
private:
  EngineContext *m_context;
  Vec<CommandSubmissionInfo> m_submissions;
  Uref<TimelineSemaphore> m_hostSyncSemaphore = nullptr;

public:
  CommandSubmissionList(EngineContext *ctx);
  void addSubmission(const CommandSubmissionInfo &info);
  void submit(bool hostSync = false);
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics