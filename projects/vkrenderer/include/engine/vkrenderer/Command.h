#pragma once
#include <memory>
#include <vector>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>

namespace Ifrit::Engine::VkRenderer {
class CommandBuffer;

struct Viewport {
  float x;
  float y;
  float width;
  float height;
  float minDepth;
  float maxDepth;
};

struct Scissor {
  int32_t x;
  int32_t y;
  uint32_t width;
  uint32_t height;
};

enum class VertexInputRate { Vertex, Instance };

struct VertexBufferDescriptor {
  std::vector<VkVertexInputAttributeDescription2EXT> m_attributes;
  std::vector<VkVertexInputBindingDescription2EXT> m_bindings;
  inline void addBinding(std::vector<uint32_t> location,
                         std::vector<VkFormat> format,
                         std::vector<uint32_t> offset, uint32_t stride,
                         VertexInputRate inputRate = VertexInputRate::Vertex) {
    VkVertexInputBindingDescription2EXT binding{};
    binding.binding = m_bindings.size();
    binding.stride = stride;
    binding.divisor = 1;
    binding.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT;

    if (inputRate == VertexInputRate::Vertex) {
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
      attribute.format = format[i];
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
  uint32_t m_recordedCounter = 0;

public:
  TimelineSemaphore(EngineContext *ctx);
  ~TimelineSemaphore();
  inline VkSemaphore getSemaphore() const { return m_semaphore; }
};

struct TimelineSemaphoreWait {
  VkSemaphore m_semaphore;
  uint64_t m_value;
  VkFlags m_waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
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

class IFRIT_APIDECL CommandBuffer {
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
  void setViewports(const std::vector<Viewport> &viewport) const;
  void setScissors(const std::vector<Scissor> &scissor) const;
  void draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
            uint32_t firstInstance) const;
  void drawIndexed(uint32_t indexCount, uint32_t instanceCount,
                   uint32_t firstIndex, int32_t vertexOffset,
                   uint32_t firstInstance) const;

  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t size,
                  uint32_t srcOffset = 0, uint32_t dstOffset = 0) const;
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

class IFRIT_APIDECL Queue {
private:
  EngineContext *m_context;
  VkQueue m_queue;
  uint32_t m_queueFamily;
  uint32_t m_capability;
  std::unique_ptr<CommandPool> m_commandPool;
  std::unique_ptr<TimelineSemaphore> m_timelineSemaphore;
  std::vector<std::unique_ptr<CommandBuffer>> m_cmdBufInUse;
  uint32_t m_recordedCounter = 0;
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
  void submitCommand(const std::vector<TimelineSemaphoreWait> &waitSemaphores,
                     VkFence fence, VkSemaphore swapchainSemaphore = nullptr);
  void waitIdle();
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
} // namespace Ifrit::Engine::VkRenderer