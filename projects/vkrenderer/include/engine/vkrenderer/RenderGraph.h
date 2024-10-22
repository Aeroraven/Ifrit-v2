#pragma once
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vkrenderer/include/engine/vkrenderer/Binding.h>
#include <vkrenderer/include/engine/vkrenderer/Command.h>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vkrenderer/include/engine/vkrenderer/MemoryResource.h>
#include <vkrenderer/include/engine/vkrenderer/Pipeline.h>
#include <vkrenderer/include/engine/vkrenderer/Shader.h>
#include <vkrenderer/include/engine/vkrenderer/Swapchain.h>

namespace Ifrit::Engine::VkRenderer {

class RenderGraph;
class CommandExecutor;
// End declaration of classes

enum class QueueRequirement {
  Graphics = VK_QUEUE_GRAPHICS_BIT,
  Compute = VK_QUEUE_COMPUTE_BIT,
  Transfer = VK_QUEUE_TRANSFER_BIT,

  Graphics_Compute = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
  Graphics_Transfer = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT,
  Compute_Transfer = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,

  Universal =
      VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT
};

struct StencilOps {
  VkStencilFaceFlags faceMask = VK_STENCIL_FACE_FRONT_AND_BACK;
  VkStencilOp failOp = VK_STENCIL_OP_KEEP;
  VkStencilOp passOp = VK_STENCIL_OP_KEEP;
  VkStencilOp depthFailOp = VK_STENCIL_OP_KEEP;
  VkCompareOp compareOp = VK_COMPARE_OP_ALWAYS;
};

class IFRIT_APIDECL SwapchainImageResource : public DeviceImage {
private:
  Swapchain *m_swapchain;

public:
  SwapchainImageResource(Swapchain *swapchain) : m_swapchain(swapchain) {
    m_isSwapchainImage = true;
  }
  virtual ~SwapchainImageResource() {}
  virtual VkFormat getFormat() override;
  virtual VkImage getImage() override;
  virtual VkImageView getImageView() override;
};

struct RenderPassResourceTransition {
  bool m_required = false;
  bool m_requireExplicitBarrier = false;
  bool m_allowUndefinedOldLayout = true;

  // Old layout should be set to undefined to allow tracer to
  // automatically transition the image
  VkImageLayout m_oldLayout;
  VkImageLayout m_newLayout;
  VkAccessFlags m_srcAccess;
  VkAccessFlags m_dstAccess;
};

class IFRIT_APIDECL RegisteredResource {
protected:
  uint32_t m_originalQueueFamily = 0xFFFFFFFF;
  VkImageLayout m_originalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkAccessFlags m_originalAccess = VK_ACCESS_NONE;

  uint32_t m_currentQueueFamily = 0xFFFFFFFF;
  VkImageLayout m_currentLayout;
  VkAccessFlags m_currentAccess;

  bool m_isSwapchainImage = false;

public:
  void resetState();
  inline bool getIsSwapchainImage() { return m_isSwapchainImage; }
  void recordTransition(const RenderPassResourceTransition &transition,
                        uint32_t queueFamily);
  virtual ~RegisteredResource() {}
};

class IFRIT_APIDECL RegisteredBufferHandle : public RegisteredResource {
private:
  bool m_isMultipleBuffered = false;
  SingleBuffer *m_buffer;
  MultiBuffer *m_multiBuffer;

public:
  RegisteredBufferHandle(SingleBuffer *buffer)
      : m_buffer(buffer), m_isMultipleBuffered(false) {}
  RegisteredBufferHandle(MultiBuffer *buffer)
      : m_multiBuffer(buffer), m_isMultipleBuffered(true) {}
  inline SingleBuffer *getBuffer(uint32_t frame) {
    return m_isMultipleBuffered ? m_multiBuffer->getBuffer(frame) : m_buffer;
  }
  inline bool isMultipleBuffered() { return m_isMultipleBuffered; }
  inline uint32_t getNumBuffers() {
    return m_isMultipleBuffered ? m_multiBuffer->getBufferCount() : 1;
  }
  virtual ~RegisteredBufferHandle() {}
};

class IFRIT_APIDECL RegisteredImageHandle : public RegisteredResource {
private:
  DeviceImage *m_image;

public:
  RegisteredImageHandle(DeviceImage *image) : m_image(image) {}
  inline virtual DeviceImage *getImage() { return m_image; }
  virtual ~RegisteredImageHandle() {}
};

class IFRIT_APIDECL RegisteredSwapchainImage : public RegisteredImageHandle {
private:
  SwapchainImageResource *m_image;

public:
  RegisteredSwapchainImage(SwapchainImageResource *image)
      : RegisteredImageHandle(image), m_image(image) {
    m_isSwapchainImage = true;
  }
  inline virtual SwapchainImageResource *getImage() { return m_image; }
  virtual ~RegisteredSwapchainImage() {}
};

struct RenderPassContext {
  const CommandBuffer *m_cmd;
  uint32_t m_frame;
};

struct RenderPassAttachment {
  RegisteredImageHandle *m_image = nullptr;
  VkAttachmentLoadOp m_loadOp;
  VkClearValue m_clearValue;
};

// Reuse pipelines that share the same layout
class IFRIT_APIDECL PipelineCache {
private:
  EngineContext *m_context;
  std::vector<std::unique_ptr<GraphicsPipeline>> m_graphicsPipelines;
  std::vector<GraphicsPipelineCreateInfo> m_graphicsPipelineCI;
  std::unordered_map<uint64_t, std::vector<int>> m_graphicsPipelineMap;

public:
  PipelineCache(EngineContext *context);
  PipelineCache(const PipelineCache &p) = delete;
  PipelineCache &operator=(const PipelineCache &p) = delete;
  uint64_t graphicsPipelineHash(const GraphicsPipelineCreateInfo &ci);
  bool graphicsPipelineEqual(const GraphicsPipelineCreateInfo &a,
                             const GraphicsPipelineCreateInfo &b);
  GraphicsPipeline *getGraphicsPipeline(const GraphicsPipelineCreateInfo &ci);
};

class IFRIT_APIDECL RenderGraphPass {
protected:
  EngineContext *m_context;
  DescriptorManager *m_descriptorManager;
  std::vector<RegisteredResource *> m_inputResources;
  std::vector<RegisteredResource *> m_outputResources;
  std::vector<RenderPassResourceTransition> m_inputTransition;
  std::vector<RenderPassResourceTransition> m_outputTransition;
  RenderPassContext m_passContext;
  bool m_operateOnSwapchain = false;
  bool m_passBuilt = false;

  std::unordered_map<uint32_t, std::vector<uint32_t>>
      m_resourceDescriptorHandle;
  std::vector<DescriptorType> m_passDescriptorLayout;
  std::vector<DescriptorBindRange> m_descriptorBindRange;
  std::function<void(RenderPassContext *)> m_recordFunction = nullptr;
  std::function<void(RenderPassContext *)> m_executeFunction = nullptr;

  uint32_t m_activeFrame = 0;

protected:
  std::vector<RegisteredResource *> &getInputResources();
  std::vector<RegisteredResource *> &getOutputResources();
  inline void setBuilt() { m_passBuilt = true; }

protected:
  void buildDescriptorParamHandle(uint32_t numMultiBuffers);

public:
  RenderGraphPass(EngineContext *context, DescriptorManager *descriptorManager)
      : m_context(context), m_descriptorManager(descriptorManager) {}
  void setPassDescriptorLayout(const std::vector<DescriptorType> &layout);

  inline void setActiveFrame(uint32_t frame) { m_activeFrame = frame; }
  virtual void addInputResource(RegisteredResource *resource,
                                RenderPassResourceTransition transition);
  virtual void addOutputResource(RegisteredResource *resource,
                                 RenderPassResourceTransition transition);
  virtual uint32_t getRequiredQueueCapability() = 0;
  virtual void withCommandBuffer(CommandBuffer *commandBuffer,
                                 std::function<void()> func) = 0;

  void addUniformBuffer(RegisteredBufferHandle *buffer, uint32_t position);

  inline bool getOperatesOnSwapchain() { return m_operateOnSwapchain; }
  inline bool isBuilt() const { return m_passBuilt; }

  void setRecordFunction(std::function<void(RenderPassContext *)> func);
  void setExecutionFunction(std::function<void(RenderPassContext *)> func);

  virtual void build(uint32_t numMultiBuffers) = 0;
  virtual void record() = 0;
  virtual void execute();



  friend class RenderGraph;
};

// Graphics Pass performs rendering operations
class IFRIT_APIDECL GraphicsPass : public RenderGraphPass {
protected:
  RenderPassAttachment m_depthAttachment;
  std::vector<RenderPassAttachment> m_colorAttachments;

  PipelineCache *m_pipelineCache;

  ShaderModule *m_vertexShader = nullptr;
  ShaderModule *m_fragmentShader = nullptr;
  ShaderModule *m_geometryShader = nullptr;
  ShaderModule *m_tessControlShader = nullptr;
  ShaderModule *m_tessEvalShader = nullptr;

  GraphicsPipeline *m_pipeline = nullptr;
  CommandBuffer *m_commandBuffer = nullptr;

  VkRect2D m_renderArea;
  bool m_depthWrite = false;
  bool m_depthTestEnable = false;
  bool m_stencilEnable = false;
  bool m_logicalOpEnable = false;
  bool m_depthBoundTestEnable = false;
  VkCompareOp m_depthCompareOp = VK_COMPARE_OP_ALWAYS;
  StencilOps m_stencilOp = {};
  VkLogicOp m_logicOp = VK_LOGIC_OP_COPY;
  std::vector<VkBool32> m_colorWrite;
  std::vector<VkBool32> m_blendEnable;
  std::vector<VkColorBlendEquationEXT> m_blendEquations;
  std::vector<VkColorComponentFlags> m_colorWriteMask;
  VkFrontFace m_frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  VkCullModeFlags m_cullMode = VK_CULL_MODE_NONE;

  VertexBufferDescriptor m_vertexBufferDescriptor;
  std::vector<RegisteredBufferHandle *> m_vertexBuffers;

  RegisteredBufferHandle *m_indexBuffer = nullptr;
  VkIndexType m_indexType;

public:
  GraphicsPass(EngineContext *context, PipelineCache *pipelineCache,
               DescriptorManager *descriptorManager);

  void record() override;
  void withCommandBuffer(CommandBuffer *commandBuffer,
                         std::function<void()> func);

  void addColorAttachment(RegisteredImageHandle *image, VkAttachmentLoadOp loadOp,
                          VkClearValue clearValue);
  void setDepthAttachment(RegisteredImageHandle *image, VkAttachmentLoadOp loadOp,
                          VkClearValue clearValue);

  void setVertexShader(ShaderModule *shader);
  void setFragmentShader(ShaderModule *shader);
  void setGeometryShader(ShaderModule *shader);
  void setTessControlShader(ShaderModule *shader);
  void setTessEvalShader(ShaderModule *shader);

  void setRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height);
  void setDepthWrite(bool write);
  void setColorWrite(const std::vector<uint32_t> &write);

  void setVertexInput(const VertexBufferDescriptor &descriptor,
                      const std::vector<RegisteredBufferHandle *> &buffers);
  void setIndexInput(RegisteredBufferHandle *buffer, VkIndexType type);

  uint32_t getRequiredQueueCapability() override;

  
  virtual void build(uint32_t numMultiBuffers) override;
  friend class RenderGraph;
};

struct RegisteredResourceGraphState {
  uint32_t indeg;
  uint32_t outdeg;
  uint32_t rwDeps;
};

class IFRIT_APIDECL RenderGraph {
private:
  EngineContext *m_context;
  DescriptorManager *m_descriptorManager;
  std::vector<std::unique_ptr<RenderGraphPass>> m_passes;
  std::vector<std::unique_ptr<RegisteredResource>> m_resources;
  std::unordered_map<RegisteredResource *, uint32_t> m_resourceMap;
  std::unique_ptr<PipelineCache> m_pipelineCache;

  // Graph States
  std::vector<RegisteredResourceGraphState> m_resourceGraphState;
  std::vector<RegisteredResource *> m_swapchainImageHandle;
  std::vector<uint32_t> m_subgraphBelonging;
  std::vector<std::vector<uint32_t>> m_subgraphs;

  std::vector<int> m_subGraphOperatesOnSwapchain;

  // Compilation
  bool m_comiled = false;
  std::vector<std::unique_ptr<CommandPool>> m_commandPools;
  std::vector<CommandSubmissionList> m_submissionLists;
  std::vector<std::unique_ptr<CommandBuffer>> m_commandBuffers;
  std::vector<Queue *> m_assignedQueues;

protected:
  std::vector<std::vector<uint32_t>> &getSubgraphs();
  void resizeCommandBuffers(uint32_t size);
  void buildPassDescriptors(uint32_t numMultiBuffer);

public:
  RenderGraph(EngineContext *context, DescriptorManager *descriptorManager);
  GraphicsPass *addGraphicsPass();
  RegisteredBufferHandle *registerBuffer(SingleBuffer *buffer);
  RegisteredBufferHandle *registerBuffer(MultiBuffer *buffer);
  RegisteredImageHandle *registerImage(DeviceImage *image);

  void build(uint32_t numMultiBuffers);
  void execute();

  friend class CommandExecutor;
};

class IFRIT_APIDECL CommandExecutor {
private:
  EngineContext *m_context;
  DescriptorManager *m_descriptorManager;

  std::vector<Queue *> m_queuesGraphics;
  std::vector<Queue *> m_queuesCompute;
  std::vector<Queue *> m_queuesTransfer;
  std::vector<Queue *> m_queues;

  RenderGraph *m_graph;
  Swapchain *m_swapchain;
  ResourceManager *m_resourceManager;

  std::unique_ptr<SwapchainImageResource> m_swapchainImageResource;
  std::unique_ptr<QueueCollections> m_queueCollections;

  std::vector<std::unique_ptr<RenderGraph>> m_renderGraph;

protected:
  void compileGraph(RenderGraph *graph, uint32_t numMultiBuffers);

public:
  CommandExecutor(EngineContext *context, Swapchain *swapchain,
                  DescriptorManager *descriptorManager,
                  ResourceManager *resourceManager)
      : m_context(context), m_swapchain(swapchain),
        m_descriptorManager(descriptorManager),
        m_resourceManager(resourceManager) {
    m_swapchainImageResource =
        std::make_unique<SwapchainImageResource>(swapchain);
    m_resourceManager->setDefaultCopies(swapchain->getNumBackbuffers());
  }
  CommandExecutor(const CommandExecutor &p) = delete;
  CommandExecutor &operator=(const CommandExecutor &p) = delete;

  RenderGraph *createRenderGraph();
  void setQueues(bool reqPresentQueue, int numGraphics, int numCompute,
                 int numTransfer);
  void syncMultiBufferStateWithSwapchain();
  void runRenderGraph(RenderGraph *graph);
  void runImmidiateCommand(std::function<void(CommandBuffer *)> func,
                           QueueRequirement req);

  SwapchainImageResource *getSwapchainImageResource();
};

} // namespace Ifrit::Engine::VkRenderer