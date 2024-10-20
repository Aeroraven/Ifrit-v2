#pragma once
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vkrenderer/include/engine/vkrenderer/Command.h>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vkrenderer/include/engine/vkrenderer/MemoryResource.h>
#include <vkrenderer/include/engine/vkrenderer/Pipeline.h>
#include <vkrenderer/include/engine/vkrenderer/Shader.h>
#include <vkrenderer/include/engine/vkrenderer/Swapchain.h>

namespace Ifrit::Engine::VkRenderer {

class RenderGraph;
class RenderGraphExecutor;
// End declaration of classes

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
  SwapchainImageResource(Swapchain *swapchain) : m_swapchain(swapchain) {}
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

class IFRIT_APIDECL RegisteredBuffer : public RegisteredResource {
private:
  Buffer *m_buffer;

public:
  RegisteredBuffer(Buffer *buffer) : m_buffer(buffer) {}
  inline Buffer *getBuffer() { return m_buffer; }
  virtual ~RegisteredBuffer() {}
};

class IFRIT_APIDECL RegisteredImage : public RegisteredResource {
private:
  DeviceImage *m_image;

public:
  RegisteredImage(DeviceImage *image) : m_image(image) {}
  inline virtual DeviceImage *getImage() { return m_image; }
  virtual ~RegisteredImage() {}
};

class IFRIT_APIDECL RegisteredSwapchainImage : public RegisteredImage {
private:
  SwapchainImageResource *m_image;

public:
  RegisteredSwapchainImage(SwapchainImageResource *image)
      : RegisteredImage(image), m_image(image) {
    m_isSwapchainImage = true;
  }
  inline virtual SwapchainImageResource *getImage() { return m_image; }
  virtual ~RegisteredSwapchainImage() {}
};

struct RenderPassContext {
  const CommandBuffer *m_cmd;
};

struct RenderPassAttachment {
  RegisteredImage *m_image = nullptr;
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
  std::vector<RegisteredResource *> m_inputResources;
  std::vector<RegisteredResource *> m_outputResources;
  std::vector<RenderPassResourceTransition> m_inputTransition;
  std::vector<RenderPassResourceTransition> m_outputTransition;
  RenderPassContext m_passContext;
  bool m_operateOnSwapchain = false;
  bool m_passBuilt = false;

protected:
  std::vector<RegisteredResource *> &getInputResources();
  std::vector<RegisteredResource *> &getOutputResources();
  inline void setBuilt() { m_passBuilt = true; }    

public:
  RenderGraphPass(EngineContext *context) : m_context(context) {}
  virtual void record() = 0;
  virtual void addInputResource(RegisteredResource *resource,
                                RenderPassResourceTransition transition);
  virtual void addOutputResource(RegisteredResource *resource,
                                 RenderPassResourceTransition transition);
  virtual uint32_t getRequiredQueueCapability() = 0;
  virtual void withCommandBuffer(CommandBuffer *commandBuffer,
                                 std::function<void()> func) = 0;
  friend class RenderGraph;

  inline bool getOperatesOnSwapchain() { return m_operateOnSwapchain; }
  inline bool isBuilt() const { return m_passBuilt; }
  virtual void build() = 0;

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

  std::function<void(RenderPassContext *)> m_executeFunction = nullptr;
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

public:
  GraphicsPass(EngineContext *context, PipelineCache *pipelineCache);

  void record() override;
  void withCommandBuffer(CommandBuffer *commandBuffer,
                         std::function<void()> func);

  void addColorAttachment(RegisteredImage *image, VkAttachmentLoadOp loadOp,
                          VkClearValue clearValue);
  void setDepthAttachment(RegisteredImage *image, VkAttachmentLoadOp loadOp,
                          VkClearValue clearValue);

  void setVertexShader(ShaderModule *shader);
  void setFragmentShader(ShaderModule *shader);
  void setGeometryShader(ShaderModule *shader);
  void setTessControlShader(ShaderModule *shader);
  void setTessEvalShader(ShaderModule *shader);

  void setRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height);
  void setDepthWrite(bool write);
  void setColorWrite(const std::vector<uint32_t> &write); // Afraid to use std::vector<bool>
                                           
  uint32_t getRequiredQueueCapability() override;

  void setRecordFunction(std::function<void(RenderPassContext *)> func);
  virtual void build() override;
};

struct RegisteredResourceGraphState {
  uint32_t indeg;
  uint32_t outdeg;
  uint32_t rwDeps;
};

class IFRIT_APIDECL RenderGraph {
private:
  EngineContext *m_context;
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

public:
  RenderGraph(EngineContext *context);
  GraphicsPass *addGraphicsPass();
  RegisteredBuffer *registerBuffer(Buffer *buffer);
  RegisteredImage *registerImage(DeviceImage *image);
  RegisteredSwapchainImage *
  registerSwapchainImage(SwapchainImageResource *image);

  void build();
  void execute();

  friend class RenderGraphExecutor;
};

class IFRIT_APIDECL RenderGraphExecutor {
private:
  EngineContext *m_context;
  std::vector<Queue *> m_queues;
  RenderGraph *m_graph;
  Swapchain *m_swapchain;
  std::unique_ptr<SwapchainImageResource> m_swapchainImageResource;

public:
  RenderGraphExecutor(EngineContext *context, Swapchain *swapchain)
      : m_context(context), m_swapchain(swapchain) {
    m_swapchainImageResource =
        std::make_unique<SwapchainImageResource>(swapchain);
  }
  void setQueues(const std::vector<Queue *> &queues);
  void compileGraph(RenderGraph *graph);
  void runGraph(RenderGraph *graph);
  SwapchainImageResource *getSwapchainImageResource();
};

} // namespace Ifrit::Engine::VkRenderer