#pragma once
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Binding.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Pipeline.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderTargets.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Shader.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Swapchain.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// TODO: render graph abstraction is deprecated in this level
// it's intended to be used in a higher level.
// So this file only encapsulates 'GraphicsPass' and 'ComputePass'
// Some redundant code is planned to be removed

namespace Ifrit::GraphicsBackend::VulkanGraphics {

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

class IFRIT_APIDECL SwapchainImageResource : public SingleDeviceImage {
private:
  Swapchain *m_swapchain;

public:
  SwapchainImageResource(Swapchain *swapchain) : m_swapchain(swapchain) {
    m_isSwapchainImage = true;
  }
  virtual ~SwapchainImageResource() {}
  virtual VkFormat getFormat() const override;
  virtual VkImage getImage() const override;
  virtual VkImageView getImageView() override;
  virtual VkImageView getImageViewMipLayer(uint32_t mip, uint32_t layer,
                                           uint32_t mipRange,
                                           uint32_t layerRange) override;
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
  SingleDeviceImage *m_image;

public:
  RegisteredImageHandle(SingleDeviceImage *image) : m_image(image) {}
  inline virtual SingleDeviceImage *getImage(uint32_t x) { return m_image; }
  inline uint32_t getNumBuffers() { return 1; }
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
  inline virtual SwapchainImageResource *getImage(uint32_t x) override {
    return m_image;
  }
  virtual ~RegisteredSwapchainImage() {}
};

class IFRIT_APIDECL RegisteredSamplerHandle : public RegisteredResource {
private:
  Sampler *m_sampler;

public:
  RegisteredSamplerHandle(Sampler *sampler) : m_sampler(sampler) {}
  inline Sampler *getSampler() { return m_sampler; }
  virtual ~RegisteredSamplerHandle() {}
};

struct RenderPassAttachment {
  RegisteredImageHandle *m_image = nullptr;
  VkAttachmentLoadOp m_loadOp;
  VkClearValue m_clearValue;
};

class IFRIT_APIDECL RegisteredResourceMapper
    : public Ifrit::Common::Utility::NonCopyable {
private:
  std::vector<std::unique_ptr<RegisteredResource>> m_resources;
  std::unordered_map<SingleBuffer *, uint32_t> m_bufferMap;
  std::unordered_map<SingleDeviceImage *, uint32_t> m_imageMap;
  std::unordered_map<MultiBuffer *, uint32_t> m_multiBufferMap;
  std::unordered_map<Sampler *, uint32_t> m_samplerMap;
  // swapchain

public:
  RegisteredResourceMapper() {}
  inline RegisteredResource *getBufferIndex(SingleBuffer *buffer) {
    // Check if buffer is already registered
    auto it = m_bufferMap.find(buffer);
    if (it != m_bufferMap.end()) {
      return m_resources[it->second].get();
    } else {
      auto registeredBuffer = std::make_unique<RegisteredBufferHandle>(buffer);
      auto ptr = registeredBuffer.get();
      m_resources.push_back(std::move(registeredBuffer));
      m_bufferMap[buffer] =
          Ifrit::Common::Utility::size_cast<uint32_t>(m_resources.size()) - 1;
      return ptr;
    }
  }
  inline RegisteredResource *getImageIndex(SingleDeviceImage *image) {
    // Check if image is already registered
    auto it = m_imageMap.find(image);
    if (it != m_imageMap.end()) {
      return m_resources[it->second].get();
    } else {
      if (image->getIsSwapchainImage()) {
        using namespace Ifrit::Common::Utility;
        auto swapchainImage = checked_cast<SwapchainImageResource>(image);
        auto registeredImage =
            std::make_unique<RegisteredSwapchainImage>(swapchainImage);
        auto ptr = registeredImage.get();
        m_resources.push_back(std::move(registeredImage));
        m_imageMap[image] = size_cast<uint32_t>(m_resources.size()) - 1;
        return ptr;
      }
      auto registeredImage = std::make_unique<RegisteredImageHandle>(image);
      auto ptr = registeredImage.get();
      m_resources.push_back(std::move(registeredImage));
      m_imageMap[image] =
          Ifrit::Common::Utility::size_cast<uint32_t>(m_resources.size()) - 1;
      return ptr;
    }
  }
  inline RegisteredResource *getMultiBufferIndex(MultiBuffer *buffer) {
    // Check if buffer is already registered
    auto it = m_multiBufferMap.find(buffer);
    if (it != m_multiBufferMap.end()) {
      return m_resources[it->second].get();
    } else {
      auto registeredBuffer = std::make_unique<RegisteredBufferHandle>(buffer);
      auto ptr = registeredBuffer.get();
      m_resources.push_back(std::move(registeredBuffer));
      m_multiBufferMap[buffer] =
          Ifrit::Common::Utility::size_cast<uint32_t>(m_resources.size()) - 1;
      return ptr;
    }
  }
  inline RegisteredResource *getSamplerIndex(Sampler *sampler) {
    throw std::runtime_error("Not implemented");
    return nullptr;
  }
};

class IFRIT_APIDECL RenderGraphPass {
protected:
  EngineContext *m_context;
  RegisteredResourceMapper *m_mapper;
  DescriptorManager *m_descriptorManager;
  std::vector<RegisteredResource *> m_inputResources;
  std::vector<RegisteredResource *> m_outputResources;
  std::vector<RenderPassResourceTransition> m_inputTransition;
  std::vector<RenderPassResourceTransition> m_outputTransition;
  Rhi::RhiRenderPassContext m_passContext;
  bool m_operateOnSwapchain = false;
  bool m_passBuilt = false;

  std::unordered_map<uint32_t, std::vector<uint32_t>>
      m_resourceDescriptorHandle;
  std::vector<Rhi::RhiDescriptorType> m_passDescriptorLayout;
  std::vector<DescriptorBindRange> m_descriptorBindRange;
  std::function<void(Rhi::RhiRenderPassContext *)> m_recordFunction = nullptr;
  std::function<void(Rhi::RhiRenderPassContext *)> m_recordPostFunction =
      nullptr;
  std::function<void(Rhi::RhiRenderPassContext *)> m_executeFunction = nullptr;
  const CommandBuffer *m_commandBuffer = nullptr;

  // SSBOs
  std::vector<RegisteredBufferHandle *> m_ssbos;
  std::vector<Rhi::RhiResourceAccessType> m_ssboAccess;

  uint32_t m_activeFrame = 0;
  uint32_t m_defaultMultibuffers = UINT_MAX;

  // 2x
  uint32_t m_numBindlessDescriptorSets = 0;
  uint32_t m_pushConstSize = 0;

protected:
  std::vector<RegisteredResource *> &getInputResources();
  std::vector<RegisteredResource *> &getOutputResources();
  inline void setBuilt() { m_passBuilt = true; }

protected:
  void buildDescriptorParamHandle(uint32_t numMultiBuffers);

public:
  RenderGraphPass(EngineContext *context, DescriptorManager *descriptorManager,
                  RegisteredResourceMapper *mapper)
      : m_context(context), m_descriptorManager(descriptorManager),
        m_mapper(mapper) {}
  void setPassDescriptorLayout_base(
      const std::vector<Rhi::RhiDescriptorType> &layout);

  inline void setActiveFrame(uint32_t frame) { m_activeFrame = frame; }
  virtual void addInputResource(RegisteredResource *resource,
                                RenderPassResourceTransition transition);
  virtual void addOutputResource(RegisteredResource *resource,
                                 RenderPassResourceTransition transition);
  virtual uint32_t getRequiredQueueCapability() = 0;
  virtual void withCommandBuffer(CommandBuffer *commandBuffer,
                                 std::function<void()> func);

  void addUniformBuffer_base(RegisteredBufferHandle *buffer, uint32_t position);
  void addCombinedImageSampler(RegisteredImageHandle *image,
                               RegisteredSamplerHandle *sampler,
                               uint32_t position);
  void addStorageBuffer_base(RegisteredBufferHandle *buffer, uint32_t position,
                             Rhi::RhiResourceAccessType access);

  inline void setDefaultNumMultiBuffers(uint32_t x) {
    m_defaultMultibuffers = x;
  }

  inline bool getOperatesOnSwapchain() { return m_operateOnSwapchain; }
  inline bool isBuilt() const { return m_passBuilt; }

  void
  setRecordFunction_base(std::function<void(Rhi::RhiRenderPassContext *)> func);
  void setExecutionFunction_base(
      std::function<void(Rhi::RhiRenderPassContext *)> func);

  virtual void build(uint32_t numMultiBuffers) = 0;
  virtual void record() {}
  virtual void execute();

  inline virtual void setNumBindlessDescriptorSets_base(uint32_t num) {
    m_numBindlessDescriptorSets = num;
  }

  friend class RenderGraph;
};

// Graphics Pass performs rendering operations
class IFRIT_APIDECL GraphicsPass : public RenderGraphPass,
                                   public Rhi::RhiGraphicsPass {
protected:
  // RenderPassAttachment m_depthAttachment;
  // std::vector<RenderPassAttachment> m_colorAttachments;
  Rhi::RhiRenderTargetsFormat m_renderTargetFormat;

  PipelineCache *m_pipelineCache;

  ShaderModule *m_vertexShader = nullptr;
  ShaderModule *m_fragmentShader = nullptr;
  ShaderModule *m_geometryShader = nullptr;
  ShaderModule *m_tessControlShader = nullptr;
  ShaderModule *m_tessEvalShader = nullptr;

  ShaderModule *m_taskShader = nullptr;
  ShaderModule *m_meshShader = nullptr;

  GraphicsPipeline *m_pipeline = nullptr;

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
  Rhi::RhiRasterizerTopology m_topology =
      Rhi::RhiRasterizerTopology::TriangleList;

public:
  GraphicsPass(EngineContext *context, PipelineCache *pipelineCache,
               DescriptorManager *descriptorManager,
               RegisteredResourceMapper *mapper);

  void
  record(Ifrit::GraphicsBackend::VulkanGraphics::RenderTargets *renderTarget);

  void
  setRenderTargetFormat(const Rhi::RhiRenderTargetsFormat &format) override;

  void setVertexShader(Rhi::RhiShader *shader) override;
  void setPixelShader(Rhi::RhiShader *shader) override;
  void setGeometryShader(ShaderModule *shader);
  void setTessControlShader(ShaderModule *shader);
  void setTessEvalShader(ShaderModule *shader);

  void setTaskShader(ShaderModule *shader);
  void setMeshShader(Rhi::RhiShader *shader) override;

  void setRenderArea(uint32_t x, uint32_t y, uint32_t width,
                     uint32_t height) override;
  void setDepthWrite(bool write) override;
  void setColorWrite(const std::vector<uint32_t> &write);
  void setDepthTestEnable(bool enable) override;
  void setDepthCompareOp(Rhi::RhiCompareOp compareOp) override;
  void setRasterizerTopology(Rhi::RhiRasterizerTopology topology) override;

  void setVertexInput(const VertexBufferDescriptor &descriptor,
                      const std::vector<RegisteredBufferHandle *> &buffers);
  void setIndexInput(RegisteredBufferHandle *buffer, VkIndexType type);

  uint32_t getRequiredQueueCapability() override;
  inline void setPushConstSize(uint32_t size) override {
    m_pushConstSize = size;
  }

  virtual void build(uint32_t numMultiBuffers) override;

  // Rhi compat
  inline void setShaderBindingLayout(
      const std::vector<Rhi::RhiDescriptorType> &layout) override {
    setPassDescriptorLayout_base(layout);
  }
  inline void
  addShaderStorageBuffer(Rhi::RhiBuffer *buffer, uint32_t position,
                         Rhi::RhiResourceAccessType access) override {
    using namespace Ifrit::Common::Utility;
    auto buf = checked_cast<SingleBuffer>(buffer);
    auto registeredBuffer = m_mapper->getBufferIndex(buf);
    auto registered = checked_cast<RegisteredBufferHandle>(registeredBuffer);
    addStorageBuffer_base(registered, position, access);
  }
  inline void addUniformBuffer(Rhi::RhiMultiBuffer *buffer, uint32_t position) {
    using namespace Ifrit::Common::Utility;
    auto buf = checked_cast<MultiBuffer>(buffer);
    auto registeredBuffer = m_mapper->getMultiBufferIndex(buf);
    auto registered = checked_cast<RegisteredBufferHandle>(registeredBuffer);
    addUniformBuffer_base(registered, position);
  }
  inline void setExecutionFunction(
      std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    setExecutionFunction_base(func);
  }
  inline void setRecordFunction(
      std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    setRecordFunction_base(func);
  }
  inline void setRecordFunctionPostRenderPass(
      std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    m_recordPostFunction = func;
  }
  inline VkPipelineLayout getPipelineLayout() {
    return m_pipeline->getLayout();
  }

  void run(const Rhi::RhiCommandBuffer *cmd,
           Rhi::RhiRenderTargets *renderTargets, uint32_t frameId) override;

  inline virtual void setNumBindlessDescriptorSets(uint32_t num) override {
    setNumBindlessDescriptorSets_base(num);
  }
};

class IFRIT_APIDECL ComputePass : public RenderGraphPass,
                                  public Rhi::RhiComputePass {
protected:
  ComputePipeline *m_pipeline;
  PipelineCache *m_pipelineCache;

  ShaderModule *m_shaderModule = nullptr;

public:
  ComputePass(EngineContext *context, PipelineCache *pipelineCache,
              DescriptorManager *descriptorManager,
              RegisteredResourceMapper *mapper)
      : RenderGraphPass(context, descriptorManager, mapper),
        m_pipelineCache(pipelineCache) {}

  void record() override;
  uint32_t getRequiredQueueCapability() override;
  void setComputeShader(Rhi::RhiShader *shader) override;
  void build(uint32_t numMultiBuffers) override;
  inline void setPushConstSize(uint32_t size) override {
    m_pushConstSize = size;
  }

  // Rhi compat
  inline void setShaderBindingLayout(
      const std::vector<Rhi::RhiDescriptorType> &layout) override {
    setPassDescriptorLayout_base(layout);
  }
  inline void
  addShaderStorageBuffer(Rhi::RhiBuffer *buffer, uint32_t position,
                         Rhi::RhiResourceAccessType access) override {
    using namespace Ifrit::Common::Utility;
    auto buf = checked_cast<SingleBuffer>(buffer);
    auto registeredBuffer = m_mapper->getBufferIndex(buf);
    auto registered = checked_cast<RegisteredBufferHandle>(registeredBuffer);
    addStorageBuffer_base(registered, position, access);
  }
  inline void addUniformBuffer(Rhi::RhiMultiBuffer *buffer, uint32_t position) {
    using namespace Ifrit::Common::Utility;
    auto buf = checked_cast<MultiBuffer>(buffer);
    auto registeredBuffer = m_mapper->getMultiBufferIndex(buf);
    auto registered = checked_cast<RegisteredBufferHandle>(registeredBuffer);
    addUniformBuffer_base(registered, position);
  }
  inline void setExecutionFunction(
      std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    setExecutionFunction_base(func);
  }
  inline void setRecordFunction(
      std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    setRecordFunction_base(func);
  }
  void run(const Rhi::RhiCommandBuffer *cmd, uint32_t frameId) override;
  inline VkPipelineLayout getPipelineLayout() {
    return m_pipeline->getLayout();
  }
  inline virtual void setNumBindlessDescriptorSets(uint32_t num) override {
    setNumBindlessDescriptorSets_base(num);
  }
};

struct RegisteredResourceGraphState {
  uint32_t indeg;
  uint32_t outdeg;
  uint32_t rwDeps;
};

class IFRIT_APIDECL RenderGraph : public Rhi::RhiPassGraph {
private:
  EngineContext *m_context;
  DescriptorManager *m_descriptorManager;
  std::vector<std::unique_ptr<RenderGraphPass>> m_passes;
  std::vector<std::unique_ptr<RegisteredResource>> m_resources;
  std::unordered_map<RegisteredResource *, uint32_t> m_resourceMap;
  std::unique_ptr<PipelineCache> m_pipelineCache;

  std::unique_ptr<RegisteredResourceMapper> m_mapper;

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
  ComputePass *addComputePass();

  RegisteredBufferHandle *registerBuffer(SingleBuffer *buffer);
  RegisteredBufferHandle *registerBuffer(MultiBuffer *buffer);
  RegisteredImageHandle *registerImage(SingleDeviceImage *image);
  RegisteredSamplerHandle *registerSampler(Sampler *sampler);

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
  void beginFrame();
  void endFrame();

  // for rhi layers
  Queue *getQueue(QueueRequirement req);
};

} // namespace Ifrit::GraphicsBackend::VulkanGraphics