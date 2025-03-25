
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

  Universal = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT
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
  SwapchainImageResource(Swapchain *swapchain) : SingleDeviceImage(nullptr), m_swapchain(swapchain) {
    m_isSwapchainImage = true;
  }
  virtual ~SwapchainImageResource() {}
  virtual VkFormat getFormat() const override;
  virtual VkImage getImage() const override;
  virtual VkImageView getImageView() override;
  virtual VkImageView getImageViewMipLayer(u32 mip, u32 layer, u32 mipRange, u32 layerRange) override;
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
  u32 m_originalQueueFamily = 0xFFFFFFFF;
  VkImageLayout m_originalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkAccessFlags m_originalAccess = VK_ACCESS_NONE;

  u32 m_currentQueueFamily = 0xFFFFFFFF;
  VkImageLayout m_currentLayout;
  VkAccessFlags m_currentAccess;

  bool m_isSwapchainImage = false;

public:
  void resetState();
  inline bool getIsSwapchainImage() { return m_isSwapchainImage; }
  void recordTransition(const RenderPassResourceTransition &transition, u32 queueFamily);
  virtual ~RegisteredResource() {}
};

class IFRIT_APIDECL RegisteredBufferHandle : public RegisteredResource {
private:
  bool m_isMultipleBuffered = false;
  SingleBuffer *m_buffer;
  MultiBuffer *m_multiBuffer;

public:
  RegisteredBufferHandle(SingleBuffer *buffer) : m_buffer(buffer), m_isMultipleBuffered(false) {}
  RegisteredBufferHandle(MultiBuffer *buffer) : m_multiBuffer(buffer), m_isMultipleBuffered(true) {}
  inline SingleBuffer *getBuffer(u32 frame) {
    return m_isMultipleBuffered ? m_multiBuffer->getBuffer(frame) : m_buffer;
  }
  inline bool isMultipleBuffered() { return m_isMultipleBuffered; }
  inline u32 getNumBuffers() { return m_isMultipleBuffered ? m_multiBuffer->getBufferCount() : 1; }
  virtual ~RegisteredBufferHandle() {}
};

class IFRIT_APIDECL RegisteredImageHandle : public RegisteredResource {
private:
  SingleDeviceImage *m_image;

public:
  RegisteredImageHandle(SingleDeviceImage *image) : m_image(image) {}
  inline virtual SingleDeviceImage *getImage(u32 x) { return m_image; }
  inline u32 getNumBuffers() { return 1; }
  virtual ~RegisteredImageHandle() {}
};

class IFRIT_APIDECL RegisteredSwapchainImage : public RegisteredImageHandle {
private:
  SwapchainImageResource *m_image;

public:
  RegisteredSwapchainImage(SwapchainImageResource *image) : RegisteredImageHandle(image), m_image(image) {
    m_isSwapchainImage = true;
  }
  inline virtual SwapchainImageResource *getImage(u32 x) override { return m_image; }
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

class IFRIT_APIDECL RegisteredResourceMapper : public Ifrit::Common::Utility::NonCopyable {
private:
  std::vector<std::unique_ptr<RegisteredResource>> m_resources;
  std::unordered_map<SingleBuffer *, u32> m_bufferMap;
  std::unordered_map<SingleDeviceImage *, u32> m_imageMap;
  std::unordered_map<MultiBuffer *, u32> m_multiBufferMap;
  std::unordered_map<Sampler *, u32> m_samplerMap;
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
      m_bufferMap[buffer] = Ifrit::Common::Utility::size_cast<u32>(m_resources.size()) - 1;
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
        auto registeredImage = std::make_unique<RegisteredSwapchainImage>(swapchainImage);
        auto ptr = registeredImage.get();
        m_resources.push_back(std::move(registeredImage));
        m_imageMap[image] = size_cast<u32>(m_resources.size()) - 1;
        return ptr;
      }
      auto registeredImage = std::make_unique<RegisteredImageHandle>(image);
      auto ptr = registeredImage.get();
      m_resources.push_back(std::move(registeredImage));
      m_imageMap[image] = Ifrit::Common::Utility::size_cast<u32>(m_resources.size()) - 1;
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
      m_multiBufferMap[buffer] = Ifrit::Common::Utility::size_cast<u32>(m_resources.size()) - 1;
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

  std::unordered_map<u32, std::vector<u32>> m_resourceDescriptorHandle;
  std::vector<Rhi::RhiDescriptorType> m_passDescriptorLayout;
  std::vector<DescriptorBindRange> m_descriptorBindRange;
  std::function<void(Rhi::RhiRenderPassContext *)> m_recordFunction = nullptr;
  std::function<void(Rhi::RhiRenderPassContext *)> m_recordPostFunction = nullptr;
  std::function<void(Rhi::RhiRenderPassContext *)> m_executeFunction = nullptr;
  const CommandBuffer *m_commandBuffer = nullptr;

  // SSBOs
  std::vector<RegisteredBufferHandle *> m_ssbos;
  std::vector<Rhi::RhiResourceAccessType> m_ssboAccess;

  u32 m_activeFrame = 0;
  u32 m_defaultMultibuffers = UINT_MAX;

  // 2x
  u32 m_numBindlessDescriptorSets = 0;
  u32 m_pushConstSize = 0;

protected:
  std::vector<RegisteredResource *> &getInputResources();
  std::vector<RegisteredResource *> &getOutputResources();
  inline void setBuilt() { m_passBuilt = true; }

protected:
  void buildDescriptorParamHandle(u32 numMultiBuffers);

public:
  RenderGraphPass(EngineContext *context, DescriptorManager *descriptorManager, RegisteredResourceMapper *mapper)
      : m_context(context), m_descriptorManager(descriptorManager), m_mapper(mapper) {}
  void setPassDescriptorLayout_base(const std::vector<Rhi::RhiDescriptorType> &layout);

  inline void setActiveFrame(u32 frame) { m_activeFrame = frame; }
  virtual void addInputResource(RegisteredResource *resource, RenderPassResourceTransition transition);
  virtual void addOutputResource(RegisteredResource *resource, RenderPassResourceTransition transition);
  virtual u32 getRequiredQueueCapability() = 0;
  virtual void withCommandBuffer(CommandBuffer *commandBuffer, std::function<void()> func);

  void addUniformBuffer_base(RegisteredBufferHandle *buffer, u32 position);
  void addCombinedImageSampler(RegisteredImageHandle *image, RegisteredSamplerHandle *sampler, u32 position);
  void addStorageBuffer_base(RegisteredBufferHandle *buffer, u32 position, Rhi::RhiResourceAccessType access);

  inline void setDefaultNumMultiBuffers(u32 x) { m_defaultMultibuffers = x; }

  inline bool getOperatesOnSwapchain() { return m_operateOnSwapchain; }
  inline bool isBuilt() const { return m_passBuilt; }

  void setRecordFunction_base(std::function<void(Rhi::RhiRenderPassContext *)> func);
  void setExecutionFunction_base(std::function<void(Rhi::RhiRenderPassContext *)> func);

  virtual void build(u32 numMultiBuffers) = 0;
  virtual void record() {}
  virtual void execute();

  inline virtual void setNumBindlessDescriptorSets_base(u32 num) { m_numBindlessDescriptorSets = num; }

  friend class RenderGraph;
};

// Graphics Pass performs rendering operations
class IFRIT_APIDECL GraphicsPass : public RenderGraphPass, public Rhi::RhiGraphicsPass {
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
  Rhi::RhiRasterizerTopology m_topology = Rhi::RhiRasterizerTopology::TriangleList;

public:
  GraphicsPass(EngineContext *context, PipelineCache *pipelineCache, DescriptorManager *descriptorManager,
               RegisteredResourceMapper *mapper);

  void record(Ifrit::GraphicsBackend::VulkanGraphics::RenderTargets *renderTarget);

  void setRenderTargetFormat(const Rhi::RhiRenderTargetsFormat &format) override;

  void setVertexShader(Rhi::RhiShader *shader) override;
  void setPixelShader(Rhi::RhiShader *shader) override;
  void setGeometryShader(ShaderModule *shader);
  void setTessControlShader(ShaderModule *shader);
  void setTessEvalShader(ShaderModule *shader);

  void setTaskShader(Rhi::RhiShader *shader) override;
  void setMeshShader(Rhi::RhiShader *shader) override;

  void setRenderArea(u32 x, u32 y, u32 width, u32 height) override;
  void setDepthWrite(bool write) override;
  void setColorWrite(const std::vector<u32> &write);
  void setDepthTestEnable(bool enable) override;
  void setDepthCompareOp(Rhi::RhiCompareOp compareOp) override;
  void setRasterizerTopology(Rhi::RhiRasterizerTopology topology) override;

  void setVertexInput(const VertexBufferDescriptor &descriptor, const std::vector<RegisteredBufferHandle *> &buffers);
  void setIndexInput(RegisteredBufferHandle *buffer, VkIndexType type);

  u32 getRequiredQueueCapability() override;
  inline void setPushConstSize(u32 size) override { m_pushConstSize = size; }

  virtual void build(u32 numMultiBuffers) override;

  // Rhi compat
  inline void setShaderBindingLayout(const std::vector<Rhi::RhiDescriptorType> &layout) override {
    setPassDescriptorLayout_base(layout);
  }
  inline void addShaderStorageBuffer(Rhi::RhiBuffer *buffer, u32 position, Rhi::RhiResourceAccessType access) override {
    using namespace Ifrit::Common::Utility;
    auto buf = checked_cast<SingleBuffer>(buffer);
    auto registeredBuffer = m_mapper->getBufferIndex(buf);
    auto registered = checked_cast<RegisteredBufferHandle>(registeredBuffer);
    addStorageBuffer_base(registered, position, access);
  }
  inline void addUniformBuffer(Rhi::RhiMultiBuffer *buffer, u32 position) {
    using namespace Ifrit::Common::Utility;
    auto buf = checked_cast<MultiBuffer>(buffer);
    auto registeredBuffer = m_mapper->getMultiBufferIndex(buf);
    auto registered = checked_cast<RegisteredBufferHandle>(registeredBuffer);
    addUniformBuffer_base(registered, position);
  }
  inline void setExecutionFunction(std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    setExecutionFunction_base(func);
  }
  inline void setRecordFunction(std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    setRecordFunction_base(func);
  }
  inline void setRecordFunctionPostRenderPass(std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    m_recordPostFunction = func;
  }
  inline VkPipelineLayout getPipelineLayout() { return m_pipeline->getLayout(); }

  void run(const Rhi::RhiCommandList *cmd, Rhi::RhiRenderTargets *renderTargets, u32 frameId) override;

  inline virtual void setNumBindlessDescriptorSets(u32 num) override { setNumBindlessDescriptorSets_base(num); }
};

class IFRIT_APIDECL ComputePass : public RenderGraphPass, public Rhi::RhiComputePass {
protected:
  ComputePipeline *m_pipeline;
  PipelineCache *m_pipelineCache;

  ShaderModule *m_shaderModule = nullptr;

public:
  ComputePass(EngineContext *context, PipelineCache *pipelineCache, DescriptorManager *descriptorManager,
              RegisteredResourceMapper *mapper)
      : RenderGraphPass(context, descriptorManager, mapper), m_pipelineCache(pipelineCache) {}

  void record() override;
  u32 getRequiredQueueCapability() override;
  void setComputeShader(Rhi::RhiShader *shader) override;
  void build(u32 numMultiBuffers) override;
  inline void setPushConstSize(u32 size) override { m_pushConstSize = size; }

  // Rhi compat
  inline void setShaderBindingLayout(const std::vector<Rhi::RhiDescriptorType> &layout) override {
    setPassDescriptorLayout_base(layout);
  }
  inline void addShaderStorageBuffer(Rhi::RhiBuffer *buffer, u32 position, Rhi::RhiResourceAccessType access) override {
    using namespace Ifrit::Common::Utility;
    auto buf = checked_cast<SingleBuffer>(buffer);
    auto registeredBuffer = m_mapper->getBufferIndex(buf);
    auto registered = checked_cast<RegisteredBufferHandle>(registeredBuffer);
    addStorageBuffer_base(registered, position, access);
  }
  inline void addUniformBuffer(Rhi::RhiMultiBuffer *buffer, u32 position) {
    using namespace Ifrit::Common::Utility;
    auto buf = checked_cast<MultiBuffer>(buffer);
    auto registeredBuffer = m_mapper->getMultiBufferIndex(buf);
    auto registered = checked_cast<RegisteredBufferHandle>(registeredBuffer);
    addUniformBuffer_base(registered, position);
  }
  inline void setExecutionFunction(std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    setExecutionFunction_base(func);
  }
  inline void setRecordFunction(std::function<void(Rhi::RhiRenderPassContext *)> func) override {
    setRecordFunction_base(func);
  }
  void run(const Rhi::RhiCommandList *cmd, u32 frameId) override;
  inline VkPipelineLayout getPipelineLayout() { return m_pipeline->getLayout(); }
  inline virtual void setNumBindlessDescriptorSets(u32 num) override { setNumBindlessDescriptorSets_base(num); }
};

struct RegisteredResourceGraphState {
  u32 indeg;
  u32 outdeg;
  u32 rwDeps;
};

class IFRIT_APIDECL RenderGraph {
private:
  EngineContext *m_context;
  DescriptorManager *m_descriptorManager;
  std::vector<std::unique_ptr<RenderGraphPass>> m_passes;
  std::vector<std::unique_ptr<RegisteredResource>> m_resources;
  std::unordered_map<RegisteredResource *, u32> m_resourceMap;
  std::unique_ptr<PipelineCache> m_pipelineCache;

  std::unique_ptr<RegisteredResourceMapper> m_mapper;

  // Graph States
  std::vector<RegisteredResourceGraphState> m_resourceGraphState;
  std::vector<RegisteredResource *> m_swapchainImageHandle;
  std::vector<u32> m_subgraphBelonging;
  std::vector<std::vector<u32>> m_subgraphs;

  std::vector<int> m_subGraphOperatesOnSwapchain;

  // Compilation
  bool m_comiled = false;
  std::vector<std::unique_ptr<CommandPool>> m_commandPools;
  std::vector<CommandSubmissionList> m_submissionLists;
  std::vector<std::unique_ptr<CommandBuffer>> m_commandBuffers;
  std::vector<Queue *> m_assignedQueues;

protected:
  std::vector<std::vector<u32>> &getSubgraphs();
  void resizeCommandBuffers(u32 size);
  void buildPassDescriptors(u32 numMultiBuffer);

public:
  RenderGraph(EngineContext *context, DescriptorManager *descriptorManager);

  GraphicsPass *addGraphicsPass();
  ComputePass *addComputePass();

  RegisteredBufferHandle *registerBuffer(SingleBuffer *buffer);
  RegisteredBufferHandle *registerBuffer(MultiBuffer *buffer);
  RegisteredImageHandle *registerImage(SingleDeviceImage *image);
  RegisteredSamplerHandle *registerSampler(Sampler *sampler);

  void build(u32 numMultiBuffers);
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
  void compileGraph(RenderGraph *graph, u32 numMultiBuffers);

public:
  CommandExecutor(EngineContext *context, Swapchain *swapchain, DescriptorManager *descriptorManager,
                  ResourceManager *resourceManager)
      : m_context(context), m_swapchain(swapchain), m_descriptorManager(descriptorManager),
        m_resourceManager(resourceManager) {
    m_swapchainImageResource = std::make_unique<SwapchainImageResource>(swapchain);
    m_resourceManager->setDefaultCopies(swapchain->getNumBackbuffers());
  }
  CommandExecutor(const CommandExecutor &p) = delete;
  CommandExecutor &operator=(const CommandExecutor &p) = delete;

  RenderGraph *createRenderGraph();
  void setQueues(bool reqPresentQueue, int numGraphics, int numCompute, int numTransfer);
  void syncMultiBufferStateWithSwapchain();
  void runRenderGraph(RenderGraph *graph);
  void runImmidiateCommand(std::function<void(CommandBuffer *)> func, QueueRequirement req);
  SwapchainImageResource *getSwapchainImageResource();
  void beginFrame();
  void endFrame();

  // for rhi layers
  Queue *getQueue(QueueRequirement req);
};

} // namespace Ifrit::GraphicsBackend::VulkanGraphics