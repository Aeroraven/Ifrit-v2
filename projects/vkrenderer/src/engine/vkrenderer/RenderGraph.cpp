#include <vkrenderer/include/engine/vkrenderer/RenderGraph.h>
#include <vkrenderer/include/utility/Logger.h>

namespace Ifrit::Engine::VkRenderer {
template <typename E>
constexpr typename std::underlying_type<E>::type getUnderlying(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
}
// Class : SwapchainImageResource

IFRIT_APIDECL VkFormat SwapchainImageResource::getFormat() {
  return m_swapchain->getPreferredFormat();
}

IFRIT_APIDECL VkImage SwapchainImageResource::getImage() {
  return m_swapchain->getCurrentImage();
}

IFRIT_APIDECL VkImageView SwapchainImageResource::getImageView() {
  return m_swapchain->getCurrentImageView();
}

// Class : RegisteredResource
IFRIT_APIDECL void RegisteredResource::recordTransition(
    const RenderPassResourceTransition &transition, uint32_t queueFamily) {
  m_currentLayout = transition.m_newLayout;
  m_currentAccess = transition.m_dstAccess;
  m_currentQueueFamily = queueFamily;
}

IFRIT_APIDECL void RegisteredResource::resetState() {
  m_currentLayout = m_originalLayout;
  m_currentAccess = m_originalAccess;
  m_currentQueueFamily = m_originalQueueFamily;
}

// Class : RenderGraphPass

IFRIT_APIDECL void
RenderGraphPass::addInputResource(RegisteredResource *resource,
                                  RenderPassResourceTransition transition) {
  m_inputResources.push_back(resource);
  m_inputTransition.push_back(transition);
}

IFRIT_APIDECL void
RenderGraphPass::addOutputResource(RegisteredResource *resource,
                                   RenderPassResourceTransition transition) {
  m_outputResources.push_back(resource);
  m_outputTransition.push_back(transition);
}

// Class : Pipeline Cache
IFRIT_APIDECL PipelineCache::PipelineCache(EngineContext *context)
    : m_context(context) {}

IFRIT_APIDECL uint64_t
PipelineCache::graphicsPipelineHash(const GraphicsPipelineCreateInfo &ci) {
  uint64_t hash = 0x9e3779b9;
  std::hash<uint64_t> hashFunc;
  for (int i = 0; i < ci.shaderModules.size(); i++) {
    auto pStage = ci.shaderModules[i];
    hash ^= hashFunc(reinterpret_cast<uint64_t>(pStage));
  }
  hash ^= hashFunc(ci.viewportCount);
  hash ^= hashFunc(ci.scissorCount);
  hash ^= hashFunc(ci.stencilAttachmentFormat);
  hash ^= hashFunc(ci.depthAttachmentFormat);
  for (int i = 0; i < ci.colorAttachmentFormats.size(); i++) {
    hash ^= hashFunc(ci.colorAttachmentFormats[i]);
  }
  hash ^= hashFunc(getUnderlying(ci.topology));
  return hash;
}

IFRIT_APIDECL bool
PipelineCache::graphicsPipelineEqual(const GraphicsPipelineCreateInfo &a,
                                     const GraphicsPipelineCreateInfo &b) {
  if (a.shaderModules.size() != b.shaderModules.size())
    return false;
  for (int i = 0; i < a.shaderModules.size(); i++) {
    if (a.shaderModules[i] != b.shaderModules[i])
      return false;
  }
  if (a.viewportCount != b.viewportCount)
    return false;
  if (a.scissorCount != b.scissorCount)
    return false;
  if (a.stencilAttachmentFormat != b.stencilAttachmentFormat)
    return false;
  if (a.depthAttachmentFormat != b.depthAttachmentFormat)
    return false;
  if (a.colorAttachmentFormats.size() != b.colorAttachmentFormats.size())
    return false;
  for (int i = 0; i < a.colorAttachmentFormats.size(); i++) {
    if (a.colorAttachmentFormats[i] != b.colorAttachmentFormats[i])
      return false;
  }
  if (a.topology != b.topology)
    return false;
  return true;
}

IFRIT_APIDECL GraphicsPipeline *
PipelineCache::getGraphicsPipeline(const GraphicsPipelineCreateInfo &ci) {
  uint64_t hash = graphicsPipelineHash(ci);
  for (int i = 0; i < m_graphicsPipelineMap[hash].size(); i++) {
    int index = m_graphicsPipelineMap[hash][i];
    if (graphicsPipelineEqual(ci, m_graphicsPipelineCI[index])) {
      return m_graphicsPipelines[index].get();
    }
  }
  // Otherwise create a new pipeline
  m_graphicsPipelineCI.push_back(ci);
  auto &&p = std::make_unique<GraphicsPipeline>(m_context, ci);
  m_graphicsPipelines.push_back(std::move(p));
  m_graphicsPipelineMap[hash].push_back(m_graphicsPipelines.size() - 1);
  return m_graphicsPipelines.back().get();
}

// Class : RenderPassPass
IFRIT_APIDECL std::vector<RegisteredResource *> &
RenderGraphPass::getInputResources() {
  return m_inputResources;
}

IFRIT_APIDECL std::vector<RegisteredResource *> &
RenderGraphPass::getOutputResources() {
  return m_outputResources;
}

// Class : GraphicsPass
IFRIT_APIDECL GraphicsPass::GraphicsPass(EngineContext *context,
                                         PipelineCache *pipelineCache)
    : RenderGraphPass(context), m_pipelineCache(pipelineCache) {}

IFRIT_APIDECL void GraphicsPass::addColorAttachment(RegisteredImage *image,
                                                    VkAttachmentLoadOp loadOp,
                                                    VkClearValue clearValue) {
  RenderPassAttachment attachment;
  attachment.m_image = image;
  attachment.m_loadOp = loadOp;
  attachment.m_clearValue = clearValue;
  m_colorAttachments.push_back(attachment);

  RenderPassResourceTransition transition;
  transition.m_required = true;
  transition.m_requireExplicitBarrier = true;
  transition.m_allowUndefinedOldLayout =
      (loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
  transition.m_oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  transition.m_newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  transition.m_srcAccess = VK_ACCESS_NONE;
  transition.m_dstAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  addOutputResource(image, transition);

  if (image->getIsSwapchainImage()) {
    m_operateOnSwapchain = true;
  }

  m_colorWrite.push_back(VK_TRUE);
  m_blendEnable.push_back(VK_FALSE);

  VkColorBlendEquationEXT colorBlendEquation{};
  colorBlendEquation.alphaBlendOp = VK_BLEND_OP_ADD;
  colorBlendEquation.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlendEquation.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendEquation.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendEquation.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlendEquation.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  m_blendEquations.push_back(colorBlendEquation);

  m_colorWriteMask.push_back(
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
}

IFRIT_APIDECL void GraphicsPass::setDepthAttachment(RegisteredImage *image,
                                                    VkAttachmentLoadOp loadOp,
                                                    VkClearValue clearValue) {
  m_depthAttachment.m_image = image;
  m_depthAttachment.m_loadOp = loadOp;
  m_depthAttachment.m_clearValue = clearValue;

  RenderPassResourceTransition transition;
  transition.m_required = true;
  transition.m_requireExplicitBarrier = true;
  transition.m_allowUndefinedOldLayout =
      (loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
  transition.m_oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  transition.m_newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  transition.m_srcAccess = VK_ACCESS_NONE;
  transition.m_dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  addOutputResource(image, transition);
}

IFRIT_APIDECL void GraphicsPass::setVertexShader(ShaderModule *shader) {
  m_vertexShader = shader;
}

IFRIT_APIDECL void GraphicsPass::setFragmentShader(ShaderModule *shader) {
  m_fragmentShader = shader;
}

IFRIT_APIDECL void GraphicsPass::setGeometryShader(ShaderModule *shader) {
  m_geometryShader = shader;
}

IFRIT_APIDECL void GraphicsPass::setTessControlShader(ShaderModule *shader) {
  m_tessControlShader = shader;
}

IFRIT_APIDECL void GraphicsPass::setTessEvalShader(ShaderModule *shader) {
  m_tessEvalShader = shader;
}

IFRIT_APIDECL void GraphicsPass::setRecordFunction(
    std::function<void(RenderPassContext *)> executeFunction) {
  m_executeFunction = executeFunction;
}

IFRIT_APIDECL void GraphicsPass::record() {
  m_passContext.m_cmd = m_commandBuffer;

  PipelineBarrier barrierColor(
      m_context, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0);
  // Perform resource transitions for output resources
  for (int i = 0; i < m_colorAttachments.size(); i++) {
    auto &attachment = m_colorAttachments[i];
    auto &transition = m_outputTransition[i];
    attachment.m_image->recordTransition(transition,
                                         m_commandBuffer->getQueueFamily());
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = transition.m_oldLayout;
    barrier.newLayout = transition.m_newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
    // m_commandBuffer->getQueueFamily();
    barrier.image = attachment.m_image->getImage()->getImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = transition.m_srcAccess;
    barrier.dstAccessMask = transition.m_dstAccess;
    barrierColor.addImageMemoryBarrier(barrier);
  }
  m_commandBuffer->pipelineBarrier(barrierColor);

  if (m_depthAttachment.m_image) {
    PipelineBarrier barrierDepth(m_context,
                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0);
    auto &attachment = m_depthAttachment;
    auto &transition = m_outputTransition[m_colorAttachments.size()];
    attachment.m_image->recordTransition(transition,
                                         m_commandBuffer->getQueueFamily());
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = transition.m_oldLayout;
    barrier.newLayout = transition.m_newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = m_commandBuffer->getQueueFamily();
    barrier.image = attachment.m_image->getImage()->getImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = transition.m_srcAccess;
    barrier.dstAccessMask = transition.m_dstAccess;
    barrierDepth.addImageMemoryBarrier(barrier);
    m_commandBuffer->pipelineBarrier(barrierDepth);
  }

  // Specify rendering info
  std::vector<VkRenderingAttachmentInfoKHR> colorAttachmentInfos;
  for (int i = 0; i < m_colorAttachments.size(); i++) {
    VkRenderingAttachmentInfoKHR attachmentInfo{};
    attachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    attachmentInfo.clearValue = m_colorAttachments[i].m_clearValue;
    attachmentInfo.loadOp = m_colorAttachments[i].m_loadOp;
    attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    attachmentInfo.imageView =
        m_colorAttachments[i].m_image->getImage()->getImageView();
    colorAttachmentInfos.push_back(attachmentInfo);
  }
  VkRenderingAttachmentInfoKHR depthAttachmentInfo{};
  if (m_depthAttachment.m_image) {
    depthAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    depthAttachmentInfo.clearValue = m_depthAttachment.m_clearValue;
    depthAttachmentInfo.loadOp = m_depthAttachment.m_loadOp;
    depthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    depthAttachmentInfo.imageView =
        m_depthAttachment.m_image->getImage()->getImageView();
  }

  VkRenderingInfo renderingInfo{};
  renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  renderingInfo.renderArea = m_renderArea;
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount = m_colorAttachments.size();
  renderingInfo.pColorAttachments = colorAttachmentInfos.data();
  renderingInfo.pDepthAttachment =
      m_depthAttachment.m_image ? &depthAttachmentInfo : nullptr;
  
  auto exfun = m_context->getExtensionFunction();
  vkCmdBeginRendering(m_passContext.m_cmd->getCommandBuffer(),
                      &renderingInfo);
  vkCmdBindPipeline(m_passContext.m_cmd->getCommandBuffer(),
                    VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->getPipeline());
  vkCmdSetDepthWriteEnable(m_passContext.m_cmd->getCommandBuffer(),
                           m_depthWrite);
  exfun.p_vkCmdSetColorWriteEnableEXT(m_passContext.m_cmd->getCommandBuffer(), m_colorAttachments.size(),
                              m_colorWrite.data());
  exfun.p_vkCmdSetColorBlendEnableEXT(m_passContext.m_cmd->getCommandBuffer(), 0,
                              m_colorAttachments.size(), m_blendEnable.data());

  exfun.p_vkCmdSetColorBlendEquationEXT(m_passContext.m_cmd->getCommandBuffer(),
                                      0,m_colorAttachments.size(), m_blendEquations.data());

  exfun.p_vkCmdSetColorWriteMaskEXT(m_passContext.m_cmd->getCommandBuffer(), 0,
                            m_colorAttachments.size(), m_colorWriteMask.data());

  exfun.p_vkCmdSetVertexInputEXT(m_passContext.m_cmd->getCommandBuffer(), 0, nullptr, 0,
                         nullptr);
  exfun.p_vkCmdSetLogicOpEnableEXT(m_passContext.m_cmd->getCommandBuffer(),
                           m_logicalOpEnable);
  exfun.p_vkCmdSetLogicOpEXT(m_passContext.m_cmd->getCommandBuffer(), m_logicOp);
  exfun.p_vkCmdSetStencilTestEnable(m_passContext.m_cmd->getCommandBuffer(),
                            m_stencilEnable);
  exfun.p_vkCmdSetStencilOp(m_passContext.m_cmd->getCommandBuffer(),
                    m_stencilOp.faceMask, m_stencilOp.failOp,
                    m_stencilOp.passOp, m_stencilOp.depthFailOp, m_stencilOp.compareOp);

  exfun.p_vkCmdSetDepthBoundsTestEnable(m_passContext.m_cmd->getCommandBuffer(),
                                   m_depthBoundTestEnable);
  exfun.p_vkCmdSetDepthCompareOp(m_passContext.m_cmd->getCommandBuffer(),
                            m_depthCompareOp);
  exfun.p_vkCmdSetDepthTestEnable(m_passContext.m_cmd->getCommandBuffer(),
                             m_depthTestEnable);
  vkCmdSetFrontFace(m_passContext.m_cmd->getCommandBuffer(), m_frontFace);
  vkCmdSetCullMode(m_passContext.m_cmd->getCommandBuffer(), m_cullMode);

  if (m_executeFunction) {
    m_executeFunction(&m_passContext);
  }

  vkCmdEndRendering(m_passContext.m_cmd->getCommandBuffer());
}

IFRIT_APIDECL void GraphicsPass::setRenderArea(uint32_t x, uint32_t y,
                                               uint32_t width,
                                               uint32_t height) {
  m_renderArea.offset = {static_cast<int>(x), static_cast<int>(y)};
  m_renderArea.extent = {width, height};
}

IFRIT_APIDECL void GraphicsPass::setDepthWrite(bool write) {
  m_depthWrite = write;
}

IFRIT_APIDECL void
GraphicsPass::setColorWrite(const std::vector<uint32_t> &write) {
  vkrAssert(write.size() == m_colorAttachments.size(),
            "Num of attachments are not equal");
  for (int i = 0; i < m_colorAttachments.size(); i++) {
    m_colorWrite[i] = (write[i]) ? VK_TRUE : VK_FALSE;
  }
}

IFRIT_APIDECL void GraphicsPass::build() {
  GraphicsPipelineCreateInfo ci;
  ci.shaderModules.push_back(m_vertexShader);
  ci.shaderModules.push_back(m_fragmentShader);
  ci.viewportCount = 1;
  ci.scissorCount = 1;
  ci.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
  if (m_depthAttachment.m_image) {
    ci.depthAttachmentFormat =
        m_depthAttachment.m_image->getImage()->getFormat();
  } else {
    ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
  }
  for (int i = 0; i < m_colorAttachments.size(); i++) {
    auto r = m_colorAttachments[i].m_image->getImage();
    ci.colorAttachmentFormats.push_back(
        m_colorAttachments[i].m_image->getImage()->getFormat());
  }
  ci.topology = RasterizerTopology::TriangleList;
  m_pipeline = m_pipelineCache->getGraphicsPipeline(ci);
  setBuilt();
}

IFRIT_APIDECL void GraphicsPass::withCommandBuffer(CommandBuffer *commandBuffer,
                                                   std::function<void()> func) {
  m_commandBuffer = commandBuffer;
  func();
  m_commandBuffer = nullptr;
}

IFRIT_APIDECL uint32_t GraphicsPass::getRequiredQueueCapability() {
  return VK_QUEUE_GRAPHICS_BIT;
}

// Class: RenderGraph
IFRIT_APIDECL RenderGraph::RenderGraph(EngineContext *context) {
  m_context = context;
  m_pipelineCache = std::make_unique<PipelineCache>(context);
}

IFRIT_APIDECL GraphicsPass *RenderGraph::addGraphicsPass() {
  auto pass = std::make_unique<GraphicsPass>(m_context, m_pipelineCache.get());
  auto ptr = pass.get();
  m_passes.push_back(std::move(pass));
  return ptr;
}

IFRIT_APIDECL RegisteredBuffer *RenderGraph::registerBuffer(Buffer *buffer) {
  auto registeredBuffer = std::make_unique<RegisteredBuffer>(buffer);
  auto ptr = registeredBuffer.get();
  m_resources.push_back(std::move(registeredBuffer));
  m_resourceMap[ptr] = m_resources.size() - 1;
  return ptr;
}

IFRIT_APIDECL RegisteredImage *RenderGraph::registerImage(DeviceImage *image) {
  auto registeredImage = std::make_unique<RegisteredImage>(image);
  auto ptr = registeredImage.get();
  m_resources.push_back(std::move(registeredImage));
  m_resourceMap[ptr] = m_resources.size() - 1;
  return ptr;
}

IFRIT_APIDECL RegisteredSwapchainImage *
RenderGraph::registerSwapchainImage(SwapchainImageResource *image) {
  auto registeredImage = std::make_unique<RegisteredSwapchainImage>(image);
  auto ptr = registeredImage.get();
  m_resources.push_back(std::move(registeredImage));
  m_resourceMap[ptr] = m_resources.size() - 1;
  m_swapchainImageHandle.push_back(ptr);
  return ptr;
}

IFRIT_APIDECL void RenderGraph::build() {
    //Build all passes
  for (int i = 0; i < m_passes.size(); i++) {
    if (!m_passes[i]->isBuilt()) {
      m_passes[i]->build();
    }
  }
  int numSubgraph = 0;
  m_subgraphBelonging.resize(m_resources.size());
  for (int i = 0; i < m_resources.size(); i++) {
    m_subgraphBelonging[i] = UINT32_MAX;
  }

  std::unordered_set<RegisteredResource *> dependencySet;
  for (int i = 0; i < m_swapchainImageHandle.size(); i++) {
    dependencySet.insert(m_swapchainImageHandle[i]);
  }
  uint32_t assignedPass = 0;
  while (true) {
    for (int i = m_passes.size() - 1; i >= 0; i--) {
      auto pass = m_passes[i].get();
      if (m_subgraphBelonging[i] != UINT32_MAX)
        continue;
      auto &inputResources = pass->getInputResources();
      auto &outputResources = pass->getOutputResources();

      // If input or output contains object in dependency set
      // then add all input and output resources to dependency set
      bool hasDependency = false;
      for (int j = 0; j < inputResources.size(); j++) {
        if (dependencySet.find(inputResources[j]) != dependencySet.end()) {
          hasDependency = true;
          break;
        }
      }
      for (int j = 0; j < outputResources.size(); j++) {
        if (dependencySet.find(outputResources[j]) != dependencySet.end()) {
          hasDependency = true;
          break;
        }
      }
      if (hasDependency) {
        for (int j = 0; j < inputResources.size(); j++) {
          dependencySet.insert(inputResources[j]);
        }
        for (int j = 0; j < outputResources.size(); j++) {
          dependencySet.insert(outputResources[j]);
        }
        m_subgraphBelonging[i] = numSubgraph;
        assignedPass++;
        // m_subgraphs[numSubgraph].push_back(i);
      }
    }
    
    if (assignedPass != 0)
      numSubgraph++;
    if (assignedPass == m_passes.size())
      break;
    // Otherwise, find an unassigned pass that has no dependency
    // and add it to the dependency set
    dependencySet.clear();
    for (int i = 0; i < m_passes.size(); i++) {
      if (m_subgraphBelonging[i] == UINT32_MAX) {
        auto pass = m_passes[i].get();
        auto &inputResources = pass->getInputResources();
        auto &outputResources = pass->getOutputResources();
        for (int j = 0; j < inputResources.size(); j++) {
          dependencySet.insert(inputResources[j]);
        }
        for (int j = 0; j < outputResources.size(); j++) {
          dependencySet.insert(outputResources[j]);
        }
      }
    }
  }
  m_assignedQueues.resize(numSubgraph);
  // add to m_subgraphs
  m_subgraphs.resize(numSubgraph);
  m_subGraphOperatesOnSwapchain.resize(numSubgraph);
  for (int i = 0; i < m_passes.size(); i++) {
    m_subgraphs[m_subgraphBelonging[i]].push_back(i);
  }

  // stat
  printf("Number of subgraphs: %d\n", numSubgraph);
  for (int i = 0; i < m_subgraphs.size(); i++) {
    printf("Subgraph %d: ", i);
    for (int j = 0; j < m_subgraphs[i].size(); j++) {
      printf("%d ", m_subgraphs[i][j]);
    }
    printf("\n");
  }
}

IFRIT_APIDECL std::vector<std::vector<uint32_t>> &RenderGraph::getSubgraphs() {
  return m_subgraphs;
}

IFRIT_APIDECL void RenderGraph::resizeCommandBuffers(uint32_t size) {
  m_commandBuffers.resize(size);
}

// Class : RenderGraphExecutor
IFRIT_APIDECL void
RenderGraphExecutor::setQueues(const std::vector<Queue *> &queues) {
  m_queues = queues;
}

IFRIT_APIDECL void RenderGraphExecutor::compileGraph(RenderGraph *graph) {
  m_graph = graph;
  auto subgraphs = graph->getSubgraphs();
  for (int i = 0; i < m_queues.size(); i++) {
    auto queue = m_queues[i];
    auto commandPool =
        std::make_unique<CommandPool>(m_context, queue->getQueueFamily());
    m_graph->m_commandPools.push_back(std::move(commandPool));
  }

  std::vector<uint32_t> queueAssignedTimes(m_queues.size(), 0);
  for (int i = 0; i < subgraphs.size(); i++) {
    uint32_t requiredCapability = 0;
    bool operateOnSwapchain = false;
    for (int j = 0; j < subgraphs[i].size(); j++) {
      auto pass = m_graph->m_passes[subgraphs[i][j]].get();
      requiredCapability |= pass->getRequiredQueueCapability();
      operateOnSwapchain |= pass->getOperatesOnSwapchain();
    }
    m_graph->m_subGraphOperatesOnSwapchain[i] = operateOnSwapchain;

    // Find a queue that has the required capability,
    // with the least number of assigned times
    uint32_t queueIndex = UINT32_MAX;
    for (int j = 0; j < m_queues.size(); j++) {
      if ((m_queues[j]->getCapability() & requiredCapability) ==
          requiredCapability) {
        if (requiredCapability | VK_QUEUE_GRAPHICS_BIT) {
          queueIndex = j;
          break;
        }
        if (queueAssignedTimes[j] < queueAssignedTimes[queueIndex]) {
          queueIndex = j;
        }
      }
    }
    if (queueIndex == UINT32_MAX) {
      vkrError("No queue found for subgraph");
    }
    queueAssignedTimes[queueIndex]++;
    m_graph->m_assignedQueues[i] = m_queues[queueIndex];
  }
  m_graph->m_comiled = true;
}

IFRIT_APIDECL void RenderGraphExecutor::runGraph(RenderGraph *graph) {
  if (!graph->m_comiled) {
    vkrError("Graph not compiled");
  }
  auto subgraphs = graph->getSubgraphs();
  // for each subgraph, create a command buffer
  for (int i = 0; i < subgraphs.size(); i++) {
    auto queue = m_graph->m_assignedQueues[i];
    auto commandBuffer = queue->beginRecording();
    for (int j = 0; j < subgraphs[i].size(); j++) {
      auto pass = m_graph->m_passes[subgraphs[i][j]].get();
      pass->withCommandBuffer(commandBuffer,
                              [&pass]() { pass->record(); });
    }
    //TODO: Semaphores & Sync
    if (m_graph->m_subGraphOperatesOnSwapchain[i]) {
      // Make swapchain image in layout present
      PipelineBarrier barrier(m_context,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0);
      VkImageMemoryBarrier imageBarrier{};
      imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      imageBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      imageBarrier.srcQueueFamilyIndex = queue->getQueueFamily();
      imageBarrier.dstQueueFamilyIndex = m_swapchain->getQueueFamily();
      imageBarrier.image = m_swapchain->getCurrentImage();
      imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageBarrier.subresourceRange.levelCount = 1;
      imageBarrier.subresourceRange.layerCount = 1;
      imageBarrier.srcAccessMask = 0;
      imageBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
      barrier.addImageMemoryBarrier(imageBarrier);
      commandBuffer->pipelineBarrier(barrier);

      TimelineSemaphoreWait wait;
      wait.m_semaphore = m_swapchain->getImageAvailableSemaphoreCurrentFrame();
      wait.m_value = 0;
      wait.m_waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

      auto renderFinishSemaphore =
          m_swapchain->getRenderingFinishSemaphoreCurrentFrame();

      auto fence = m_swapchain->getCurrentFrameFence();
      queue->submitCommand({wait}, fence, renderFinishSemaphore);
    } else {
      queue->submitCommand({}, nullptr);
    }
  }
}

IFRIT_APIDECL SwapchainImageResource *
RenderGraphExecutor::getSwapchainImageResource() {
  return m_swapchainImageResource.get();
}

} // namespace Ifrit::Engine::VkRenderer
