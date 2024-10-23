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

IFRIT_APIDECL void RenderGraphPass::setPassDescriptorLayout(
    const std::vector<DescriptorType> &layout) {
  m_passDescriptorLayout = layout;
}

IFRIT_APIDECL void
RenderGraphPass::addUniformBuffer(RegisteredBufferHandle *buffer,
                                  uint32_t position) {
  auto numCopies = buffer->getNumBuffers();
  m_resourceDescriptorHandle[position].resize(numCopies);
  for (int i = 0; i < numCopies; i++) {
    auto v = m_descriptorManager->registerUniformBuffer(buffer->getBuffer(i));
    m_resourceDescriptorHandle[position][i] = v;
  }
  // Add as input
  RenderPassResourceTransition transition{};
  transition.m_required = false;
  addInputResource(buffer, transition);
}

IFRIT_APIDECL
void RenderGraphPass::addStorageBuffer(RegisteredBufferHandle *buffer,
                                       uint32_t position,
                                       ResourceAccessType access) {
  auto numCopies = buffer->getNumBuffers();
  m_resourceDescriptorHandle[position].resize(numCopies);
  for (int i = 0; i < numCopies; i++) {
    auto v = m_descriptorManager->registerStorageBuffer(buffer->getBuffer(i));
    m_resourceDescriptorHandle[position][i] = v;
  }
  // Add as input
  RenderPassResourceTransition transition{};
  transition.m_required = false;
  addInputResource(buffer, transition);
  m_ssbos.push_back(buffer);
  m_ssboAccess.push_back(access);
}

IFRIT_APIDECL void
RenderGraphPass::addCombinedImageSampler(RegisteredImageHandle *image,
                                         RegisteredSamplerHandle *sampler,
                                         uint32_t position) {
  auto numCopies = image->getNumBuffers();
  m_resourceDescriptorHandle[position].resize(numCopies);
  for (int i = 0; i < numCopies; i++) {
    auto v = m_descriptorManager->registerCombinedImageSampler(
        image->getImage(i), sampler->getSampler());
    m_resourceDescriptorHandle[position][i] = v;
  }
  // Add as input
  RenderPassResourceTransition transition{};
  transition.m_required = false;
  addInputResource(image, transition);
  addInputResource(sampler, transition);
}

IFRIT_APIDECL void
RenderGraphPass::buildDescriptorParamHandle(uint32_t numMultiBuffers) {
  if (m_passDescriptorLayout.size() == 0)
    return;
  for (auto &[k, v] : m_resourceDescriptorHandle) {
    if (v.size() != 1 && v.size() != numMultiBuffers) {
      vkrError("Descriptor handle size mismatch");
    }
  }
  m_descriptorBindRange.resize(numMultiBuffers);
  for (int T = 0; T < numMultiBuffers; T++) {
    std::vector<uint32_t> descriptorHandles;
    for (int i = 0; i < m_passDescriptorLayout.size(); i++) {
      auto type = m_passDescriptorLayout[i];
      if (m_resourceDescriptorHandle.count(i) == 0) {
        vkrError("Descriptor handle not found for position");
      }
      auto &handleR = m_resourceDescriptorHandle[i];
      if (handleR.size() != 1 && handleR.size() != numMultiBuffers) {
        vkrLog("Inconsistent double buffers");
      }
      auto handle = i >= handleR.size() ? handleR.back() : handleR[T];
      descriptorHandles.push_back(handle);
    }
    m_descriptorBindRange[T] =
        m_descriptorManager->registerBindlessParameterRaw(
            reinterpret_cast<const char *>(descriptorHandles.data()),
            descriptorHandles.size() * sizeof(uint32_t));
  }
}

IFRIT_APIDECL void RenderGraphPass::setRecordFunction(
    std::function<void(RenderPassContext *)> executeFunction) {
  m_recordFunction = executeFunction;
}

IFRIT_APIDECL void RenderGraphPass::setExecutionFunction(
    std::function<void(RenderPassContext *)> func) {
  m_executeFunction = func;
}

IFRIT_APIDECL void RenderGraphPass::execute() {
  if (m_executeFunction) {
    m_executeFunction(&m_passContext);
  }
}

IFRIT_APIDECL std::vector<RegisteredResource *> &
RenderGraphPass::getInputResources() {
  return m_inputResources;
}

IFRIT_APIDECL std::vector<RegisteredResource *> &
RenderGraphPass::getOutputResources() {
  return m_outputResources;
}

IFRIT_APIDECL void
RenderGraphPass::withCommandBuffer(CommandBuffer *commandBuffer,
                                   std::function<void()> func) {
  m_commandBuffer = commandBuffer;
  func();
  m_commandBuffer = nullptr;
}

// Class : GraphicsPass
IFRIT_APIDECL GraphicsPass::GraphicsPass(EngineContext *context,
                                         PipelineCache *pipelineCache,
                                         DescriptorManager *descriptorManager)
    : RenderGraphPass(context, descriptorManager),
      m_pipelineCache(pipelineCache) {}

IFRIT_APIDECL void
GraphicsPass::addColorAttachment(RegisteredImageHandle *image,
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

IFRIT_APIDECL void
GraphicsPass::setDepthAttachment(RegisteredImageHandle *image,
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
  transition.m_dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
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

IFRIT_APIDECL void GraphicsPass::record() {
  m_passContext.m_cmd = m_commandBuffer;
  m_passContext.m_frame = m_activeFrame;

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
    barrier.image = attachment.m_image->getImage(m_activeFrame)->getImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = transition.m_srcAccess;
    barrier.dstAccessMask = transition.m_dstAccess;
    barrierColor.addImageMemoryBarrier(barrier);
  }
  m_commandBuffer->pipelineBarrier(barrierColor);

  if (m_depthAttachment.m_image) {
    PipelineBarrier barrierDepth(m_context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0);
    auto &attachment = m_depthAttachment;
    auto &transition = m_outputTransition[m_colorAttachments.size()];
    attachment.m_image->recordTransition(transition,
                                         m_commandBuffer->getQueueFamily());
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = transition.m_oldLayout;
    barrier.newLayout = transition.m_newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = attachment.m_image->getImage(m_activeFrame)->getImage();
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
        m_colorAttachments[i].m_image->getImage(m_activeFrame)->getImageView();
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
        m_depthAttachment.m_image->getImage(m_activeFrame)->getImageView();
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
  vkCmdBeginRendering(m_passContext.m_cmd->getCommandBuffer(), &renderingInfo);
  vkCmdBindPipeline(m_passContext.m_cmd->getCommandBuffer(),
                    VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->getPipeline());
  vkCmdSetDepthWriteEnable(m_passContext.m_cmd->getCommandBuffer(),
                           m_depthWrite);
  exfun.p_vkCmdSetColorWriteEnableEXT(m_passContext.m_cmd->getCommandBuffer(),
                                      m_colorAttachments.size(),
                                      m_colorWrite.data());
  exfun.p_vkCmdSetColorBlendEnableEXT(m_passContext.m_cmd->getCommandBuffer(),
                                      0, m_colorAttachments.size(),
                                      m_blendEnable.data());

  exfun.p_vkCmdSetColorBlendEquationEXT(m_passContext.m_cmd->getCommandBuffer(),
                                        0, m_colorAttachments.size(),
                                        m_blendEquations.data());

  exfun.p_vkCmdSetColorWriteMaskEXT(m_passContext.m_cmd->getCommandBuffer(), 0,
                                    m_colorAttachments.size(),
                                    m_colorWriteMask.data());

  exfun.p_vkCmdSetVertexInputEXT(m_passContext.m_cmd->getCommandBuffer(),
                                 m_vertexBufferDescriptor.m_bindings.size(),
                                 m_vertexBufferDescriptor.m_bindings.data(),
                                 m_vertexBufferDescriptor.m_attributes.size(),
                                 m_vertexBufferDescriptor.m_attributes.data());

  std::vector<VkBuffer> vxbuffers;
  std::vector<VkDeviceSize> offsets;
  for (int i = 0; i < m_vertexBuffers.size(); i++) {
    vxbuffers.push_back(
        m_vertexBuffers[i]->getBuffer(m_passContext.m_frame)->getBuffer());
    offsets.push_back(0);
  }
  if (vxbuffers.size() > 0) {
    vkCmdBindVertexBuffers(m_passContext.m_cmd->getCommandBuffer(), 0,
                           m_vertexBuffers.size(), vxbuffers.data(),
                           offsets.data());
  }

  if (m_indexBuffer) {
    vkCmdBindIndexBuffer(
        m_passContext.m_cmd->getCommandBuffer(),
        m_indexBuffer->getBuffer(m_passContext.m_frame)->getBuffer(), 0,
        m_indexType);
  }

  if (m_passDescriptorLayout.size() > 0) {
    auto bindlessSet = m_descriptorManager->getBindlessSet();
    auto dynamicSet = m_descriptorManager->getParameterDescriptorSet(
        m_descriptorBindRange[m_passContext.m_frame].rangeId);
    auto dynamicRange =
        m_descriptorBindRange[m_passContext.m_frame].rangeOffset;
    vkCmdBindDescriptorSets(m_passContext.m_cmd->getCommandBuffer(),
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline->getLayout(), 0, 1, &bindlessSet, 0,
                            nullptr);
    vkCmdBindDescriptorSets(m_passContext.m_cmd->getCommandBuffer(),
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline->getLayout(), 1, 1, &dynamicSet, 1,
                            &dynamicRange);
  }

  exfun.p_vkCmdSetLogicOpEnableEXT(m_passContext.m_cmd->getCommandBuffer(),
                                   m_logicalOpEnable);
  exfun.p_vkCmdSetLogicOpEXT(m_passContext.m_cmd->getCommandBuffer(),
                             m_logicOp);
  exfun.p_vkCmdSetStencilTestEnable(m_passContext.m_cmd->getCommandBuffer(),
                                    m_stencilEnable);
  exfun.p_vkCmdSetStencilOp(m_passContext.m_cmd->getCommandBuffer(),
                            m_stencilOp.faceMask, m_stencilOp.failOp,
                            m_stencilOp.passOp, m_stencilOp.depthFailOp,
                            m_stencilOp.compareOp);

  exfun.p_vkCmdSetDepthBoundsTestEnable(m_passContext.m_cmd->getCommandBuffer(),
                                        m_depthBoundTestEnable);
  exfun.p_vkCmdSetDepthCompareOp(m_passContext.m_cmd->getCommandBuffer(),
                                 m_depthCompareOp);
  exfun.p_vkCmdSetDepthTestEnable(m_passContext.m_cmd->getCommandBuffer(),
                                  m_depthTestEnable);
  vkCmdSetFrontFace(m_passContext.m_cmd->getCommandBuffer(), m_frontFace);
  vkCmdSetCullMode(m_passContext.m_cmd->getCommandBuffer(), m_cullMode);

  if (m_recordFunction) {
    m_recordFunction(&m_passContext);
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

IFRIT_APIDECL void GraphicsPass::setDepthTestEnable(bool enable) {
  m_depthTestEnable = enable;
}
IFRIT_APIDECL void GraphicsPass::setDepthCompareOp(VkCompareOp compareOp) {
  m_depthCompareOp = compareOp;
}
IFRIT_APIDECL void
GraphicsPass::setRasterizerTopology(RasterizerTopology topology) {
  m_topology = topology;
}

IFRIT_APIDECL void
GraphicsPass::setColorWrite(const std::vector<uint32_t> &write) {
  vkrAssert(write.size() == m_colorAttachments.size(),
            "Num of attachments are not equal");
  for (int i = 0; i < m_colorAttachments.size(); i++) {
    m_colorWrite[i] = (write[i]) ? VK_TRUE : VK_FALSE;
  }
}

IFRIT_APIDECL void GraphicsPass::setVertexInput(
    const VertexBufferDescriptor &descriptor,
    const std::vector<RegisteredBufferHandle *> &buffers) {
  m_vertexBufferDescriptor = descriptor;
  m_vertexBuffers = buffers;
  if (m_vertexBuffers.size() != m_vertexBufferDescriptor.m_bindings.size()) {
    vkrError("Num of vertex buffers and descriptor bindings are not equal");
  }
  RenderPassResourceTransition trans;
  trans.m_required = false;
  for (auto *x : buffers) {
    addInputResource(x, trans);
  }
  
}

IFRIT_APIDECL
void GraphicsPass::setIndexInput(RegisteredBufferHandle *buffer,
                                 VkIndexType type) {
  m_indexBuffer = buffer;
  m_indexType = type;

  RenderPassResourceTransition trans;
  trans.m_required = false;
  addInputResource(buffer, trans);
}

IFRIT_APIDECL void GraphicsPass::build(uint32_t numMultiBuffers) {
  GraphicsPipelineCreateInfo ci;
  ci.shaderModules.push_back(m_vertexShader);
  ci.shaderModules.push_back(m_fragmentShader);
  ci.viewportCount = 1;
  ci.scissorCount = 1;
  ci.topology = m_topology;
  ci.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
  if (m_depthAttachment.m_image) {
    ci.depthAttachmentFormat =
        m_depthAttachment.m_image->getImage(0)->getFormat();
  } else {
    ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
  }
  for (int i = 0; i < m_colorAttachments.size(); i++) {
    ci.colorAttachmentFormats.push_back(
        m_colorAttachments[i].m_image->getImage(0)->getFormat());
  }
  if (m_passDescriptorLayout.size() > 0) {
    ci.descriptorSetLayouts.push_back(m_descriptorManager->getBindlessLayout());
    ci.descriptorSetLayouts.push_back(
        m_descriptorManager->getParameterDescriptorSetLayout(
            m_descriptorBindRange[0].rangeId));
  }

  m_pipeline = m_pipelineCache->getGraphicsPipeline(ci);
  setBuilt();
}

IFRIT_APIDECL uint32_t GraphicsPass::getRequiredQueueCapability() {
  return VK_QUEUE_GRAPHICS_BIT;
}

// Class : ComputePass

IFRIT_APIDECL void ComputePass::build(uint32_t numMultiBuffers) {
  ComputePipelineCreateInfo ci;
  ci.shaderModules = m_shaderModule;
  if (m_passDescriptorLayout.size() > 0) {
    ci.descriptorSetLayouts.push_back(m_descriptorManager->getBindlessLayout());
    ci.descriptorSetLayouts.push_back(
        m_descriptorManager->getParameterDescriptorSetLayout(
            m_descriptorBindRange[0].rangeId));
  }
  m_pipeline = m_pipelineCache->getComputePipeline(ci);
  setBuilt();
}

IFRIT_APIDECL void ComputePass::setComputeShader(ShaderModule *shader) {
  m_shaderModule = shader;
}

IFRIT_APIDECL uint32_t ComputePass::getRequiredQueueCapability() {
  return VK_QUEUE_COMPUTE_BIT;
}

IFRIT_APIDECL void ComputePass::record() {
  m_passContext.m_cmd = m_commandBuffer;
  m_passContext.m_frame = m_activeFrame;

  if (m_passDescriptorLayout.size() > 0) {
    auto bindlessSet = m_descriptorManager->getBindlessSet();
    auto dynamicSet = m_descriptorManager->getParameterDescriptorSet(
        m_descriptorBindRange[m_passContext.m_frame].rangeId);
    auto dynamicRange =
        m_descriptorBindRange[m_passContext.m_frame].rangeOffset;
    vkCmdBindDescriptorSets(
        m_passContext.m_cmd->getCommandBuffer(), VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline->getLayout(), 0, 1, &bindlessSet, 0, nullptr);
    vkCmdBindDescriptorSets(
        m_passContext.m_cmd->getCommandBuffer(), VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline->getLayout(), 1, 1, &dynamicSet, 1, &dynamicRange);
  }

  vkCmdBindPipeline(m_passContext.m_cmd->getCommandBuffer(),
                    VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->getPipeline());

  if (m_recordFunction) {
    m_recordFunction(&m_passContext);
  }
}

// Class: RenderGraph
IFRIT_APIDECL
RenderGraph::RenderGraph(EngineContext *context,
                         DescriptorManager *descriptorManager) {
  m_context = context;
  m_descriptorManager = descriptorManager;
  m_pipelineCache = std::make_unique<PipelineCache>(context);
}

IFRIT_APIDECL GraphicsPass *RenderGraph::addGraphicsPass() {
  auto pass = std::make_unique<GraphicsPass>(m_context, m_pipelineCache.get(),
                                             m_descriptorManager);
  auto ptr = pass.get();
  m_passes.push_back(std::move(pass));
  return ptr;
}

IFRIT_APIDECL ComputePass *RenderGraph::addComputePass() {
  auto pass = std::make_unique<ComputePass>(m_context, m_pipelineCache.get(),
                                            m_descriptorManager);
  auto ptr = pass.get();
  m_passes.push_back(std::move(pass));
  return ptr;
}

IFRIT_APIDECL RegisteredBufferHandle *
RenderGraph::registerBuffer(SingleBuffer *buffer) {
  auto registeredBuffer = std::make_unique<RegisteredBufferHandle>(buffer);
  auto ptr = registeredBuffer.get();
  m_resources.push_back(std::move(registeredBuffer));
  m_resourceMap[ptr] = m_resources.size() - 1;
  return ptr;
}

IFRIT_APIDECL RegisteredSamplerHandle *
RenderGraph::registerSampler(Sampler *sampler) {
  auto registeredSampler = std::make_unique<RegisteredSamplerHandle>(sampler);
  auto ptr = registeredSampler.get();
  m_resources.push_back(std::move(registeredSampler));
  m_resourceMap[ptr] = m_resources.size() - 1;
  return ptr;
}

IFRIT_APIDECL RegisteredBufferHandle *
RenderGraph::registerBuffer(MultiBuffer *buffer) {
  auto registeredBuffer = std::make_unique<RegisteredBufferHandle>(buffer);
  auto ptr = registeredBuffer.get();
  m_resources.push_back(std::move(registeredBuffer));
  m_resourceMap[ptr] = m_resources.size() - 1;
  return ptr;
}

IFRIT_APIDECL RegisteredImageHandle *
RenderGraph::registerImage(SingleDeviceImage *image) {
  if (image->getIsSwapchainImage()) {
    auto p = dynamic_cast<SwapchainImageResource *>(image);
    if (p == nullptr) {
      vkrError("Invalid swapchain image");
    }
    auto registeredImage = std::make_unique<RegisteredSwapchainImage>(p);
    auto ptr = registeredImage.get();
    m_resources.push_back(std::move(registeredImage));
    m_resourceMap[ptr] = m_resources.size() - 1;
    m_swapchainImageHandle.push_back(ptr);
    return ptr;
  } else {
    auto registeredImage = std::make_unique<RegisteredImageHandle>(image);
    auto ptr = registeredImage.get();
    m_resources.push_back(std::move(registeredImage));
    m_resourceMap[ptr] = m_resources.size() - 1;
    return ptr;
  }
  return nullptr;
}

IFRIT_APIDECL void RenderGraph::build(uint32_t numMultiBuffers) {
  // Build all passes
  for (int i = 0; i < m_passes.size(); i++) {
    if (!m_passes[i]->isBuilt()) {
      m_passes[i]->build(numMultiBuffers);
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

IFRIT_APIDECL void RenderGraph::buildPassDescriptors(uint32_t numMultiBuffer) {
  for (int i = 0; i < m_passes.size(); i++) {
    auto p = m_passes[i].get();
    p->buildDescriptorParamHandle(numMultiBuffer);
  }
}

// Class : CommandExecutor
IFRIT_APIDECL RenderGraph *CommandExecutor::createRenderGraph() {
  auto p = std::make_unique<RenderGraph>(m_context, m_descriptorManager);
  auto ptr = p.get();
  m_renderGraph.push_back(std::move(p));
  return ptr;
}

IFRIT_APIDECL void CommandExecutor::setQueues(bool reqPresentQueue,
                                              int numGraphics, int numCompute,
                                              int numTransfer) {
  if (m_queueCollections == nullptr) {
    m_queueCollections = std::make_unique<QueueCollections>(m_context);
    m_queueCollections->loadQueues();
  }
  auto graphicsQueues = m_queueCollections->getGraphicsQueues();
  auto computeQueues = m_queueCollections->getComputeQueues();
  std::vector<Queue *> chosenQueue;
  uint32_t numGraphicsQueues = numGraphics;
  uint32_t numComputeQueues = numCompute;
  if (reqPresentQueue) {
    VkQueue presentQueue = m_swapchain->getPresentQueue();
    bool found = false;
    for (int i = 0; i < graphicsQueues.size(); i++) {
      if (graphicsQueues[i]->getQueue() == presentQueue) {
        chosenQueue.push_back(graphicsQueues[i]);
        found = true;
        numGraphicsQueues--;
        break;
      }
    }
    if (!found) {
      // TODO: Wrong handling
      vkrError("Present queue not found in graphics queues");
    }
  }
  for (int i = 0; i < numGraphicsQueues; i++) {
    chosenQueue.push_back(graphicsQueues[i]);
  }

  m_queuesGraphics = chosenQueue;
  chosenQueue.clear();

  // Compute queue
  for (int i = 0, j = 0; i < numComputeQueues; j++) {
    bool isGraphicsQueue = false;
    for (int i = 0; i < m_queuesGraphics.size(); i++) {
      if (m_queuesGraphics[i]->getQueue() == computeQueues[j]->getQueue()) {
        j++;
        isGraphicsQueue = true;
        break;
      }
    }
    if (isGraphicsQueue)
      continue;
    chosenQueue.push_back(computeQueues[j]);
    i++;
  }
  m_queuesCompute = chosenQueue;

  // Transfer queue
  chosenQueue.clear();
  auto transferQueues = m_queueCollections->getTransferQueues();
  for (int i = 0, j = 0; i < numTransfer; j++) {
    bool isGraphicsQueue = false;
    bool isComputeQueue = false;
    for (int k = 0; k < m_queuesGraphics.size(); k++) {
      if (m_queuesGraphics[k]->getQueue() == transferQueues[j]->getQueue()) {
        j++;
        isGraphicsQueue = true;
        break;
      }
    }
    for (int k = 0; k < m_queuesCompute.size(); k++) {
      if (m_queuesCompute[k]->getQueue() == transferQueues[j]->getQueue()) {
        j++;
        isComputeQueue = true;
        break;
      }
    }
    if (isGraphicsQueue || isComputeQueue)
      continue;
    chosenQueue.push_back(transferQueues[j]);
    i++;
  }
  m_queuesTransfer = chosenQueue;

  // insert all into m_queue
  m_queues.insert(m_queues.end(), m_queuesGraphics.begin(),
                  m_queuesGraphics.end());
  m_queues.insert(m_queues.end(), m_queuesCompute.begin(),
                  m_queuesCompute.end());
  m_queues.insert(m_queues.end(), m_queuesTransfer.begin(),
                  m_queuesTransfer.end());
}

IFRIT_APIDECL void CommandExecutor::compileGraph(RenderGraph *graph,
                                                 uint32_t numMultiBuffers) {
  m_graph = graph;
  if (m_graph->m_subgraphs.size() == 0) {
    graph->buildPassDescriptors(numMultiBuffers);
    m_descriptorManager->buildBindlessParameter();
    graph->build(numMultiBuffers);
  }
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

IFRIT_APIDECL void CommandExecutor::syncMultiBufferStateWithSwapchain() {
  m_resourceManager->setActiveFrame(m_swapchain->getCurrentImageId());
}

IFRIT_APIDECL void CommandExecutor::runRenderGraph(RenderGraph *graph) {
  syncMultiBufferStateWithSwapchain();
  auto numBackBuffers = m_swapchain->getNumBackbuffers();
  auto currentFrame = m_swapchain->getCurrentImageId();
  if (!graph->m_comiled) {
    compileGraph(graph, numBackBuffers);
  }

  // perform 'execution' handlers attached on passes
  for (int i = 0; i < m_graph->m_passes.size(); i++) {
    auto pass = m_graph->m_passes[i].get();
    pass->setActiveFrame(currentFrame);
    pass->execute();
  }

  // for each subgraph, create a command buffer
  auto subgraphs = graph->getSubgraphs();
  for (int i = 0; i < subgraphs.size(); i++) {
    auto queue = m_graph->m_assignedQueues[i];
    TimelineSemaphoreWait lastWait;
    for (int j = 0; j < subgraphs[i].size(); j++) {
      auto commandBuffer = queue->beginRecording();
      auto pass = m_graph->m_passes[subgraphs[i][j]].get();
      pass->setActiveFrame(currentFrame);
      pass->withCommandBuffer(commandBuffer, [&pass]() { pass->record(); });

      // TODO: Semaphores & Sync
      if (m_graph->m_subGraphOperatesOnSwapchain[i] &&
          j == subgraphs[i].size() - 1) {
        // Make swapchain image in layout present
        PipelineBarrier barrier(
            m_context, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
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
        wait.m_semaphore =
            m_swapchain->getImageAvailableSemaphoreCurrentFrame();
        wait.m_value = 0;
        wait.m_waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        auto renderFinishSemaphore =
            m_swapchain->getRenderingFinishSemaphoreCurrentFrame();

        auto fence = m_swapchain->getCurrentFrameFence();

        TimelineSemaphoreWait newWait;
        if (j == 0) {
          newWait = queue->submitCommand({wait}, fence, renderFinishSemaphore);
        } else {
          newWait = queue->submitCommand({wait, lastWait}, fence,
                                         renderFinishSemaphore);
        }
        lastWait = newWait;
      } else {
        TimelineSemaphoreWait newWait;
        if (j == 0) {
          newWait = queue->submitCommand({}, nullptr);
        } else {
          newWait = queue->submitCommand({lastWait}, nullptr);
        }
        lastWait = newWait;
      }
    }
  }
}

IFRIT_APIDECL void
CommandExecutor::runImmidiateCommand(std::function<void(CommandBuffer *)> func,
                                     QueueRequirement req) {
  auto queue = m_queues;
  auto requiredCapability = getUnderlying(req);
  for (int i = 0; i < queue.size(); i++) {
    if ((queue[i]->getCapability() & requiredCapability) ==
        requiredCapability) {
      auto commandBuffer = queue[i]->beginRecording();
      func(commandBuffer);
      queue[i]->submitCommand({}, nullptr);
      queue[i]->waitIdle();
      return;
    }
  }
  vkrError("No queue found for immediate command");
}

IFRIT_APIDECL SwapchainImageResource *
CommandExecutor::getSwapchainImageResource() {
  return m_swapchainImageResource.get();
}

IFRIT_APIDECL void CommandExecutor::beginFrame() {
  m_swapchain->acquireNextImage();
}
IFRIT_APIDECL void CommandExecutor::endFrame() { m_swapchain->presentImage(); }

} // namespace Ifrit::Engine::VkRenderer
