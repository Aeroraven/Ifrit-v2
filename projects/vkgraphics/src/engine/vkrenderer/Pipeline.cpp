#include "ifrit/vkgraphics/engine/vkrenderer/Pipeline.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/utility/Logger.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::GraphicsBackend::VulkanGraphics {
template <typename E>
constexpr typename std::underlying_type<E>::type getUnderlying(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

IFRIT_APIDECL void GraphicsPipeline::init() {
  // Dynamic states
  std::vector<VkDynamicState> dynamicStates = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,

      VK_DYNAMIC_STATE_CULL_MODE_EXT,
      VK_DYNAMIC_STATE_FRONT_FACE_EXT,
      VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT, //
      VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT,
      VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
      VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
      VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
      VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
      VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
      VK_DYNAMIC_STATE_STENCIL_OP_EXT,
      VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT, // f
      VK_DYNAMIC_STATE_LOGIC_OP_EXT,
      VK_DYNAMIC_STATE_BLEND_CONSTANTS,
      VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
      VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT};

  if (m_createInfo.geomGenType ==
      Rhi::RhiGeometryGenerationType::Conventional) {
    dynamicStates.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_EXT);
  }

  VkPipelineDynamicStateCreateInfo dynamicStateCI{};
  dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicStateCI.dynamicStateCount = size_cast<int>(dynamicStates.size());
  dynamicStateCI.pDynamicStates = dynamicStates.data();

  // Input assembly
  VkPipelineInputAssemblyStateCreateInfo inputAssemblyCI{};
  inputAssemblyCI.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  if (m_createInfo.topology == Rhi::RhiRasterizerTopology::TriangleList) {
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  } else if (m_createInfo.topology == Rhi::RhiRasterizerTopology::Line) {
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  } else if (m_createInfo.topology == Rhi::RhiRasterizerTopology::Point) {
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  }

  inputAssemblyCI.primitiveRestartEnable = VK_FALSE;

  // Viewport
  VkPipelineViewportStateCreateInfo viewportCI{};
  viewportCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportCI.viewportCount = m_createInfo.viewportCount;
  viewportCI.scissorCount = m_createInfo.scissorCount;

  // Rasterization
  VkPipelineRasterizationStateCreateInfo rasterizationCI{};
  rasterizationCI.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizationCI.depthClampEnable = VK_FALSE;
  rasterizationCI.rasterizerDiscardEnable = VK_FALSE;
  rasterizationCI.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizationCI.lineWidth = 1.0f;
  rasterizationCI.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizationCI.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizationCI.depthBiasEnable = VK_FALSE;

  // Multisampling
  VkPipelineMultisampleStateCreateInfo multisampleCI{};
  multisampleCI.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampleCI.sampleShadingEnable = VK_FALSE;
  multisampleCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  // Render Info
  bool reqDepth = m_createInfo.depthAttachmentFormat != VK_FORMAT_UNDEFINED;
  VkPipelineRenderingCreateInfo renderCI{};
  renderCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
  renderCI.colorAttachmentCount =
      size_cast<int>(m_createInfo.colorAttachmentFormats.size());
  renderCI.pColorAttachmentFormats = m_createInfo.colorAttachmentFormats.data();
  renderCI.depthAttachmentFormat = m_createInfo.depthAttachmentFormat;
  renderCI.stencilAttachmentFormat = m_createInfo.stencilAttachmentFormat;
  renderCI.pNext = nullptr;

  // Color blending info
  std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment;
  for (int i = 0; i < m_createInfo.colorAttachmentFormats.size(); i++) {
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    attachment.blendEnable = VK_FALSE;
    colorBlendAttachment.push_back(attachment);
  }

  VkPipelineColorBlendStateCreateInfo colorBlendCI{};
  colorBlendCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlendCI.logicOpEnable = VK_FALSE;
  colorBlendCI.attachmentCount = size_cast<int>(colorBlendAttachment.size());
  colorBlendCI.pAttachments = colorBlendAttachment.data();

  // Depth Stencil
  VkPipelineDepthStencilStateCreateInfo depthStencilCI{};
  depthStencilCI.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencilCI.depthTestEnable = VK_FALSE;
  depthStencilCI.depthWriteEnable = VK_FALSE;
  depthStencilCI.depthCompareOp = VK_COMPARE_OP_ALWAYS;
  depthStencilCI.depthBoundsTestEnable = VK_FALSE;
  depthStencilCI.stencilTestEnable = VK_FALSE;

  // TODO: Pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutCI{};
  pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCI.setLayoutCount =
      size_cast<int>(m_createInfo.descriptorSetLayouts.size());
  pipelineLayoutCI.pSetLayouts = m_createInfo.descriptorSetLayouts.data();
  pipelineLayoutCI.pushConstantRangeCount = 0;
  pipelineLayoutCI.pPushConstantRanges = nullptr;
  if (m_createInfo.pushConstSize != 0) {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.offset = 0;
    pushConstantRange.size = m_createInfo.pushConstSize;
    pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
    pipelineLayoutCI.pushConstantRangeCount = 1;
    pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
  }
  vkrVulkanAssert(vkCreatePipelineLayout(m_context->getDevice(),
                                         &pipelineLayoutCI, nullptr, &m_layout),
                  "Failed to create pipeline layout");
  m_layoutCreated = true;

  // Stage
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
  for (auto &shaderModule : m_createInfo.shaderModules) {
    shaderStages.push_back(shaderModule->getStageCI());
  }

  auto device = m_context->getDevice();

  VkGraphicsPipelineCreateInfo pipelineCI{};
  pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineCI.layout = m_layout;
  pipelineCI.renderPass = nullptr;
  pipelineCI.subpass = 0;
  pipelineCI.pNext = &renderCI;
  pipelineCI.stageCount = size_cast<int>(shaderStages.size());
  pipelineCI.pStages = shaderStages.data();
  pipelineCI.pVertexInputState = nullptr;
  pipelineCI.pInputAssemblyState = &inputAssemblyCI;
  pipelineCI.pViewportState = &viewportCI;
  pipelineCI.pRasterizationState = &rasterizationCI;
  pipelineCI.pMultisampleState = &multisampleCI;
  pipelineCI.pDepthStencilState = nullptr;
  if (reqDepth) {
    pipelineCI.pDepthStencilState = &depthStencilCI;
  }
  pipelineCI.pColorBlendState = &colorBlendCI;
  pipelineCI.pDynamicState = &dynamicStateCI;
  pipelineCI.basePipelineHandle = VK_NULL_HANDLE;

  auto res = vkCreateGraphicsPipelines(device, nullptr, 1, &pipelineCI, nullptr,
                                       &m_pipeline);
  vkrVulkanAssert(res, "Failed to create graphics pipeline");
  m_pipelineCreated = true;
}

IFRIT_APIDECL GraphicsPipeline::~GraphicsPipeline() {
  if (m_layoutCreated) {
    vkDestroyPipelineLayout(m_context->getDevice(), m_layout, nullptr);
  }
  if (m_pipelineCreated) {
    vkDestroyPipeline(m_context->getDevice(), m_pipeline, nullptr);
  }
}

IFRIT_APIDECL void ComputePipeline::init() {
  VkPipelineLayoutCreateInfo pipelineLayoutCI{};
  pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCI.setLayoutCount =
      size_cast<int>(m_createInfo.descriptorSetLayouts.size());
  pipelineLayoutCI.pSetLayouts = m_createInfo.descriptorSetLayouts.data();
  pipelineLayoutCI.pushConstantRangeCount = 0;
  pipelineLayoutCI.pPushConstantRanges = nullptr;
  if (m_createInfo.pushConstSize != 0) {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.offset = 0;
    pushConstantRange.size = m_createInfo.pushConstSize;
    pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
    pipelineLayoutCI.pushConstantRangeCount = 1;
    pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
  }
  vkrVulkanAssert(vkCreatePipelineLayout(m_context->getDevice(),
                                         &pipelineLayoutCI, nullptr, &m_layout),
                  "Failed to create pipeline layout");
  m_layoutCreated = true;

  VkComputePipelineCreateInfo pipelineCI{};
  pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.layout = m_layout;
  pipelineCI.basePipelineHandle = VK_NULL_HANDLE;
  pipelineCI.basePipelineIndex = -1;

  // Stage
  pipelineCI.stage = m_createInfo.shaderModules->getStageCI();

  auto res = vkCreateComputePipelines(m_context->getDevice(), nullptr, 1,
                                      &pipelineCI, nullptr, &m_pipeline);
  vkrVulkanAssert(res, "Failed to create compute pipeline");
  m_pipelineCreated = true;
}

IFRIT_APIDECL ComputePipeline::~ComputePipeline() {
  if (m_layoutCreated) {
    vkDestroyPipelineLayout(m_context->getDevice(), m_layout, nullptr);
  }
  if (m_pipelineCreated) {
    vkDestroyPipeline(m_context->getDevice(), m_pipeline, nullptr);
  }
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
  for (int i = 0; i < ci.descriptorSetLayouts.size(); i++) {
    hash ^= hashFunc(reinterpret_cast<uint64_t>(ci.descriptorSetLayouts[i]));
  }
  hash ^= hashFunc(getUnderlying(ci.geomGenType));
  hash ^= hashFunc(ci.pushConstSize);
  return hash;
}

IFRIT_APIDECL uint64_t
PipelineCache::computePipelineHash(const ComputePipelineCreateInfo &ci) {
  uint64_t hash = 0x9e3779b9;
  std::hash<uint64_t> hashFunc;
  auto pStage = ci.shaderModules;
  hash ^= hashFunc(reinterpret_cast<uint64_t>(pStage));
  for (int i = 0; i < ci.descriptorSetLayouts.size(); i++) {
    hash ^= hashFunc(reinterpret_cast<uint64_t>(ci.descriptorSetLayouts[i]));
  }
  hash ^= hashFunc(ci.pushConstSize);
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
  if (a.descriptorSetLayouts.size() != b.descriptorSetLayouts.size())
    return false;
  for (int i = 0; i < a.descriptorSetLayouts.size(); i++) {
    if (a.descriptorSetLayouts[i] != b.descriptorSetLayouts[i])
      return false;
  }
  if (a.pushConstSize != b.pushConstSize) {
    return false;
  }

  return true;
}

IFRIT_APIDECL bool
PipelineCache::computePipelineEqual(const ComputePipelineCreateInfo &a,
                                    const ComputePipelineCreateInfo &b) {
  if (a.shaderModules != b.shaderModules)
    return false;
  if (a.descriptorSetLayouts.size() != b.descriptorSetLayouts.size())
    return false;
  for (int i = 0; i < a.descriptorSetLayouts.size(); i++) {
    if (a.descriptorSetLayouts[i] != b.descriptorSetLayouts[i])
      return false;
  }
  if (a.pushConstSize != b.pushConstSize) {
    return false;
  }
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
  m_graphicsPipelineMap[hash].push_back(
      size_cast<int>(m_graphicsPipelines.size()) - 1);
  return m_graphicsPipelines.back().get();
}

IFRIT_APIDECL ComputePipeline *
PipelineCache::getComputePipeline(const ComputePipelineCreateInfo &ci) {
  uint64_t hash = computePipelineHash(ci);
  for (int i = 0; i < m_computePipelineMap[hash].size(); i++) {
    int index = m_computePipelineMap[hash][i];
    if (computePipelineEqual(ci, m_computePipelineCI[index])) {
      return m_computePipelines[index].get();
    }
  }
  // Otherwise create a new pipeline
  m_computePipelineCI.push_back(ci);
  auto &&p = std::make_unique<ComputePipeline>(m_context, ci);
  m_computePipelines.push_back(std::move(p));
  m_computePipelineMap[hash].push_back(
      size_cast<int>(m_computePipelines.size()) - 1);
  return m_computePipelines.back().get();
}

} // namespace Ifrit::GraphicsBackend::VulkanGraphics