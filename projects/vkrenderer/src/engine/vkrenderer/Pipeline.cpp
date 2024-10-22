#include <vkrenderer/include/engine/vkrenderer/Pipeline.h>
#include <vkrenderer/include/utility/Logger.h>

namespace Ifrit::Engine::VkRenderer {
IFRIT_APIDECL void GraphicsPipeline::init() {
  // Dynamic states
  std::vector<VkDynamicState> dynamicStates = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_VERTEX_INPUT_EXT,
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

  //

  VkPipelineDynamicStateCreateInfo dynamicStateCI{};
  dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicStateCI.dynamicStateCount = dynamicStates.size();
  dynamicStateCI.pDynamicStates = dynamicStates.data();

  // Input assembly
  VkPipelineInputAssemblyStateCreateInfo inputAssemblyCI{};
  inputAssemblyCI.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  if (m_createInfo.topology == RasterizerTopology::TriangleList) {
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  } else if (m_createInfo.topology == RasterizerTopology::Line) {
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
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
  renderCI.colorAttachmentCount = m_createInfo.colorAttachmentFormats.size();
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
  colorBlendCI.attachmentCount = colorBlendAttachment.size();
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
  pipelineLayoutCI.setLayoutCount = m_createInfo.descriptorSetLayouts.size();
  pipelineLayoutCI.pSetLayouts = m_createInfo.descriptorSetLayouts.data();
  pipelineLayoutCI.pushConstantRangeCount = 0;
  pipelineLayoutCI.pPushConstantRanges = nullptr;
  vkrVulkanAssert(vkCreatePipelineLayout(m_context->getDevice(),
                                         &pipelineLayoutCI, nullptr, &m_layout),
                  "Failed to create pipeline layout");
  m_layoutCreated = true;

  // Stage
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
  for (auto &shaderModule : m_createInfo.shaderModules) {
    shaderStages.push_back(shaderModule->getStageCI());
  }

  VkRenderPass renderPass;
  auto device = m_context->getDevice();

  VkGraphicsPipelineCreateInfo pipelineCI{};
  pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineCI.layout = m_layout;
  pipelineCI.renderPass = nullptr;
  pipelineCI.subpass = 0;
  pipelineCI.pNext = &renderCI;
  pipelineCI.stageCount = shaderStages.size();
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
  // TODO
}

IFRIT_APIDECL ComputePipeline::~ComputePipeline() {
  // TODO
}
} // namespace Ifrit::Engine::VkRenderer