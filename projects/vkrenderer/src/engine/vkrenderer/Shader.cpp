#include <vkrenderer/include/engine/vkrenderer/Shader.h>
#include <vkrenderer/include/utility/Logger.h>

namespace Ifrit::Engine::VkRenderer {
IFRIT_APIDECL ShaderModule::ShaderModule(EngineContext *ctx,
                                         const ShaderModuleCI &ci) {
  m_context = ctx;
  VkDevice device = m_context->getDevice();
  VkShaderModuleCreateInfo moduleCI{};
  moduleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  moduleCI.codeSize = ci.code.size();
  moduleCI.pCode = reinterpret_cast<const uint32_t *>(ci.code.data());
  vkrVulkanAssert(vkCreateShaderModule(device, &moduleCI, nullptr, &m_module),
                  "Failed to create shader module");
  m_stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  m_stageCI.stage = ci.stage == ShaderStage::Vertex
                        ? VK_SHADER_STAGE_VERTEX_BIT
                        : VK_SHADER_STAGE_FRAGMENT_BIT;
  m_stageCI.module = m_module;
  m_stageCI.pName = ci.entryPoint.c_str();
  m_stageCI.flags = 0;
  m_stageCI.pNext = nullptr;
  m_ci = ci;
}

ShaderModule::ShaderModule(EngineContext *ctx, const std::vector<char> &code,
                           const std::string &entryPoint, ShaderStage stage) {
  m_context = ctx;
  VkDevice device = m_context->getDevice();
  VkShaderModuleCreateInfo moduleCI{};
  moduleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  moduleCI.codeSize = code.size();
  moduleCI.pCode = reinterpret_cast<const uint32_t *>(code.data());
  vkrVulkanAssert(vkCreateShaderModule(device, &moduleCI, nullptr, &m_module),
                  "Failed to create shader module");
  m_stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  m_stageCI.stage = stage == ShaderStage::Vertex ? VK_SHADER_STAGE_VERTEX_BIT
                                                 : VK_SHADER_STAGE_FRAGMENT_BIT;
  m_entryPoint = entryPoint;
  m_stageCI.module = m_module;
  m_stageCI.pName = m_entryPoint.c_str();
  m_stageCI.flags = 0;
  m_stageCI.pNext = nullptr;
  m_ci.code = code;
  m_ci.entryPoint = entryPoint;
  m_ci.stage = stage;
}

IFRIT_APIDECL ShaderModule::~ShaderModule() {
  vkDestroyShaderModule(m_context->getDevice(), m_module, nullptr);
  m_module = VK_NULL_HANDLE;
}

IFRIT_APIDECL VkShaderModule ShaderModule::getModule() const {
  return m_module;
}

IFRIT_APIDECL VkPipelineShaderStageCreateInfo ShaderModule::getStageCI() const {
  return m_stageCI;
}
} // namespace Ifrit::Engine::VkRenderer
