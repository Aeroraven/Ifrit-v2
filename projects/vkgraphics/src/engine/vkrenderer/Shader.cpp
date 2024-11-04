#include "ifrit/vkgraphics/engine/vkrenderer/Shader.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include <iostream>
#include <shaderc/shaderc.hpp>

namespace Ifrit::GraphicsBackend::VulkanGraphics {
std::vector<uint32_t> compileShaderFile(const std::string &source_name,
                                   shaderc_shader_kind kind,
                                   const std::string &source,
                                   bool optimize = false) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;
  if (optimize)
    options.SetOptimizationLevel(shaderc_optimization_level_size);
  options.SetTargetEnvironment(shaderc_target_env_vulkan,
                               shaderc_env_version_vulkan_1_2);

  shaderc::SpvCompilationResult module =
      compiler.CompileGlslToSpv(source, kind, source_name.c_str(), options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << module.GetErrorMessage();
    std::abort();
    return std::vector<uint32_t>();
  }

  return {module.cbegin(), module.cend()};
}

IFRIT_APIDECL ShaderModule::ShaderModule(EngineContext *ctx,
                                         const ShaderModuleCI &ci) {
  m_context = ctx;
  VkShaderModuleCreateInfo moduleCI{};
  shaderc_shader_kind kind;
  if (ci.stage == Rhi::RhiShaderStage::Vertex) {
    m_stageCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
    kind = shaderc_vertex_shader;
  } else if (ci.stage == Rhi::RhiShaderStage::Fragment) {
    m_stageCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    kind = shaderc_fragment_shader;
  } else if (ci.stage == Rhi::RhiShaderStage::Compute) {
    m_stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    kind = shaderc_compute_shader;
  } else if (ci.stage == Rhi::RhiShaderStage::Mesh) {
    m_stageCI.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    kind = shaderc_mesh_shader;
  } else if (ci.stage == Rhi::RhiShaderStage::Task) {
    m_stageCI.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
    kind = shaderc_task_shader;
  }

  std::vector<uint32_t> compiledCode;
  if (ci.sourceType == Rhi::RhiShaderSourceType::GLSLCode) {
    std::string rawCode(ci.code.begin(), ci.code.end());
    compiledCode = compileShaderFile(
        ci.entryPoint, static_cast<shaderc_shader_kind>(kind), rawCode);
    moduleCI.codeSize = compiledCode.size() * sizeof(uint32_t);
    moduleCI.pCode = compiledCode.data();

  } else {
    moduleCI.codeSize = ci.code.size();
    moduleCI.pCode = reinterpret_cast<const uint32_t *>(ci.code.data());
  }
  VkDevice device = m_context->getDevice();
  moduleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

  vkrVulkanAssert(vkCreateShaderModule(device, &moduleCI, nullptr, &m_module),
                  "Failed to create shader module");
  m_stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

  m_ci = ci;
  m_stageCI.module = m_module;
  m_stageCI.pName = m_ci.entryPoint.c_str();
  m_stageCI.flags = 0;
  m_stageCI.pNext = nullptr;
  
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
} // namespace Ifrit::GraphicsBackend::VulkanGraphics
