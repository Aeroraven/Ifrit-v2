#pragma once
#include <string>
#include <vector>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vulkan/vulkan.h>

namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics {
enum class ShaderStage { Vertex, Fragment, Compute, Mesh, Task };

struct ShaderModuleCI {
  std::vector<char> code;
  std::string entryPoint;
  ShaderStage stage;
};

class IFRIT_APIDECL ShaderModule {
private:
  VkShaderModule m_module;
  VkPipelineShaderStageCreateInfo m_stageCI{};
  EngineContext *m_context;
  ShaderModuleCI m_ci;
  std::string m_entryPoint;

public:
  ShaderModule(EngineContext *ctx, const std::vector<char> &code,
               const std::string &entryPoint, ShaderStage stage);
  ShaderModule(EngineContext *ctx, const ShaderModuleCI &ci);
  ~ShaderModule();
  VkShaderModule getModule() const;
  VkPipelineShaderStageCreateInfo getStageCI() const;
  inline uint32_t getCodeSize() const { return m_ci.code.size(); }
};
} // namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics