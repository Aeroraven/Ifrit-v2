#pragma once
#include <string>
#include <vector>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vulkan/vulkan.h>

namespace Ifrit::Engine::VkRenderer {
enum class ShaderStage { Vertex, Fragment };

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

public:
  ShaderModule(EngineContext *ctx, const ShaderModuleCI &ci);
  ~ShaderModule();
  VkShaderModule getModule() const;
  VkPipelineShaderStageCreateInfo getStageCI() const;
  inline uint32_t getCodeSize() const { return m_ci.code.size(); }
};
} // namespace Ifrit::Engine::VkRenderer