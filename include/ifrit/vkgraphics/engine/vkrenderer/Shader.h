#pragma once
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace Ifrit::GraphicsBackend::VulkanGraphics {

struct ShaderModuleCI {
  std::vector<char> code;
  std::string entryPoint;
  Rhi::RhiShaderStage stage;
  Rhi::RhiShaderSourceType sourceType;
};

class IFRIT_APIDECL ShaderModule : public Rhi::RhiShader {
private:
  VkShaderModule m_module;
  VkPipelineShaderStageCreateInfo m_stageCI{};
  EngineContext *m_context;
  ShaderModuleCI m_ci;
  std::string m_entryPoint;

public:
  ShaderModule(EngineContext *ctx, const ShaderModuleCI &ci);
  ~ShaderModule();
  VkShaderModule getModule() const;
  VkPipelineShaderStageCreateInfo getStageCI() const;
  inline uint32_t getCodeSize() const {
    using namespace Ifrit::Common::Utility;
    return size_cast<int>(m_ci.code.size());
  }
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics