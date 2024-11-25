#pragma once
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include "spirv_reflect/spirv_reflect.h"
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace Ifrit::GraphicsBackend::VulkanGraphics {

struct ShaderModuleCI {
  std::vector<char> code;
  std::string entryPoint;
  Rhi::RhiShaderStage stage;
  Rhi::RhiShaderSourceType sourceType;
  std::string fileName;
};

class IFRIT_APIDECL ShaderModule : public Rhi::RhiShader {
private:
  VkShaderModule m_module;
  VkPipelineShaderStageCreateInfo m_stageCI{};
  EngineContext *m_context;
  ShaderModuleCI m_ci;
  std::string m_entryPoint;
  SpvReflectShaderModule m_reflectModule;
  std::vector<SpvReflectDescriptorSet *> m_reflectSets;
  bool m_reflectionCreated = false;

  // Intended for pipeline cache
  std::string m_signature;

public:
  ShaderModule(EngineContext *ctx, const ShaderModuleCI &ci);
  ~ShaderModule();
  VkShaderModule getModule() const;
  VkPipelineShaderStageCreateInfo getStageCI() const;
  inline uint32_t getCodeSize() const {
    using namespace Ifrit::Common::Utility;
    return size_cast<uint32_t>(m_ci.code.size());
  }
  inline uint32_t getNumDescriptorSets() const override {
    using namespace Ifrit::Common::Utility;
    return size_cast<uint32_t>(m_reflectSets.size());
  }
  virtual Rhi::RhiShaderStage getStage() const override { return m_ci.stage; }

  void cacheReflectionData();
  void recoverReflectionData();

  // get signature
  inline std::string getSignature() const { return m_signature; }
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics