
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include "spirv_reflect/spirv_reflect.h"
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace Ifrit::GraphicsBackend::VulkanGraphics {

struct ShaderModuleCI {
  Vec<char> code;
  String entryPoint;
  Rhi::RhiShaderStage stage;
  Rhi::RhiShaderSourceType sourceType;
  String fileName;
};

class IFRIT_APIDECL ShaderModule : public Rhi::RhiShader {
private:
  VkShaderModule m_module;
  VkPipelineShaderStageCreateInfo m_stageCI{};
  EngineContext *m_context;
  ShaderModuleCI m_ci;
  String m_entryPoint;
  SpvReflectShaderModule m_reflectModule;
  Vec<SpvReflectDescriptorSet *> m_reflectSets;
  bool m_reflectionCreated = false;

  // Intended for pipeline cache
  String m_signature;

public:
  ShaderModule(EngineContext *ctx, const ShaderModuleCI &ci);
  ~ShaderModule();
  VkShaderModule getModule() const;
  VkPipelineShaderStageCreateInfo getStageCI() const;
  inline u32 getCodeSize() const {
    using namespace Ifrit::Common::Utility;
    return size_cast<u32>(m_ci.code.size());
  }
  inline u32 getNumDescriptorSets() const override {
    using namespace Ifrit::Common::Utility;
    return size_cast<u32>(m_reflectSets.size());
  }
  virtual Rhi::RhiShaderStage getStage() const override { return m_ci.stage; }

  void cacheReflectionData();
  void recoverReflectionData();

  // get signature
  inline String getSignature() const { return m_signature; }
};
} // namespace Ifrit::GraphicsBackend::VulkanGraphics