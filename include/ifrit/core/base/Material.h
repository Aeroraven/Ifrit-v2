#pragma once
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/core/base/AssetReference.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {
class IShaderAsset {};

enum class GraphicsShaderPassType {
  Opaque,
  Transparent,
  PostProcess,
  DirectLighting,
};

class ShaderEffect {
  using DrawPass = Ifrit::GraphicsBackend::Rhi::RhiGraphicsPass;
  using Shader = Ifrit::GraphicsBackend::Rhi::RhiShader;

public:
  std::vector<Shader *> m_shaders;
  std::vector<AssetReference> m_shaderReferences;
  DrawPass *m_drawPass;

  IFRIT_STRUCT_SERIALIZE(m_shaderReferences);

  bool operator==(const ShaderEffect &other) const {
    bool result = true;
    for (size_t i = 0; i < m_shaderReferences.size(); i++) {
      result &= m_shaderReferences[i] == other.m_shaderReferences[i];
    }
    return result;
  }
};

class ShaderEffectHash {
public:
  size_t operator()(const ShaderEffect &effect) const {
    size_t hash = 0;
    for (const auto &ref : effect.m_shaderReferences) {
      hash ^= std::hash<std::string>()(ref.m_uuid);
    }
    return hash;
  }
};

class Material : public IAssetCompatible {

public:
  std::string m_name;
  std::string m_uuid;
  std::vector<std::vector<char>> m_data;
  std::unordered_map<GraphicsShaderPassType, ShaderEffect> m_effectTemplates;
  std::unordered_map<GraphicsShaderPassType,
                     std::unordered_map<std::string, uint32_t>>
      m_shaderParameters;
  IFRIT_STRUCT_SERIALIZE(m_effectTemplates, m_data, m_shaderParameters);
};

} // namespace Ifrit::Core