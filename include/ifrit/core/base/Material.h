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

class ShaderPass {
  using DrawPass = Ifrit::GraphicsBackend::Rhi::RhiGraphicsPass;

public:
  std::vector<IShaderAsset *> m_shaders;
  std::vector<AssetReference> m_shaderReferences;
  DrawPass *m_drawPass = nullptr;

  IFRIT_STRUCT_SERIALIZE(m_shaderReferences);
};

class Material : public IAssetCompatible {
public:
  std::string m_name;
  std::string m_uuid;
  std::vector<std::vector<char>> m_data;
  std::unordered_map<GraphicsShaderPassType, ShaderPass> m_passes;
  std::unordered_map<GraphicsShaderPassType,
                     std::unordered_map<std::string, uint32_t>>
      m_shaderParameters;
  IFRIT_STRUCT_SERIALIZE(m_passes, m_data, m_shaderParameters);
};

} // namespace Ifrit::Core