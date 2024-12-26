#pragma once

#include "ifrit/core/application/Application.h"
#include "ifrit/core/base/Material.h"

namespace Ifrit::Core {

struct SyaroDefaultGBufEmitterData {
  uint32_t m_albedoId;
  uint32_t m_normalMapId;
};

class IFRIT_APIDECL SyaroDefaultGBufEmitter : public Material {
private:
  SyaroDefaultGBufEmitterData m_materialData;
  static GraphicsBackend::Rhi::RhiShader *m_shader;
  static ShaderEffect m_shaderEffect;

public:
  SyaroDefaultGBufEmitter(IApplication *app);
  void buildMaterial();
  ~SyaroDefaultGBufEmitter() = default;

  inline void setAlbedoId(uint32_t id) { m_materialData.m_albedoId = id; }
  inline void setNormalMapId(uint32_t id) { m_materialData.m_normalMapId = id; }
};
} // namespace Ifrit::Core