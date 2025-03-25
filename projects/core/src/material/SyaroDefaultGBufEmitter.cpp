#include "ifrit/core/material/SyaroDefaultGBufEmitter.h"
#include "ifrit/common/util/FileOps.h"

namespace Ifrit::Core {
IFRIT_APIDECL
SyaroDefaultGBufEmitter::SyaroDefaultGBufEmitter(IApplication *app) : Material() {
  m_materialData.m_albedoId = ~0u;
  if (m_shader == nullptr) {
    auto rhi = app->getRhiLayer();
    String shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
    auto path = shaderBasePath + "/Syaro/Syaro.EmitGBuffer.Default.comp.glsl";
    auto shaderCode = Ifrit::Common::Utility::readTextFile(path);
    std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
    m_shader = rhi->createShader(path, shaderCodeVec, "main", GraphicsBackend::Rhi::RhiShaderStage::Compute,
                                 GraphicsBackend::Rhi::RhiShaderSourceType::GLSLCode);
    m_shaderEffect.m_shaders.push_back(m_shader);
    m_shaderEffect.m_type = ShaderEffectType::Compute;
  }
  buildMaterial();
}

IFRIT_APIDECL void SyaroDefaultGBufEmitter::buildMaterial() {
  if (m_data.size() < 1) {
    m_data.resize(1);
  }
  m_data[0].resize(sizeof(SyaroDefaultGBufEmitterData));
  auto &data = *reinterpret_cast<SyaroDefaultGBufEmitterData *>(m_data[0].data());
  data = m_materialData;
  this->m_effectTemplates[GraphicsShaderPassType::Opaque] = m_shaderEffect;
}

GraphicsBackend::Rhi::RhiShader *SyaroDefaultGBufEmitter::m_shader = nullptr;
ShaderEffect SyaroDefaultGBufEmitter::m_shaderEffect = {};

} // namespace Ifrit::Core