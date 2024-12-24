#include "ifrit/core/material/SyaroDefaultGBufEmitter.h"

namespace Ifrit::Core {
IFRIT_APIDECL
SyaroDefaultGBufEmitter::SyaroDefaultGBufEmitter(IApplication *app)
    : Material() {
  m_materialData.m_albedoId = 0;
  buildMaterial();
}

IFRIT_APIDECL void SyaroDefaultGBufEmitter::buildMaterial() {
  if (m_data.size() < 1) {
    m_data.resize(1);
  }
  m_data[0].resize(sizeof(SyaroDefaultGBufEmitterData));
  auto &data =
      *reinterpret_cast<SyaroDefaultGBufEmitterData *>(m_data[0].data());
}
} // namespace Ifrit::Core