#pragma once

#include "ifrit/core/application/Application.h"
#include "ifrit/core/base/Material.h"

namespace Ifrit::Core {

struct SyaroDefaultGBufEmitterData {
  uint32_t m_albedoId;
};

class IFRIT_APIDECL SyaroDefaultGBufEmitter : public Material {
private:
  SyaroDefaultGBufEmitterData m_materialData;

public:
  SyaroDefaultGBufEmitter(IApplication *app);
  void buildMaterial();
  ~SyaroDefaultGBufEmitter() = default;
};
} // namespace Ifrit::Core