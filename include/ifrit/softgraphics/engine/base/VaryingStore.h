#pragma once
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
#include "ifrit/softgraphics/core/definition/CoreTypes.h"

namespace Ifrit::GraphicsBackend::SoftGraphics {
union IFRIT_APIDECL VaryingStore {
  float vf;
  int vi;
  uint32_t vui;
  ifloat2 vf2;
  ifloat3 vf3;
  ifloat4 vf4;
  iint2 vi2;
  iint3 vi3;
  iint4 vi4;
  iuint2 vui2;
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics
