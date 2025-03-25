
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
#include "RhiBaseTypes.h"

namespace Ifrit::GraphicsBackend::Rhi {
class IFRIT_APIDECL RhiShader {
public:
  virtual RhiShaderStage getStage() const = 0;
  virtual u32 getNumDescriptorSets() const = 0;
};

class IFRIT_APIDECL RhiRTShaderBindingTable {
public:
  virtual void _polymorphismPlaceHolder() {}
};

struct IFRIT_APIDECL RhiRTShaderGroup {
  RhiShader *m_generalShader = nullptr;
  RhiShader *m_closestHitShader = nullptr;
  RhiShader *m_anyHitShader = nullptr;
  RhiShader *m_intersectionShader = nullptr;
};

} // namespace Ifrit::GraphicsBackend::Rhi