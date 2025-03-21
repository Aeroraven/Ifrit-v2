
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

#include "ifrit/core/base/Component.h"

namespace Ifrit::Core::Ayanami {
class IFRIT_APIDECL AyanamiMeshDF : public Component {
private:
  Vec<f32> m_sdfData;
  u32 m_sdWidth;
  u32 m_sdHeight;
  u32 m_sdDepth;
  ifloat3 m_sdBoxMin;
  ifloat3 m_sdBoxMax;

public:
  AyanamiMeshDF() {}
  AyanamiMeshDF(std::shared_ptr<SceneObject> owner) : Component(owner) {}
  virtual ~AyanamiMeshDF() = default;

  inline std::string serialize() override { return ""; }
  inline void deserialize() override {}

public:
  void buildMeshDF(const std::string_view &cachePath);
  IFRIT_COMPONENT_SERIALIZE(m_sdWidth, m_sdHeight, m_sdDepth, m_sdBoxMin, m_sdBoxMax);
};

} // namespace Ifrit::Core::Ayanami

IFRIT_COMPONENT_REGISTER(Ifrit::Core::Ayanami::AyanamiMeshDF)
