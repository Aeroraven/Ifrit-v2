
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
#include "Component.h"
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/common/serialization/MathTypeSerialization.h"
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/TypingUtil.h"

namespace Ifrit::Core {

enum class LightType { Directional };
struct LightData {
  LightType m_type = LightType::Directional;
  bool m_affectPbrSky = false;
  bool m_shadowMap = false;
  u32 m_shadowMapResolution = 512;
  IFRIT_STRUCT_SERIALIZE(m_type, m_affectPbrSky, m_shadowMap);
};

class IFRIT_APIDECL Light : public Component, public AttributeOwner<LightData> {
public:
  Light(Ref<SceneObject> owner) : Component(owner), AttributeOwner() {}
  virtual ~Light() = default;
  inline String serialize() override { return serializeAttribute(); }
  inline void deserialize() override { deserializeAttribute(); }

  // getters
  inline LightType getType() const { return m_attributes.m_type; }
  inline bool getAffectPbrSky() const { return m_attributes.m_affectPbrSky; }
  inline bool getShadowMap() const { return m_attributes.m_shadowMap; }
  inline u32 getShadowMapResolution() const { return m_attributes.m_shadowMapResolution; }

  // setters
  inline void setType(const LightType &type) { m_attributes.m_type = type; }
  inline void setAffectPbrSky(bool affectPbrSky) { m_attributes.m_affectPbrSky = affectPbrSky; }
  inline void setShadowMap(bool shadowMap) { m_attributes.m_shadowMap = shadowMap; }
  inline void setShadowMapResolution(u32 resolution) { m_attributes.m_shadowMapResolution = resolution; }
};
} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::Light)
IFRIT_ENUMCLASS_SERIALIZE(Ifrit::Core::LightType)