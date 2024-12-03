
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
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/common/serialization/MathTypeSerialization.h"
#include "ifrit/common/serialization/SerialInterface.h"

#include "ifrit/common/util/TypingUtil.h"

namespace Ifrit::Core {
struct CameraData {
  float m_fov = 60.0f;
  float m_aspect = 1.0f;
  float m_near = 0.1f;
  float m_far = 1000.0f;
  bool m_isMainCamera = false;
  IFRIT_STRUCT_SERIALIZE(m_fov, m_aspect, m_near, m_far, m_isMainCamera);
};
class IFRIT_APIDECL Camera : public Component,
                             public AttributeOwner<CameraData> {
public:
  Camera(std::shared_ptr<SceneObject> owner)
      : Component(owner), AttributeOwner() {}
  virtual ~Camera() = default;
  inline std::string serialize() override { return serializeAttribute(); }
  inline void deserialize() override { deserializeAttribute(); }
  float4x4 worldToCameraMatrix() const;
  float4x4 projectionMatrix() const;
  ifloat4 getFront() const;

  // getters
  inline float getFov() const { return m_attributes.m_fov; }
  inline float getAspect() const { return m_attributes.m_aspect; }
  inline float getNear() const { return m_attributes.m_near; }
  inline float getFar() const { return m_attributes.m_far; }
  inline bool isMainCamera() const { return m_attributes.m_isMainCamera; }

  // setters
  inline void setFov(float fov) { m_attributes.m_fov = fov; }
  inline void setAspect(float aspect) { m_attributes.m_aspect = aspect; }
  inline void setNear(float nearx) { m_attributes.m_near = nearx; }
  inline void setFar(float farx) { m_attributes.m_far = farx; }
  inline void setMainCamera(bool isMain) {
    m_attributes.m_isMainCamera = isMain;
  }
};
} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::Camera)