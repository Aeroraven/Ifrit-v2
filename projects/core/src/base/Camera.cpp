
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

#include "ifrit/core/base/Camera.h"
#include "ifrit/common/math/LinalgOps.h"

using namespace Ifrit::Math;

namespace Ifrit::Core {
IFRIT_APIDECL float4x4 Camera::worldToCameraMatrix() const {
  auto p = getParent();
  auto transform = p->getComponent<Transform>();
  auto pos = transform->getPosition();
  auto rot = transform->getRotation();
  ifloat4 frontRaw = ifloat4{0.0f, 0.0f, 1.0f, 0.0f};
  auto rotationMatrix = eulerAngleToMatrix(rot);
  auto front = matmul(rotationMatrix, frontRaw);
  auto upRaw = ifloat4{0.0f, 1.0f, 0.0f, 0.0f};
  auto up = matmul(rotationMatrix, upRaw);
  auto center = pos + ifloat3{front.x, front.y, front.z};
  return (
      lookAt(ifloat3{pos.x, pos.y, pos.z}, center, ifloat3{up.x, up.y, up.z}));
}
IFRIT_APIDECL float4x4 Camera::projectionMatrix() const {
  auto data = m_attributes;
  if (data.m_type == CameraType::Perspective) {
    return (
        perspectiveNegateY(data.m_fov, data.m_aspect, data.m_near, data.m_far));
  } else {
    return (orthographicNegateY(data.m_orthoSpaceSize, data.m_aspect,
                                data.m_near, data.m_far));
  }
}

IFRIT_APIDECL ifloat4 Camera::getFront() const {
  auto p = getParent();
  auto transform = p->getComponent<Transform>();
  auto pos = transform->getPosition();
  auto rot = transform->getRotation();
  ifloat4 frontRaw = ifloat4{0.0f, 0.0f, 1.0f, 0.0f};
  auto rotationMatrix = eulerAngleToMatrix(rot);
  auto front = matmul(rotationMatrix, frontRaw);
  return front;
}
} // namespace Ifrit::Core