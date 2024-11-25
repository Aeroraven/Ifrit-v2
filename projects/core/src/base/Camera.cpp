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
  return transpose(
      lookAt(ifloat3{pos.x, pos.y, pos.z}, center, ifloat3{up.x, up.y, up.z}));
}
IFRIT_APIDECL float4x4 Camera::projectionMatrix() const {
  auto data = m_attributes;
  return transpose(
      perspectiveNegateY(data.m_fov, data.m_aspect, data.m_near, data.m_far));
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