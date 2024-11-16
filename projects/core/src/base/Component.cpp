#include "ifrit/core/base/Component.h"
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/Identifier.h"
#include <atomic>
#include <random>

using namespace Ifrit::Math;

namespace Ifrit::Core {

IFRIT_APIDECL Component::Component(std::shared_ptr<SceneObject> parent)
    : m_parentObject(parent) {
  Ifrit::Common::Utility::generateUuid(m_id.m_uuid);
}
IFRIT_APIDECL void SceneObject::initialize() { addComponent<Transform>(); }
IFRIT_APIDECL SceneObject::SceneObject() {
  Ifrit::Common::Utility::generateUuid(m_id.m_uuid);
}

IFRIT_APIDECL float4x4 Transform::getModelToWorldMatrix() {
  float4x4 model = identity();
  model = matmul(model, translate3D(m_attributes.m_position));
  model = matmul(model, eulerAngleToMatrix(m_attributes.m_rotation));
  model = matmul(model, scale3D(m_attributes.m_scale));
  return transpose(model);
}

IFRIT_APIDECL float4x4 Transform::getModelToWorldMatrixLast() {
  float4x4 model = identity();
  model = matmul(model, translate3D(m_lastFrame.m_position));
  model = matmul(model, eulerAngleToMatrix(m_lastFrame.m_rotation));
  model = matmul(model, scale3D(m_lastFrame.m_scale));
  return transpose(model);
}

} // namespace Ifrit::Core