
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
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/Identifier.h"
#include <atomic>
#include <random>


using namespace Ifrit::Math;

namespace Ifrit::Core {

IFRIT_APIDECL std::shared_ptr<SceneObject> SceneObject::createPrefab() {
  auto prefab = std::make_shared<SceneObjectPrefab>();
  prefab->initialize();
  prefab->addComponent<Transform>();
  return prefab;
}

IFRIT_APIDECL
Component::Component(std::shared_ptr<SceneObject> parent)
    : m_parentObject(parent) {
  Ifrit::Common::Utility::generateUuid(m_id.m_uuid);
}
IFRIT_APIDECL void SceneObject::initialize() { addComponent<Transform>(); }
IFRIT_APIDECL SceneObject::SceneObject() {
  Ifrit::Common::Utility::generateUuid(m_id.m_uuid);
}

IFRIT_APIDECL float4x4 Transform::getModelToWorldMatrix() {
  float4x4 model = identity();
  model = matmul(scale3D(m_attributes.m_scale), model);
  model = matmul(eulerAngleToMatrix(m_attributes.m_rotation), model);
  model = matmul(translate3D(m_attributes.m_position), model);
  return model;
}

IFRIT_APIDECL float4x4 Transform::getModelToWorldMatrixLast() {
  float4x4 model = identity();
  model = matmul(scale3D(m_lastFrame.m_scale), model);
  model = matmul(eulerAngleToMatrix(m_lastFrame.m_rotation), model);
  model = matmul(translate3D(m_lastFrame.m_position), model);
  return model;
}

} // namespace Ifrit::Core