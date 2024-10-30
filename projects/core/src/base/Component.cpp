#include "ifrit/core/base/Component.h"
#include "ifrit/common/util/Identifier.h"
#include <atomic>
#include <random>

namespace Ifrit::Core {

IFRIT_APIDECL Component::Component(std::shared_ptr<SceneObject> parent)
    : m_parentObject(parent) {
  Ifrit::Common::Utility::generateUuid(m_id.m_uuid);
}
IFRIT_APIDECL void SceneObject::initialize() { addComponent<Transform>(); }
IFRIT_APIDECL SceneObject::SceneObject() {
  Ifrit::Common::Utility::generateUuid(m_id.m_uuid);
}

} // namespace Ifrit::Core