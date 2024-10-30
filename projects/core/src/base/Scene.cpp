#pragma once

#include "ifrit/core/base/Scene.h"
namespace Ifrit::Core {

IFRIT_APIDECL std::shared_ptr<SceneNode> SceneNode::addChildNode() {
  auto node = std::make_shared<SceneNode>();
  m_children.push_back(node);
  return node;
}
IFRIT_APIDECL std::shared_ptr<SceneObject> SceneNode::addGameObject() {
  auto obj = std::make_shared<SceneObject>();
  obj->initialize();
  m_gameObjects.push_back(obj);
  return obj;
}
IFRIT_APIDECL std::shared_ptr<SceneNode> Scene::addSceneNode() {
  return m_root->addChildNode();
}

} // namespace Ifrit::Core