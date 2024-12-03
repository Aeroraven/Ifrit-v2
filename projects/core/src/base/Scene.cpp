
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


#include "ifrit/core/base/Scene.h"
#include "ifrit/core/base/Component.h"
namespace Ifrit::Core {

IFRIT_APIDECL std::shared_ptr<SceneNode> SceneNode::addChildNode() {
  auto node = std::make_shared<SceneNode>();
  m_children.push_back(node);
  return node;
}
IFRIT_APIDECL std::shared_ptr<SceneObject>
SceneNode::addGameObject(const std::string &name) {
  auto obj = std::make_shared<SceneObject>();
  obj->initialize();
  obj->setName(name);
  m_gameObjects.push_back(obj);
  return obj;
}
IFRIT_APIDECL std::shared_ptr<SceneNode> Scene::addSceneNode() {
  return m_root->addChildNode();
}

IFRIT_APIDECL Camera *Scene::getMainCamera() {
  std::vector<SceneNode *> nodes;
  nodes.push_back(m_root.get());
  while (!nodes.empty()) {
    auto node = nodes.back();
    nodes.pop_back();
    for (auto &child : node->getChildren()) {
      nodes.push_back(child.get());
    }
    for (auto &obj : node->getGameObjects()) {
      auto camera = obj->getComponent<Camera>();
      if (camera) {
        if (camera->isMainCamera())
          return camera.get();
      }
    }
  }
  return nullptr;
}
} // namespace Ifrit::Core