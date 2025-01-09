
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

#include "ifrit/core/base/Camera.h"
#include "ifrit/core/base/Component.h"

namespace Ifrit::Core {

class IFRIT_APIDECL SceneNode {
protected:
  std::vector<std::shared_ptr<SceneNode>> m_children;
  std::vector<std::shared_ptr<SceneObject>> m_gameObjects;

public:
  SceneNode() = default;
  virtual ~SceneNode() = default;
  std::shared_ptr<SceneNode> addChildNode();
  std::shared_ptr<SceneObject> addGameObject(const std::string &name);
  std::shared_ptr<SceneObject>
  addGameObjectTransferred(std::shared_ptr<SceneObject> &&obj);
  inline std::shared_ptr<SceneNode> getSceneNode(uint32_t x) {
    return m_children.at(x);
  }
  inline std::shared_ptr<SceneObject> getGameObject(uint32_t x) {
    return m_gameObjects.at(x);
  }
  inline std::vector<std::shared_ptr<SceneNode>> getChildren() {
    std::vector<std::shared_ptr<SceneNode>> x;
    for (auto &y : m_children) {
      x.push_back(y);
    }
    return x;
  }
  inline std::vector<std::shared_ptr<SceneObject>> getGameObjects() {
    std::vector<std::shared_ptr<SceneObject>> x;
    for (auto &y : m_gameObjects) {
      x.push_back(y);
    }
    return x;
  }

  IFRIT_STRUCT_SERIALIZE(m_children, m_gameObjects);
};

class IFRIT_APIDECL Scene {
protected:
  std::shared_ptr<SceneNode> m_root;

public:
  Scene() : m_root(std::make_shared<SceneNode>()) {}
  std::shared_ptr<SceneNode> addSceneNode();
  inline std::shared_ptr<SceneNode> getRootNode() { return m_root; }

  Camera *getMainCamera();

  std::vector<std::shared_ptr<SceneObject>>
  filterObjects(std::function<bool(std::shared_ptr<SceneObject>)> filter);

  std::vector<SceneObject *>
  filterObjectsUnsafe(std::function<bool(std::shared_ptr<SceneObject>)> filter);

  IFRIT_STRUCT_SERIALIZE(m_root);
};

} // namespace Ifrit::Core