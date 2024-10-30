#pragma once

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
  std::shared_ptr<SceneObject> addGameObject();
  inline std::shared_ptr<SceneNode> getSceneNode(uint32_t x) {
    return m_children.at(x);
  }
  inline std::shared_ptr<SceneObject> getGameObject(uint32_t x) {
    return m_gameObjects.at(x);
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
  IFRIT_STRUCT_SERIALIZE(m_root);
};

} // namespace Ifrit::Core