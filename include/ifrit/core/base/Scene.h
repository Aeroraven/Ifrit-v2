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
  std::shared_ptr<SceneObject> addGameObject();
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

  IFRIT_STRUCT_SERIALIZE(m_root);
};

} // namespace Ifrit::Core