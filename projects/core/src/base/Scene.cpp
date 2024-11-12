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