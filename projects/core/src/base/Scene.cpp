
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

IFRIT_APIDECL Ref<SceneNode> SceneNode::addChildNode() {
  auto node = std::make_shared<SceneNode>();
  m_children.push_back(node);
  return node;
}
IFRIT_APIDECL Ref<SceneObject> SceneNode::addGameObject(const String &name) {
  auto obj = std::make_shared<SceneObject>();
  obj->initialize();
  obj->setName(name);
  m_gameObjects.push_back(obj);
  return obj;
}

IFRIT_APIDECL Ref<SceneObject> SceneNode::addGameObjectTransferred(Ref<SceneObject> &&obj) {
  m_gameObjects.push_back(std::move(obj));
  return obj;
}

IFRIT_APIDECL void SceneNode::onUpdate() {
  for (auto &child : m_children) {
    child->onUpdate();
  }
  for (auto &obj : m_gameObjects) {
    for (auto &comp : obj->getAllComponents()) {
      comp->onUpdate();
    }
  }
}

IFRIT_APIDECL void SceneNode::onComponentStart() {
  for (auto &child : m_children) {
    child->onComponentStart();
  }
  for (auto &obj : m_gameObjects) {
    for (auto &comp : obj->getAllComponents()) {
      comp->invokeStart();
    }
  }
}

IFRIT_APIDECL void SceneNode::onComponentAwake() {
  for (auto &child : m_children) {
    child->onComponentAwake();
  }
  for (auto &obj : m_gameObjects) {
    for (auto &comp : obj->getAllComponents()) {
      comp->invokeAwake();
    }
  }
}

IFRIT_APIDECL void SceneNode::onFixedUpdate() {
  for (auto &child : m_children) {
    child->onFixedUpdate();
  }
  for (auto &obj : m_gameObjects) {
    for (auto &comp : obj->getAllComponents()) {
      comp->onFixedUpdate();
    }
  }
}

IFRIT_APIDECL Ref<SceneNode> Scene::addSceneNode() { return m_root->addChildNode(); }

IFRIT_APIDECL Camera *Scene::getMainCamera() {
  Vec<SceneNode *> nodes;
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

IFRIT_APIDECL Vec<Ref<SceneObject>> Scene::filterObjects(Fn<bool(Ref<SceneObject>)> filter) {
  Vec<Ref<SceneObject>> result;
  Vec<SceneNode *> nodes;
  nodes.push_back(m_root.get());
  while (!nodes.empty()) {
    auto node = nodes.back();
    nodes.pop_back();
    for (auto &child : node->getChildren()) {
      nodes.push_back(child.get());
    }
    for (auto &obj : node->getGameObjects()) {
      if (filter(obj)) {
        result.push_back(obj);
      }
    }
  }
  return result;
}

IFRIT_APIDECL Vec<SceneObject *> Scene::filterObjectsUnsafe(Fn<bool(SceneObject *)> filter) {
  Vec<SceneObject *> result;
  Vec<SceneNode *> nodes;
  nodes.push_back(m_root.get());
  while (!nodes.empty()) {
    auto node = nodes.back();
    nodes.pop_back();
    for (auto &child : node->getChildren()) {
      nodes.push_back(child.get());
    }
    for (auto &obj : node->getGameObjects()) {
      if (filter(obj.get())) {
        result.push_back(obj.get());
      }
    }
  }
  return result;
}

IFRIT_APIDECL void Scene::onUpdate() { m_root->onUpdate(); }
IFRIT_APIDECL void Scene::onComponentAwake() { m_root->onComponentAwake(); }
IFRIT_APIDECL void Scene::onComponentStart() { m_root->onComponentStart(); }

IFRIT_APIDECL void Scene::onFixedUpdate(TimingRecorder *stopwatch, u32 fixedUpdateRate, u32 maxCompensationFrames) {
  auto lastTimeStamp = stopwatch->getCurTimeUs();
  auto totalFrames = lastTimeStamp / fixedUpdateRate;
  auto sourceFrame = m_curFixedFrame;

  if (sourceFrame >= totalFrames) {
    return;
  }
  auto framesToUpdate = totalFrames - sourceFrame;
  if (framesToUpdate > maxCompensationFrames) {
    framesToUpdate = maxCompensationFrames;
  }
  for (u32 i = 0; i < framesToUpdate; i++) {
    m_root->onFixedUpdate();
  }
  m_curFixedFrame = totalFrames;
}

IFRIT_APIDECL void Scene::invokeFrameUpdate() {
  // TODO: the awake logic here is not correct.
  // In Unity awake is called once after the system is initialized.
  // Then if the system is initialized, components dynamically added to the scene will got
  // the Awake called immediately, before GameObject.AddComponent<T> returns.
  onComponentAwake();

  onComponentStart();
  onUpdate();
}

} // namespace Ifrit::Core