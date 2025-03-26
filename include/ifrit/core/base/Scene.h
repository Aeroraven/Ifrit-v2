
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/core/base/Camera.h"
#include "ifrit/core/base/Component.h"
#include "ifrit/core/util/TimingRecorder.h"

namespace Ifrit::Core {

class IFRIT_APIDECL SceneNode {
protected:
  Vec<Ref<SceneNode>> m_children;
  Vec<Ref<SceneObject>> m_gameObjects;

public:
  SceneNode() = default;
  virtual ~SceneNode() = default;
  Ref<SceneNode> addChildNode();
  Ref<SceneObject> addGameObject(const String &name);
  Ref<SceneObject> addGameObjectTransferred(Ref<SceneObject> &&obj);

  inline Ref<SceneNode> getSceneNode(u32 x) { return m_children.at(x); }
  inline Ref<SceneObject> getGameObject(u32 x) { return m_gameObjects.at(x); }
  inline Vec<Ref<SceneNode>> getChildren() {
    Vec<Ref<SceneNode>> x;
    for (auto &y : m_children) {
      x.push_back(y);
    }
    return x;
  }
  inline Vec<Ref<SceneObject>> getGameObjects() {
    Vec<Ref<SceneObject>> x;
    for (auto &y : m_gameObjects) {
      x.push_back(y);
    }
    return x;
  }

  void onComponentStart();
  void onComponentAwake();
  void onUpdate();
  void onFixedUpdate();

  IFRIT_STRUCT_SERIALIZE(m_children, m_gameObjects);
};

class IFRIT_APIDECL Scene {
protected:
  Ref<SceneNode> m_root;
  bool m_isAwake = false;
  u64 m_curFixedFrame = 0;

public:
  Scene() : m_root(std::make_shared<SceneNode>()) {}

  inline Ref<SceneNode> getRootNode() { return m_root; }
  Camera *getMainCamera();

  Ref<SceneNode> addSceneNode();
  Vec<Ref<SceneObject>> filterObjects(Fn<bool(Ref<SceneObject>)> filter);
  Vec<SceneObject *> filterObjectsUnsafe(Fn<bool(SceneObject *)> filter);

  void onComponentStart();
  void onComponentAwake();
  void onUpdate();
  void onFixedUpdate(TimingRecorder *stopwatch, u32 fixedUpdateRate, u32 maxCompensationFrames);

  void invokeFrameUpdate();

  IFRIT_STRUCT_SERIALIZE(m_root);
};

} // namespace Ifrit::Core