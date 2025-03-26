
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
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/base/Camera.h"
#include "ifrit/core/base/Scene.h"
#include "ifrit/core/scene/FrameCollector.h"

namespace Ifrit::Core {

class IFRIT_APIDECL SceneManager {
private:
  IApplication *m_app;
  Ref<Scene> m_activeScene;

public:
  SceneManager(IApplication *app) : m_app(app) {}
  virtual ~SceneManager() = default;

  void collectPerframeData(PerFrameData &perframeData, Scene *scene, Camera *camera = nullptr,
                           GraphicsShaderPassType passType = GraphicsShaderPassType::Opaque);

  inline void setActiveScene(Ref<Scene> scene) { m_activeScene = scene; }
  inline Ref<Scene> getActiveScene() { return m_activeScene; }
  void invokeActiveSceneUpdate();
};

} // namespace Ifrit::Core