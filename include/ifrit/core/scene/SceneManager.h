#pragma once
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/base/Camera.h"
#include "ifrit/core/base/Scene.h"
#include "ifrit/core/scene/FrameCollector.h"

namespace Ifrit::Core {

class SceneManager {
private:
  IApplication *m_app;

public:
  SceneManager(IApplication *app) : m_app(app) {}
  virtual ~SceneManager() = default;

  void collectPerframeData(
      PerFrameData &perframeData, Scene *scene, Camera *camera = nullptr,
      GraphicsShaderPassType passType = GraphicsShaderPassType::Opaque);
};

} // namespace Ifrit::Core