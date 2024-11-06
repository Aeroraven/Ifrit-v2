#pragma once
#include "ifrit/core/base/ApplicationInterface.h"

namespace Ifrit::Core {

class SceneManager {
private:
  IApplication *m_app;

public:
  SceneManager(IApplication *app) : m_app(app) {}
  virtual ~SceneManager() = default;
};

} // namespace Ifrit::Core