#include "ifrit/core/input/InputSystem.h"
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"

using namespace Ifrit;
using namespace Ifrit::Core;

static InputSystem *activeInputSystem = nullptr;

void key_callback_glfw_input_system(int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS || action == GLFW_REPEAT) {
    activeInputSystem->updateKeyStatus(key, 1);
  }
}

namespace Ifrit::Core {

IFRIT_APIDECL void InputSystem::init() {
  for (auto &key : m_keyStatus) {
    key.stat = 0;
  }
  using namespace Ifrit::Display::Window;
  activeInputSystem = this;
  auto windowProvider = static_cast<GLFWWindowProvider *>(m_app->getWindowProvider());
  auto windowHandle = static_cast<GLFWwindow *>(windowProvider->getGLFWWindow());
  windowProvider->registerKeyCallback(key_callback_glfw_input_system);
  iInfo("Input system initialized");
}

IFRIT_APIDECL void InputSystem::onFrameUpdate() {
  for (auto &key : m_keyStatus) {
    if (key.stat == 1) {
      key.stat = 0;
    }
  }
}

} // namespace Ifrit::Core