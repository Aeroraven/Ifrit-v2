#include "ifrit/core/input/InputSystem.h"
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"

using namespace Ifrit;
using namespace Ifrit::Core;

void key_callback_glfw_input_system(GLFWwindow *window, int key, int scancode, int action, int mods) {
  auto activeInputSystem = static_cast<InputSystem *>(glfwGetWindowUserPointer(window));
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
  auto windowHandle =
      static_cast<GLFWwindow *>(static_cast<GLFWWindowProvider *>(m_app->getWindowProvider())->getGLFWWindow());
  glfwSetKeyCallback(windowHandle, key_callback_glfw_input_system);
  glfwSetWindowUserPointer(windowHandle, this);
}

IFRIT_APIDECL void InputSystem::onFrameUpdate() {
  for (auto &key : m_keyStatus) {
    if (key.stat == 1) {
      key.stat = 0;
    }
  }
}

} // namespace Ifrit::Core