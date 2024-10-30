#include "ifrit/display/presentation/window/WindowSelector.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"

namespace Ifrit::Display::Window {
IFRIT_APIDECL std::unique_ptr<WindowProvider>
WindowSelector::createWindowProvider(WindowProviderType type,
                                     const WindowProviderSetupArgs &args) {
  GLFWWindowProviderInitArgs initArgs;
  initArgs.vulkanMode = args.useVulkan;
  switch (type) {
  case WindowProviderType::GLFW:
    return std::make_unique<GLFWWindowProvider>(GLFWWindowProvider(initArgs));
  default:
    return nullptr;
  }
}
} // namespace Ifrit::Display::Window