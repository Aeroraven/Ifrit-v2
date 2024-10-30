#pragma once
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/display/presentation/window/WindowProvider.h"
#include <memory>

namespace Ifrit::Display::Window {
enum class WindowProviderType { GLFW };

struct WindowProviderSetupArgs {
  bool useVulkan = false;
};
class IFRIT_APIDECL WindowSelector {
public:
  static std::unique_ptr<WindowProvider>
  createWindowProvider(WindowProviderType type,
                       const WindowProviderSetupArgs &args);
};
} // namespace Ifrit::Display::Window