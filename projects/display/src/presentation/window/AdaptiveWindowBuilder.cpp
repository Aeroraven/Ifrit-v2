#include "ifrit/display/presentation/window/AdaptiveWindowBuilder.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"

namespace Ifrit::Presentation::Window {
IFRIT_APIDECL std::unique_ptr<WindowProvider>
AdaptiveWindowBuilder::buildUniqueWindowProvider() {
  auto obj = std::make_unique<GLFWWindowProvider>();
  return obj;
}
} // namespace Ifrit::Presentation::Window