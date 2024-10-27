#pragma once
#include "ifrit/display/presentation/window/WindowProvider.h"
#include <memory>

namespace Ifrit::Presentation::Window {
class IFRIT_APIDECL AdaptiveWindowBuilder {
public:
  std::unique_ptr<WindowProvider> buildUniqueWindowProvider();
};
} // namespace Ifrit::Presentation::Window