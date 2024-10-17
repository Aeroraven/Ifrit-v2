#pragma once
#include <memory>
#include "./presentation/window/WindowProvider.h"
namespace Ifrit::Presentation::Window {
class IFRIT_APIDECL AdaptiveWindowBuilder {
public:
  std::unique_ptr<WindowProvider> buildUniqueWindowProvider();
};
} // namespace Ifrit::Presentation::Window