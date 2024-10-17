#include "presentation/window/WindowProvider.h"

namespace Ifrit::Presentation::Window {
IFRIT_APIDECL size_t WindowProvider::getWidth() const { return width; }
IFRIT_APIDECL size_t WindowProvider::getHeight() const { return height; }
} // namespace Ifrit::Presentation::Window