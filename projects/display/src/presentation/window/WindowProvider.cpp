#include "ifrit/display/presentation/window/WindowProvider.h"

namespace Ifrit::Display::Window {
IFRIT_APIDECL size_t WindowProvider::getWidth() const { return width; }
IFRIT_APIDECL size_t WindowProvider::getHeight() const { return height; }
} // namespace Ifrit::Display::Window