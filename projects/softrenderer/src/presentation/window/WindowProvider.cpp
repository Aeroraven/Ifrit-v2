#include "presentation/window/WindowProvider.h"

namespace Ifrit::Presentation::Window {
size_t WindowProvider::getWidth() const { return width; }
size_t WindowProvider::getHeight() const { return height; }
} // namespace Ifrit::Presentation::Window