#include "./presentation/window/AdaptiveWindowBuilder.h"
#include "./presentation/window/GLFWWindowProvider.h"

namespace Ifrit::Presentation::Window {
	std::unique_ptr<WindowProvider> AdaptiveWindowBuilder::buildUniqueWindowProvider() {
		auto obj = std::make_unique<GLFWWindowProvider>();
		return obj;
	}
}