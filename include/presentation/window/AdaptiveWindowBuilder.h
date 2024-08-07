#pragma once
#include "./core/definition/CoreExports.h"
#include "./presentation/window/WindowProvider.h"
namespace Ifrit::Presentation::Window {
	class AdaptiveWindowBuilder {
	public:
		std::unique_ptr<WindowProvider> buildUniqueWindowProvider();
	};
}