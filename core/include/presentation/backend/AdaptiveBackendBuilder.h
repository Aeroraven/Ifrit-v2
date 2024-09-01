#pragma once
#include "./core/definition/CoreExports.h"
#include "./presentation/backend/BackendProvider.h"
namespace Ifrit::Presentation::Backend {
	class AdaptiveBackendBuilder {
	public:
		std::unique_ptr<BackendProvider> buildUniqueBackend();
	};
}