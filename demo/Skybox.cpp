#include "Skybox.h"
#include "presentation/backend/AdaptiveBackendBuilder.h"
#include "presentation/window/AdaptiveWindowBuilder.h"

namespace Ifrit::Demo::Skybox {
#ifdef IFRIT_FEATURE_CUDA
	int mainGpu() {
		using namespace Ifrit::Presentation::Backend;
		using namespace Ifrit::Presentation::Window;
		
		auto windowBuilder = std::make_unique<AdaptiveWindowBuilder>();
		auto windowProvider = windowBuilder->buildUniqueWindowProvider();
		windowProvider->setup(2048, 1152);

		auto backendBuilder = std::make_unique<AdaptiveBackendBuilder>();
		auto backend = backendBuilder->buildUniqueBackend();
		
		backend->setViewport(0, 0, windowProvider->getWidth(), windowProvider->getHeight());
		windowProvider->loop([&](int* coreTime) {
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			*coreTime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		});
		return 0;
	}
#endif

	int mainCpu() {
		return 0;
	}
}