#pragma once

namespace Ifrit::Demo::Skybox {
	int mainCpu();
#ifdef IFRIT_FEATURE_CUDA
	int mainGpu();
#endif
}
