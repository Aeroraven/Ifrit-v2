#pragma once

namespace Ifrit::Demo::MeshletDemo {
	int mainCpu();
#ifdef IFRIT_FEATURE_CUDA
	int mainGpu();
#endif
}
