#pragma once

namespace Ifrit::Demo {
	int mainCpu();
#ifdef IFRIT_FEATURE_CUDA
	int mainGpu();
#endif
}
