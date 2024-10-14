#pragma once

namespace Ifrit::Demo::DemoDefault {
int mainCpu();
#ifdef IFRIT_FEATURE_CUDA
int mainGpu();
#endif
} // namespace Ifrit::Demo::DemoDefault
