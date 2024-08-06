#include "demo/DefaultDemo.h"

int main() {
#ifdef IFRIT_FEATURE_CUDA
	Ifrit::Demo::mainGpu();
#else
	Ifrit::Demo::mainCpu();
#endif
}