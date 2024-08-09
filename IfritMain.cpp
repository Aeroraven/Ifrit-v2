#include "demo/DefaultDemo.h"
#include "demo/Skybox.h"
#include "demo/MeshletDemo.h"

int demoDefault() {
#ifdef IFRIT_FEATURE_CUDA
	Ifrit::Demo::DemoDefault::mainGpu();
#else
	Ifrit::Demo::DemoDefault::mainCpu();
#endif
	return 0;
}

int demoSkybox() {
#ifdef IFRIT_FEATURE_CUDA
	Ifrit::Demo::Skybox::mainGpu();
#else
	Ifrit::Demo::Skybox::mainCpu();
#endif
	return 0;
}

int demoMeshlet(){
	Ifrit::Demo::MeshletDemo::mainCpu();
	return 0;
}

int main() {
	demoMeshlet();
	return 0;
}