#include "demo/DefaultDemo.h"
#include "demo/Skybox.h"
#include "demo/MeshletDemo.h"
#include "demo/ShaderVMDemo.h"

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
#ifdef IFRIT_FEATURE_CUDA
	Ifrit::Demo::MeshletDemo::mainGpu();
#else
	Ifrit::Demo::MeshletDemo::mainCpu();
#endif
	return 0;
}

int demoShaderVMTest() {
	return Ifrit::Demo::ShaderVMDemo::mainEntry();
}

int main() {
	demoShaderVMTest();
	return 0;
}