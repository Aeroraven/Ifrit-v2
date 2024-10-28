#define _CRTDBG_MAP_ALLOC
#include "demo/AccelStructDemo.h"
#include "demo/DefaultDemo.h"
#include "demo/MeshletDemo.h"
#include "demo/OglBenchmarking.h"
#include "demo/ShaderVMDemo.h"
#include "demo/Skybox.h"

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

int demoMeshlet() {
#ifdef IFRIT_FEATURE_CUDA
  Ifrit::Demo::MeshletDemo::mainGpu();
#else
  Ifrit::Demo::MeshletDemo::mainCpu();
#endif
  return 0;
}

int demoShaderVMTest() { return Ifrit::Demo::ShaderVMDemo::mainTest(); }

int demoASTest() { return Ifrit::Demo::AccelStructDemo::mainCpu(); }

int main() {
  Ifrit::Demo::ShaderVMDemo::mainTest();
  // Ifrit::Demo::OglBenchmarking::mainCpu();
  return 0;
}