
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


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