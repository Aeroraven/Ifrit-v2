#include "AccelStructDemo.h"
#include "presentation/window/GLFWWindowProvider.h"
#include "engine/raytracer/TrivialRaytracer.h"
#include "engine/raytracer/accelstruct/RtBoundingVolumeHierarchy.h"
#include "engine/raytracer/accelstruct/RtTrivialIterativeAccelStruct.h"
#include "utility/loader/WavefrontLoader.h"
#include "utility/loader/ImageLoader.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "presentation/backend/TerminalAsciiBackend.h"
#include "presentation/backend/OpenGLBackend.h"
#include "presentation/backend/TerminalCharColorBackend.h"
#include "engine/tilerastercuda/TileRasterRendererCuda.h"

using namespace std;
using namespace Ifrit::Core::Data;

using namespace Ifrit::Engine::Raytracer;
using namespace Ifrit::Utility::Loader;
using namespace Ifrit::Math;
using namespace Ifrit::Presentation::Window;
using namespace Ifrit::Presentation::Backend;

namespace Ifrit::Demo::AccelStructDemo {
	int mainCpu() {
		WavefrontLoader loader;
		std::vector<ifloat3> pos;
		std::vector<ifloat3> normal;
		std::vector<ifloat2> uv;
		std::vector<uint32_t> index;
		std::vector<ifloat3> procNormal;

		loader.loadObject(IFRIT_ASSET_PATH"/bunny.obj", pos, normal, uv, index);

		BoundingVolumeHierarchyBottomLevelAS blas;
		blas.bufferData(pos);
		blas.buildAccelerationStructure();

		BoundingVolumeHierarchyTopLevelAS tlas;
		std::vector<BoundingVolumeHierarchyBottomLevelAS*> blasArray = { &blas };
		tlas.bufferData(blasArray);
		tlas.buildAccelerationStructure();

		std::shared_ptr<TrivialRaytracer> raytracer = std::make_shared<TrivialRaytracer>();
		raytracer->init();
		raytracer->bindAccelerationStructure(&tlas);
		
		constexpr int DEMO_RESOLUTION = 128;
		std::shared_ptr<ImageF32> image = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4);
		raytracer->bindTestImage(image.get());
		raytracer->traceRays(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);

		ifritLog2("Start Rendering");
		GLFWWindowProvider windowProvider;
		windowProvider.setup(1920, 1080);
		windowProvider.setTitle("Ifrit-v2 CPU Multithreading");

		OpenGLBackend backend;
		backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());
		windowProvider.loop([&](int* coreTime) {
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			*coreTime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			backend.updateTexture(*image);
			backend.draw();
		 });


		return 0;
	}
}