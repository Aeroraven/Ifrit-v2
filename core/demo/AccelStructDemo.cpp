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
#include "engine/raytracer/RtShaders.h"
#include "engine/raytracer/shaderops/RtShaderOps.h"
#include "engine/shadervm/spirv/SpvVMReader.h"
#include "engine/shadervm/spirv/SpvVMInterpreter.h"
#include "engine/shadervm/spirv/SpvVMShader.h"
#include "engine/comllvmrt/WrappedLLVMRuntime.h"
using namespace Ifrit::Engine::BufferManager;
using namespace Ifrit::Engine::ComLLVMRuntime;



using namespace std;
using namespace Ifrit::Engine;
using namespace Ifrit::Core::Data;
using namespace Ifrit::Engine::Raytracer;
using namespace Ifrit::Utility::Loader;
using namespace Ifrit::Math;
using namespace Ifrit::Presentation::Window;
using namespace Ifrit::Presentation::Backend;

namespace Ifrit::Demo::AccelStructDemo {
	static std::shared_ptr<ImageF32> image;

	struct Payload {
		ifloat4 color;
	};

	class DemoRayGen : public RayGenShader {
	public:
		IFRIT_DUAL virtual void execute(
			const iint3& inputInvocation,
			const iint3& dimension,
			void* context
		) {
			float dx = 1.0f * inputInvocation.x / dimension.x;
			float dy = 1.0f * inputInvocation.y / dimension.y;
			float dz = 1.0f * inputInvocation.z / dimension.z;
			float rx = 0.25f * dx - 0.125f - 0.02f;
			float ry = 0.25f * dy - 0.125f + 0.1f;
			float rz = -1.0f;
			Payload payload;
			ifritShaderOps_Raytracer_TraceRay({ rx,ry,rz }, 0, 0, 0, 0, 0,0,0.0f,{ 0.0f,0.0f,1.0f }, 1.0f, &payload, context
			);
			image->fillPixelRGBA(inputInvocation.x, inputInvocation.y, payload.color.x, payload.color.y, payload.color.z, payload.color.w);
		}
		IFRIT_HOST virtual std::unique_ptr<RayGenShader> getThreadLocalCopy() {
			return std::make_unique<DemoRayGen>();
		}
	};

	class DemoClosetHit : public CloseHitShader {
	public:
		IFRIT_DUAL virtual void execute(
			const RayHit& hitAttribute,
			const Ray& ray,
			void* payload,
			void* context
		) {
			auto p = reinterpret_cast<Payload*>(payload);
			p->color = { 1.0f,0.0f,0.0f,1.0f };
		}
		IFRIT_HOST virtual void onStackPushComplete() {}
		IFRIT_HOST virtual void onStackPopComplete() {}

		IFRIT_HOST virtual std::unique_ptr<CloseHitShader> getThreadLocalCopy() {
			return std::make_unique<DemoClosetHit>();
		}
	};

	class DemoMiss : public MissShader {
	public:
		IFRIT_DUAL virtual void execute(
			const Ray& ray,
			void* payload,
			void* context
		) {
			auto p = reinterpret_cast<Payload*>(payload);
			p->color = { 0.0f,1.0f,0.0f,1.0f };
		}
		IFRIT_HOST virtual void onStackPushComplete() {}
		IFRIT_HOST virtual void onStackPopComplete() {}

		IFRIT_HOST virtual std::unique_ptr<MissShader> getThreadLocalCopy() {
			return std::make_unique<DemoMiss>();
		}
	};


	int mainCpu() {
		using namespace Ifrit::Engine::ShaderVM::Spirv;

		WavefrontLoader loader;
		std::vector<ifloat3> posRaw;
		std::vector<ifloat3> normal;
		std::vector<ifloat2> uv;
		std::vector<uint32_t> index;
		std::vector<ifloat3> procNormal;

		loader.loadObject(IFRIT_ASSET_PATH"/bunny.obj", posRaw, normal, uv, index);

		std::vector<int> indexBuffer = { 0,1,2,2,3,0 };
		indexBuffer.resize(index.size() / 3);
		for (int i = 0; i < index.size(); i += 3) {
			indexBuffer[i / 3] = index[i];
		}
		std::vector<ifloat3> pos;
		for (int i = 0; i < indexBuffer.size(); i++) {
			pos.push_back(posRaw[indexBuffer[i]]);
		}

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
		
		constexpr int DEMO_RESOLUTION = 1024;
		image = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4);
		raytracer->bindTestImage(image.get());

		std::shared_ptr<TrivialBufferManager> bufferman = std::make_shared<TrivialBufferManager>();
		bufferman->init();

		auto imageptrv = image.get();
		auto imageptr = bufferman->createBuffer({ sizeof(image.get()) });
		bufferman->bufferData(imageptr,&imageptrv, 0, sizeof(image.get()));
		
		//DemoRayGen raygen;
		SpvVMReader reader;
		WrappedLLVMRuntimeBuilder builder;

		auto rgenCode = reader.readFile(IFRIT_ASSET_PATH"/shaders/raytracer/rtdemo.rgen.spv");
		SpvRaygenShader raygen(builder, rgenCode);
		//DemoRayGen raygen;
		DemoClosetHit hit;
		DemoMiss miss;

		raytracer->bindClosestHitShader(&hit);
		raytracer->bindRaygenShader(&raygen);
		raytracer->bindMissShader(&miss);
		raytracer->bindUniformBuffer(0, 1, imageptr);



		GLFWWindowProvider windowProvider;
		windowProvider.setup(1920, 1080);
		windowProvider.setTitle("Ifrit-v2 CPU Multithreading");

		OpenGLBackend backend;
		backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());
		windowProvider.loop([&](int* coreTime) {
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			raytracer->traceRays(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			*coreTime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			backend.updateTexture(*image);
			backend.draw();
		 });


		return 0;
	}

	int mainCpuSpirv() {
		using namespace Ifrit::Engine::ShaderVM::Spirv;
		SpvVMReader reader;
		SpvVMContext spvContext;
		reader.initializeContext(&spvContext);
		SpvVMInterpreter interpreter;
		SpvVMIntermediateRepresentation spvIR;
		
		auto fsCode = reader.readFile(IFRIT_ASSET_PATH"/shaders/raytracer/rtdemo.rgen.spv");
		reader.parseByteCode(fsCode.data(), fsCode.size() / 4, &spvContext);
		interpreter.parseRawContext(&spvContext, &spvIR);
		std::string result;
		interpreter.exportLlvmIR(&spvIR, &result);

		WrappedLLVMRuntime llvmRuntime;
		llvmRuntime.initLlvmBackend();
		llvmRuntime.loadIR(result, "raygen");

		auto symbol = llvmRuntime.lookupSymbol("ifspvm_func_1");
		return 0;
	}
}