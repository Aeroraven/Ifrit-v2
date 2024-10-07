#include "engine/shadervm/spirv/SpvVMReader.h"
#include "engine/shadervm/spirv/SpvVMInterpreter.h"
#include "engine/shadervm/spirv/SpvVMShader.h"
#include "ShaderVMDemo.h"
#include "engine/comllvmrt/WrappedLLVMRuntime.h"
#include "presentation/window/GLFWWindowProvider.h"
#include "./demo/shader/DefaultDemoShaders.cuh"
#include "presentation/backend/OpenGLBackend.h"
#include "core/data/Image.h"
#include "engine/tileraster/TileRasterWorker.h"
#include "engine/tileraster/TileRasterRenderer.h"
#include "utility/loader/WavefrontLoader.h"
#include "utility/loader/ImageLoader.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "presentation/backend/TerminalAsciiBackend.h"
#include "presentation/backend/TerminalCharColorBackend.h"
#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "engine/comllvmrt/WrappedLLVMRuntime.h"
#include "engine/bufferman/BufferManager.h"
#include "engine/shadervm/spirvvec/SpvMdQuadIRGenerator.h"
#include "engine/shadervm/spirvvec/SpvMdShader.h"
#include "math/LinalgOps.h"

using namespace std;
using namespace Ifrit::Core::Data;
using namespace Ifrit::Engine::TileRaster;
using namespace Ifrit::Utility::Loader;
#ifdef IFRIT_FEATURE_CUDA
using namespace Ifrit::Engine::Math::ShaderOps;
#endif
using namespace Ifrit::Presentation::Window;
using namespace Ifrit::Presentation::Backend;
using namespace Ifrit::Math;
using namespace Ifrit::Engine::ShaderVM::Spirv;
using namespace Ifrit::Engine::ShaderVM::SpirvVec;
using namespace Ifrit::Engine::ComLLVMRuntime;
using namespace Ifrit::Engine::BufferManager;

namespace Ifrit::Demo::ShaderVMDemo {

	int mainTest2() {
		SpvVMReader reader;
		SpvVMContext ctx;
		reader.initializeContext(&ctx);
		auto cv = reader.readFile(IFRIT_ASSET_PATH"/shaders/diffuse.frag.hlsl.spv");
		reader.parseByteCode(cv.data(), cv.size() / 4, &ctx);

		SpVcQuadGroupedIRGenerator intp;
		SpVcVMGeneratorContext irctx;
		intp.bindBytecode(&ctx, &irctx);
		intp.init();

		intp.parse();
		intp.verbose();

		auto ir = intp.generateIR();
		WrappedLLVMRuntimeBuilder builder;
		auto ax = builder.buildRuntime();
		ax->loadIR(ir, "identifier");
		return 0;
	}

	int mainTest() {
		//float4x4 view = (lookAt({ 0,0.01,0.02 }, { 0,0.01,0.0 }, { 0,1,0 }));
		//float4x4 view = (lookAt({ 0,0.75,1.50 }, { 0,0.75,0.0 }, { 0,1,0 }));
		//float4x4 view = (lookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
		//float4x4 view = (lookAt({ 0,1.95,1.50 }, { 0,0.95,0.0 }, { 0,1,0 }));
		//float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.1, 3000));
		//float4x4 view = (lookAt({ 0,60.0,130.0 }, { 0,60.0,0.0 }, { 0,1,0 }));

		float4x4 view = (lookAt({ 500,300,0 }, { -100,300,-0 }, { 0,1,0 }));
		float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 60.1, 3000));

		float4x4 mvp = transpose(matmul(proj, view));

		WavefrontLoader loader;
		std::vector<ifloat3> pos;
		std::vector<ifloat3> normal;
		std::vector<ifloat2> uv;
		std::vector<uint32_t> index;
		std::vector<ifloat3> procNormal;
		loader.loadObject(IFRIT_ASSET_PATH"/sponza.obj", pos, normal, uv, index);
		procNormal = loader.remapNormals(normal, index, pos.size());


		constexpr int DEMO_RESOLUTION_X = 2048;
		constexpr int DEMO_RESOLUTION_Y = 2048;
		std::shared_ptr<ImageF32> image = std::make_shared<ImageF32>(DEMO_RESOLUTION_X, DEMO_RESOLUTION_Y, 4);
		std::shared_ptr<ImageF32> depth = std::make_shared<ImageF32>(DEMO_RESOLUTION_X, DEMO_RESOLUTION_Y, 1);
		std::shared_ptr<TileRasterRenderer> renderer = std::make_shared<TileRasterRenderer>();
		std::shared_ptr<TrivialBufferManager> bufferman = std::make_shared<TrivialBufferManager>();
		bufferman->init();
		FrameBuffer frameBuffer;

		VertexBuffer vertexBuffer;
		vertexBuffer.setLayout({ TypeDescriptors.FLOAT4,TypeDescriptors.FLOAT4 });
		vertexBuffer.allocateBuffer(pos.size());
		for (int i = 0; i < pos.size(); i++) {
			vertexBuffer.setValue(i, 0, ifloat4(pos[i].x, pos[i].y, pos[i].z, 1));
			vertexBuffer.setValue(i, 1, ifloat4(procNormal[i].x, procNormal[i].y, procNormal[i].z, 0));
		}

		std::vector<int> indexBuffer = { 0,1,2,2,3,0 };
		indexBuffer.resize(index.size() / 3);
		for (int i = 0; i < index.size(); i += 3) {
			indexBuffer[i / 3] = index[i];
		}
		printf("Num Triangles: %d\n", indexBuffer.size() / 3);

		frameBuffer.setColorAttachments({ image.get() });
		frameBuffer.setDepthAttachment(*depth);

		renderer->init();
		renderer->bindFrameBuffer(frameBuffer);
		renderer->bindVertexBuffer(vertexBuffer);
		
		renderer->optsetForceDeterministic(false);
		renderer->optsetDepthTestEnable(true);

		struct Uniform {
			ifloat4 t1 = { 1,1,1,0 };
			ifloat4 t2 = { 0.0,0,0,0 };
		} uniform;

		auto uniform1 = bufferman->createBuffer({ sizeof(uniform) });
		bufferman->bufferData(uniform1, &uniform, 0, sizeof(uniform));
		auto uniform2 = bufferman->createBuffer({ sizeof(mvp) });
		bufferman->bufferData(uniform2, &mvp, 0, sizeof(mvp));
		auto indexBuffer1 = bufferman->createBuffer({ sizeof(indexBuffer[0]) * indexBuffer.size() });
		bufferman->bufferData(indexBuffer1, indexBuffer.data(), 0, sizeof(indexBuffer[0]) * indexBuffer.size());

		renderer->bindUniformBuffer(0, 0, uniform1);
		renderer->bindUniformBuffer(1, 0, uniform2);
		renderer->bindIndexBuffer(indexBuffer1);

		SpvVMReader reader;
		auto fsCode = reader.readFile(IFRIT_ASSET_PATH"/shaders/demo.frag.hlsl.spv");
		auto vsCode = reader.readFile(IFRIT_ASSET_PATH"/shaders/demo.vert.hlsl.spv");
		
		WrappedLLVMRuntimeBuilder llvmRuntime;
		SpvVertexShader vertexShader(llvmRuntime, vsCode);
		renderer->bindVertexShader(vertexShader);
		SpvVecFragmentShader fragmentShader(llvmRuntime, fsCode);
		//SpvFragmentShader fragmentShader(llvmRuntime,fsCode);
		renderer->bindFragmentShader(fragmentShader);

		GLFWWindowProvider windowProvider;
		windowProvider.setup(1920, 1080);
		windowProvider.setTitle("Ifrit-v2 CPU Multithreading");

		OpenGLBackend backend;
		backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());
		windowProvider.loop([&](int* coreTime) {
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			uniform.t1.x = 0.4f*std::sin((float)std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count() / 1000.0f);
			bufferman->bufferData(uniform1, &uniform, 0, sizeof(uniform));
			renderer->drawElements(indexBuffer.size(),true);
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			*coreTime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			backend.updateTexture(*image);
			backend.draw();
		});
		return 0;
	}
}