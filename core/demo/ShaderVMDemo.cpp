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
#include "engine/math/ShaderOps.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "presentation/backend/TerminalAsciiBackend.h"
#include "presentation/backend/TerminalCharColorBackend.h"
#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "engine/comllvmrt/WrappedLLVMRuntime.h"

using namespace std;
using namespace Ifrit::Core::Data;
using namespace Ifrit::Engine::TileRaster;
using namespace Ifrit::Utility::Loader;
using namespace Ifrit::Engine::Math::ShaderOps;
using namespace Ifrit::Presentation::Window;
using namespace Ifrit::Presentation::Backend;

using namespace Ifrit::Engine::ShaderVM::Spirv;
using namespace Ifrit::Engine::ComLLVMRuntime;

namespace Ifrit::Demo::ShaderVMDemo {

	int mainTest() {
		constexpr int DEMO_RESOLUTION = 512;
		std::shared_ptr<ImageF32> image = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4);
		std::shared_ptr<ImageF32> depth = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
		std::shared_ptr<TileRasterRenderer> renderer = std::make_shared<TileRasterRenderer>();
		FrameBuffer frameBuffer;

		VertexBuffer vertexBuffer;
		vertexBuffer.setLayout({ TypeDescriptors.FLOAT4,TypeDescriptors.FLOAT4 });

		vertexBuffer.setVertexCount(3);
		vertexBuffer.allocateBuffer(3);
		vertexBuffer.setValue(0, 0, ifloat4(0, 0.5, 0.1, 1));
		vertexBuffer.setValue(1, 0, ifloat4(-0.5, -0.5, 0.1, 1));
		vertexBuffer.setValue(2, 0, ifloat4(0.5, -0.5, 0.1, 1));
		vertexBuffer.setValue(0, 1, ifloat4(1, 0, 0, 0));
		vertexBuffer.setValue(1, 1, ifloat4(0, 1, 0, 0));
		vertexBuffer.setValue(2, 1, ifloat4(0, 0, 1, 0));

		std::vector<int> indexBuffer = { 0,1,2 };

		frameBuffer.setColorAttachments({ image.get() });
		frameBuffer.setDepthAttachment(*depth);

		renderer->init();
		renderer->bindFrameBuffer(frameBuffer);
		renderer->bindVertexBuffer(vertexBuffer);
		renderer->bindIndexBuffer(indexBuffer);
		renderer->optsetForceDeterministic(true);

		struct Uniform {
			ifloat4 t1 = { 0,0,0,0 };
			ifloat4 t2 = { 0.1,0,0,0 };
		} uniform;
		struct Uniform2 {
			float mat[16] = {
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0.5, 0, 0, 1
			};
		} uniform2;
		renderer->bindUniformBuffer(0, 0, &uniform);
		renderer->bindUniformBuffer(1, 0, &uniform2);


		WrappedLLVMRuntime::initLlvmBackend();

		SpvVMReader reader;
		auto fsCode = reader.readFile(IFRIT_ASSET_PATH"/shaders/demo.frag.hlsl.spv");
		auto vsCode = reader.readFile(IFRIT_ASSET_PATH"/shaders/demo.vert.hlsl.spv");
		
		WrappedLLVMRuntime fsRuntime, vsRuntime;
		SpvVertexShader vertexShader(&vsRuntime, vsCode);
		renderer->bindVertexShader(vertexShader);
		SpvFragmentShader fragmentShader(&fsRuntime,fsCode);
		renderer->bindFragmentShader(fragmentShader);

		GLFWWindowProvider windowProvider;
		windowProvider.setup(1920, 1080);
		windowProvider.setTitle("Ifrit-v2 CPU Multithreading");

		OpenGLBackend backend;
		backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());
		windowProvider.loop([&](int* coreTime) {
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			renderer->render(true);
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			*coreTime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			backend.updateTexture(*image);
			backend.draw();
		});
	}
}