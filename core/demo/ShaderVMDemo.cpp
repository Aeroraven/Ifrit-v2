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
		float4x4 view = (lookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
		float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.1, 3000));
		float4x4 mvp = transpose(multiply(proj, view));

		WavefrontLoader loader;
		std::vector<ifloat3> pos;
		std::vector<ifloat3> normal;
		std::vector<ifloat2> uv;
		std::vector<uint32_t> index;
		std::vector<ifloat3> procNormal;
		loader.loadObject(IFRIT_ASSET_PATH"/bunny.obj", pos, normal, uv, index);
		procNormal = loader.remapNormals(normal, index, pos.size());


		constexpr int DEMO_RESOLUTION = 2048;
		std::shared_ptr<ImageF32> image = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4);
		std::shared_ptr<ImageF32> depth = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
		std::shared_ptr<TileRasterRenderer> renderer = std::make_shared<TileRasterRenderer>();
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

		frameBuffer.setColorAttachments({ image.get() });
		frameBuffer.setDepthAttachment(*depth);

		renderer->init();
		renderer->bindFrameBuffer(frameBuffer);
		renderer->bindVertexBuffer(vertexBuffer);
		renderer->bindIndexBuffer(indexBuffer);
		renderer->optsetForceDeterministic(true);

		struct Uniform {
			ifloat4 t1 = { 0,0,0,0 };
			ifloat4 t2 = { 0,0,0,0 };
		} uniform;

		renderer->bindUniformBuffer(0, 0, &uniform);
		renderer->bindUniformBuffer(1, 0, &mvp);

		SpvVMReader reader;
		auto fsCode = reader.readFile(IFRIT_ASSET_PATH"/shaders/demo.frag.hlsl.spv");
		auto vsCode = reader.readFile(IFRIT_ASSET_PATH"/shaders/demo.vert.hlsl.spv");
		
		WrappedLLVMRuntimeBuilder llvmRuntime;
		SpvVertexShader vertexShader(llvmRuntime, vsCode);
		renderer->bindVertexShader(vertexShader);
		SpvFragmentShader fragmentShader(llvmRuntime,fsCode);
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