#include "DefaultDemo.h"
#include "presentation/window/GLFWWindowProvider.h"
#include "./demo/shader/DefaultDemoShaders.cuh"
#include "presentation/backend/OpenGLBackend.h"
#include "core/data/Image.h"
#include "math/LinalgOps.h"
#include "engine/tileraster/TileRasterWorker.h"
#include "engine/tileraster/TileRasterRenderer.h"
#include "utility/loader/WavefrontLoader.h"
#include "utility/loader/ImageLoader.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "presentation/backend/TerminalAsciiBackend.h"
#include "presentation/backend/TerminalCharColorBackend.h"
#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "engine/bufferman/BufferManager.h"

#define DEMO_RESOLUTION 512

namespace Ifrit::Demo::DemoDefault {
	using namespace std;
	using namespace Ifrit::Core::Data;
	using namespace Ifrit::Engine::BufferManager;
	using namespace Ifrit::Engine::TileRaster;
	using namespace Ifrit::Utility::Loader;
	using namespace Ifrit::Math;
	using namespace Ifrit::Presentation::Window;
	using namespace Ifrit::Presentation::Backend;
#ifdef IFRIT_FEATURE_CUDA
	using namespace Ifrit::Engine::TileRaster::CUDA;
	using namespace Ifrit::Core::CUDA;
	using namespace Ifrit::Engine::TileRaster::CUDA::Invocation;
#endif
	enum PresentEngine {
		PE_GLFW,
		PE_CONSOLE
	};

	PresentEngine presentEngine = PE_GLFW;
	//float4x4 view = (lookAt({ 0,1.5,5.25 }, { 0,1.5,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 })); //Bunny
	//float4x4 view = (lookAt({ 0,2600,2500}, { 0,0.1,-500.0 }, { 0,1,0 })); //Sponza
	//float4x4 view = (lookAt({ 0,0.75,1.50 }, { 0,0.75,0.0 }, { 0,1,0 })); //yomiya
	//float4x4 view = (lookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
	//float4x4 view = (lookAt({ 500,300,0 }, { -100,300,-0 }, { 0,1,0 }));
	//float4x4 proj = (perspective(60*3.14159/180, 1920.0 / 1080.0, 10.0, 4000));
	float4x4 view = (lookAt({ 0,1.5,0 }, { -100,1.5,0 }, { 0,1,0 }));
	float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 1.0, 3000));
	float4x4 model;
	float4x4 mvp = matmul(proj, view);

	float globalTime = 1.0f;

	class DemoVertexShader : public VertexShader {
	public:
		IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, ifloat4* const* outVaryings) override {
			/*const auto radius = 0.3f;
			const auto vX = sin(globalTime) * radius;
			const auto vZ = cos(globalTime) * radius;
			const auto dY = 0.1f; //sin(globalTime * 0.9f) * 0.05f + 0.1f;
			float4x4 view = (lookAt({ vX,dY,vZ }, { 0,dY,0.0 }, { 0,1,0 })); //yomiya
			float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.01, 3000));
			auto mvp = multiply(proj, view);	*/
			auto s = *reinterpret_cast<const ifloat4*>(input[0]);
			auto p = matmul(mvp, s);
			*outPos = p;
			*outVaryings[0] = *reinterpret_cast<const ifloat4*>(input[1]);
		}
	};

	class DemoFragmentShader : public FragmentShader {
	public:
		IFRIT_DUAL virtual void execute(const void* varyings, void* colorOutput, float* fragmentDepth) override {
			ifloat4 result = ((const VaryingStore*)varyings)[0].vf4;

			constexpr float fw = 0.5;
			result.x = fw * result.x + fw;
			result.y = fw * result.y + fw;
			result.z = fw * result.z + fw;
			result.w = 0.5;

			auto& co = ((ifloat4*)colorOutput)[0];
			co = result;
		}
	};


	int mainCpu() {

		WavefrontLoader loader;
		std::vector<ifloat3> pos;
		std::vector<ifloat3> normal;
		std::vector<ifloat2> uv;
		std::vector<uint32_t> index;
		std::vector<ifloat3> procNormal;

		loader.loadObject(IFRIT_ASSET_PATH"/sponza3.obj", pos, normal, uv, index);
		procNormal = loader.remapNormals(normal, index, pos.size());

		std::shared_ptr<ImageF32> image = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4);
		std::shared_ptr<ImageF32> depth = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
		std::shared_ptr<TileRasterRenderer> renderer = std::make_shared<TileRasterRenderer>();
		FrameBuffer frameBuffer;

		VertexBuffer vertexBuffer;
		vertexBuffer.setLayout({ TypeDescriptors.FLOAT4,TypeDescriptors.FLOAT4 });


		vertexBuffer.setVertexCount(4);
		vertexBuffer.allocateBuffer(4);
		//vertexBuffer.setValue(0, 0, ifloat4(-0.0027,0.3485,-0.0983,0.0026));
		//vertexBuffer.setValue(1, 0, ifloat4(0.0000,0.3294,-0.1037,-0.0037));
		//vertexBuffer.setValue(2, 0, ifloat4(0.0000,0.3487,-0.0971,-0.0028));
		vertexBuffer.setValue(0, 0, ifloat4(-0.5, 0.5, 0.1, 1));
		vertexBuffer.setValue(1, 0, ifloat4(-0.5, -0.5, 0.1, 1));
		vertexBuffer.setValue(2, 0, ifloat4(0.5, -0.5, 0.1, 1));
		vertexBuffer.setValue(3, 0, ifloat4(0.5, 0.5, 0.1, 1));
		vertexBuffer.setValue(0, 1, ifloat4(0.1, 0, 0.1, 0));
		vertexBuffer.setValue(1, 1, ifloat4(0.1, 0, 0.1, 0));
		vertexBuffer.setValue(2, 1, ifloat4(0.1, 0, 0.1, 0));
		vertexBuffer.setValue(3, 1, ifloat4(0.1, 0, 0.1, 0));



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

		frameBuffer.setColorAttachments({ image.get()});
		frameBuffer.setDepthAttachment(*depth);

		renderer->init();
		renderer->bindFrameBuffer(frameBuffer);
		renderer->bindVertexBuffer(vertexBuffer);

		std::shared_ptr<TrivialBufferManager> bufferman = std::make_shared<TrivialBufferManager>();
		bufferman->init();
		auto indexBuffer1 = bufferman->createBuffer({ sizeof(indexBuffer[0]) * indexBuffer.size() });
		bufferman->bufferData(indexBuffer1, indexBuffer.data(), 0, sizeof(indexBuffer[0]) * indexBuffer.size());

		renderer->bindIndexBuffer(indexBuffer1);
		renderer->optsetForceDeterministic(false);

		IfritColorAttachmentBlendState blendState;
		blendState.blendEnable = false;
		blendState.srcColorBlendFactor = IF_BLEND_FACTOR_SRC_ALPHA;
		blendState.dstColorBlendFactor = IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendState.srcAlphaBlendFactor = IF_BLEND_FACTOR_SRC_ALPHA;
		blendState.dstAlphaBlendFactor = IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		renderer->setBlendFunc(blendState);

		DemoVertexShader vertexShader;
		VaryingDescriptor vertexShaderLayout;
		vertexShaderLayout.setVaryingDescriptors({ TypeDescriptors.FLOAT4 });
		renderer->bindVertexShaderLegacy(vertexShader, vertexShaderLayout);
		DemoFragmentShader fragmentShader;
		renderer->bindFragmentShader(fragmentShader);

		renderer->optsetDepthTestEnable(true);

		float ang = 0;
		if (presentEngine == PE_CONSOLE) {
			TerminalCharColorBackend backend(139 * 2, 40 * 2);
			while (true) {
				renderer->drawElements(indexBuffer.size(),true);
				backend.updateTexture(*image);
				backend.draw();
				globalTime += 0.03f;
			}
		}
		else {
			ifritLog2("Start Rendering");
			GLFWWindowProvider windowProvider;
			windowProvider.setup(1920, 1080);
			windowProvider.setTitle("Ifrit-v2 CPU Multithreading");

			OpenGLBackend backend;
			backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());
			windowProvider.loop([&](int* coreTime) {
				std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
				renderer->drawElements(indexBuffer.size(), true);
				std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
				*coreTime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				//printf("PassDone %d\n", *coreTime);
				backend.updateTexture(*image);
				backend.draw();
				globalTime += 0.03f;
				//printf("PassDone %f\n", globalTime);
				});
		}
		return 0;
	}

#ifdef IFRIT_FEATURE_CUDA
	int mainGpu() {

		WavefrontLoader loader;
		std::vector<ifloat3> pos;
		std::vector<ifloat3> normal;
		std::vector<ifloat2> uv;
		std::vector<uint32_t> index;

		std::vector<ifloat3> procNormal;
		std::vector<ifloat2> procUv;

		loader.loadObject(IFRIT_ASSET_PATH"/bunny.obj", pos, normal, uv, index);
		procNormal = loader.remapNormals(normal, index, pos.size());
		//procUv = loader.remapUVs(uv, index, pos.size());

		std::shared_ptr<ImageF32> image1 = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4, true);
		std::shared_ptr<ImageF32> depth = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
		std::shared_ptr<TileRasterRendererCuda> renderer = std::make_shared<TileRasterRendererCuda>();
		FrameBuffer frameBuffer;
		VertexBuffer vertexBuffer;
		std::vector<int> indexBuffer;


		vertexBuffer.setLayout({ TypeDescriptors.FLOAT4,TypeDescriptors.FLOAT4,TypeDescriptors.FLOAT4 });
		vertexBuffer.allocateBuffer(pos.size());
		for (int i = 0; i < pos.size(); i++) {
			vertexBuffer.setValue(i, 0, ifloat4(pos[i].x, pos[i].y, pos[i].z, 1));
			vertexBuffer.setValue(i, 1, ifloat4(procNormal[i].x, procNormal[i].y, procNormal[i].z, 0));
			vertexBuffer.setValue(i, 2, ifloat4(0,0, 0, 0));
		}
		indexBuffer.resize(index.size() / 3);
		for (int i = 0; i < index.size(); i += 3) {
			indexBuffer[i / 3] = index[i];
		}

		/*
		vertexBuffer.setVertexCount(4);
		vertexBuffer.allocateBuffer(4);
		//vertexBuffer.setValue(0, 0, ifloat4(-0.0027,0.3485,-0.0983,0.0026));
		//vertexBuffer.setValue(1, 0, ifloat4(0.0000,0.3294,-0.1037,-0.0037));
		//vertexBuffer.setValue(2, 0, ifloat4(0.0000,0.3487,-0.0971,-0.0028));
		vertexBuffer.setValue(0, 0, ifloat4(-3.5, 0.0, 25.5, 1));
		vertexBuffer.setValue(1, 0, ifloat4(-3.5, 0.0, -0.5, 1));
		vertexBuffer.setValue(2, 0, ifloat4(3.5, 0.0, -0.5, 1));
		vertexBuffer.setValue(3, 0, ifloat4(3.5, 0.0, 25.5, 1));
		vertexBuffer.setValue(0, 1, ifloat4(0.1, 0, 0.1, 0));
		vertexBuffer.setValue(1, 1, ifloat4(0.1, 0, 0.1, 0));
		vertexBuffer.setValue(2, 1, ifloat4(0.1, 0, 0.1, 0));
		vertexBuffer.setValue(3, 1, ifloat4(0.1, 0, 0.1, 0));

		vertexBuffer.setValue(0, 2, ifloat4(0.0, 1.0, 0.1, 0));
		vertexBuffer.setValue(1, 2, ifloat4(0.0, 0.0, 0.1, 0));
		vertexBuffer.setValue(2, 2, ifloat4(1.0, 0.0, 0.1, 0));
		vertexBuffer.setValue(3, 2, ifloat4(1.0, 1.0, 0.1, 0));
		indexBuffer = { 2,1,0,0,3,2 };*/

		std::vector<float> texFox;
		int texFoxW, texFoxH;
		ImageLoader imageLoader;
		imageLoader.loadRGBA(IFRIT_ASSET_PATH"/fox_diffuse.png", &texFox, &texFoxH, &texFoxW);

		IfritImageCreateInfo imageCI;
		imageCI.extent.height = texFoxH;
		imageCI.extent.width = texFoxW;
		imageCI.mipLevels = 5;
		renderer->createTexture(0, imageCI);

		IfritBufferImageCopy imageCopy;
		imageCopy.imageOffset = { 0,0,0 };
		imageCopy.bufferOffset = 0;
		imageCopy.imageExtent.depth = 1;
		imageCopy.imageExtent.height = texFoxH;
		imageCopy.imageExtent.width = texFoxW;
		imageCopy.imageSubresource.baseArrayLayer = 0;
		imageCopy.imageSubresource.mipLevel = 0;
		renderer->copyHostBufferToImage(texFox.data(), 0, { imageCopy });
		renderer->generateMipmap(0, IF_FILTER_LINEAR);

		frameBuffer.setColorAttachments({ image1.get() });
		frameBuffer.setDepthAttachment(*depth);

		renderer->init();
		renderer->bindFrameBuffer(frameBuffer);
		renderer->bindVertexBuffer(vertexBuffer);
		renderer->bindIndexBuffer(indexBuffer);
		
		DemoVertexShaderCuda vertexShader;
		VaryingDescriptor vertexShaderLayout;
		vertexShaderLayout.setVaryingDescriptors({ TypeDescriptors.FLOAT4,TypeDescriptors.FLOAT4 });
		DemoFragmentShaderCuda fragmentShader;
		fragmentShader.allowDepthModification = false;
		DemoGeometryShaderCuda geometryShader;

		auto dVertexShader = vertexShader.getCudaClone();
		auto dFragmentShader = fragmentShader.getCudaClone();
		auto dGeometryShader = geometryShader.getCudaClone();
		renderer->bindFragmentShader(dFragmentShader);
		renderer->bindVertexShader(dVertexShader, vertexShaderLayout);
		//renderer->bindGeometryShader(dGeometryShader);
		//renderer->setRasterizerPolygonMode(IF_POLYGON_MODE_POINT);

		IfritSamplerT sampler;
		sampler.filterMode = IF_FILTER_LINEAR;
		sampler.addressModeU = IF_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler.addressModeV = IF_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler.borderColor = IF_BORDER_COLOR_WHITE;
		sampler.anisotropyEnable = false;
		sampler.maxAnisotropy = 16.0f;
		renderer->createSampler(0, sampler);

		IfritColorAttachmentBlendState blendState;
		blendState.blendEnable = false;
		blendState.srcColorBlendFactor = IF_BLEND_FACTOR_SRC_ALPHA;
		blendState.dstColorBlendFactor = IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendState.srcAlphaBlendFactor = IF_BLEND_FACTOR_SRC_ALPHA;
		blendState.dstAlphaBlendFactor = IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		renderer->setBlendFunc(blendState);

		renderer->setDepthFunc(IF_COMPARE_OP_LESS);
		renderer->setDepthTestEnable(true);
		renderer->setClearValues({ {0,0,0,0} }, 255.0);
		renderer->setCullMode(IF_CULL_MODE_BACK);

		
		GLFWWindowProvider windowProvider;
		windowProvider.setup(2048, 1152);
		
		stringstream ss;
		ss << "Ifrit-v2 CUDA (Resolution: " << DEMO_RESOLUTION << "x" << DEMO_RESOLUTION << ")";
		windowProvider.setTitle(ss.str().c_str());

		OpenGLBackend backend;
		backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());

		windowProvider.loop([&](int* coreTime) {
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			renderer->clear();
			renderer->drawElements();
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			*coreTime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			backend.updateTexture(*image1);
			backend.draw();

			});
		return 0;
	}
#endif


}
