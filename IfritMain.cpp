#include "presentation/window/GLFWWindowProvider.h"
#include "IfritShaders.cuh"
#include "presentation/backend/OpenGLBackend.h"
#include "core/data/Image.h"
#include "engine/tileraster/TileRasterWorker.h"
#include "engine/tileraster/TileRasterRenderer.h"
#include "utility/loader/WavefrontLoader.h"
#include "engine/math/ShaderOps.h"
#include "engine/tilerastercuda/TileRasterInvocationCuda.cuh"
#include "presentation/backend/TerminalAsciiBackend.h"
#include "engine/tilerastercuda/TileRasterRendererCuda.h"

#define DEMO_RESOLUTION 2048

using namespace std;
using namespace Ifrit::Core::Data;
using namespace Ifrit::Core::CUDA;
using namespace Ifrit::Engine::TileRaster;
using namespace Ifrit::Utility::Loader;
using namespace Ifrit::Engine::Math::ShaderOps;
using namespace Ifrit::Engine::TileRaster::CUDA::Invocation;
using namespace Ifrit::Presentation::Window;
using namespace Ifrit::Presentation::Backend;
using namespace Ifrit::Engine::TileRaster::CUDA;

enum PresentEngine {
	PE_GLFW,
	PE_CONSOLE
};

PresentEngine presentEngine = PE_GLFW;
//float4x4 view = (lookAt({ 0,1.5,5.25 }, { 0,1.5,0.0 }, { 0,1,0 }));
//float4x4 view = (lookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 })); //Bunny
//float4x4 view = (lookAt({ 0,2600,2500}, { 0,0.1,-500.0 }, { 0,1,0 })); //Sponza
//float4x4 view = (lookAt({ 0,0.75,1.50}, { 0,0.75,0.0 }, { 0,1,0 })); //yomiya
//float4x4 view = (lookAt({ 0,0.0,1.25 }, { 0,0.0,0.0 }, { 0,1,0 }));
//float4x4 view = (lookAt({ 0,0.1,0.25 }, { 0,0.1,0.0 }, { 0,1,0 }));
//float4x4 view = (lookAt({ 500,300,0 }, { -100,300,-0 }, { 0,1,0 }));
//float4x4 proj = (perspective(60*3.14159/180, 1920.0 / 1080.0, 10.0, 4000));
float4x4 view = (lookAt({ 0,1.5,0 }, { -100,1.5,0 }, { 0,1,0 }));
float4x4 proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 1.0, 3000));
float4x4 model;
float4x4 mvp = multiply(proj, view);

class DemoVertexShader : public VertexShader {
public:
	IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, VaryingStore** outVaryings) override{
		auto s = *reinterpret_cast<const ifloat4*>(input[0]);
		auto p = multiply(mvp,s);
		*outPos = p;
		outVaryings[0]->vf4 = *reinterpret_cast<const ifloat4*>(input[1]);
	}
};

class DemoFragmentShader : public FragmentShader {
public:
	IFRIT_DUAL virtual void execute(const void* varyings, void* colorOutput, int stride) override {
		ifloat4 result = ((const VaryingStore*)varyings)[0].vf4;
		constexpr float fw = 0.5;
		result.x = fw * result.x + fw;
		result.y = fw * result.y + fw;
		result.z = fw * result.z + fw;
		result.w = fw * result.w + fw;

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

	loader.loadObject(IFRIT_ASSET_PATH"/sponza2.obj",pos,normal,uv,index);
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
	vertexBuffer.setValue(0, 0, ifloat4(-0.5,0.5,-0.1,1));
	vertexBuffer.setValue(1, 0, ifloat4(-0.5,-0.5,-0.1,1));
	vertexBuffer.setValue(2, 0, ifloat4(0.5,-0.5,-0.1,1));
	vertexBuffer.setValue(3, 0, ifloat4(0.5,0.5,-0.1,1));
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
	for (int i = 0; i < index.size(); i+=3) {
		indexBuffer[i/3] = index[i];
	}

	frameBuffer.setColorAttachments({ image });
	frameBuffer.setDepthAttachment(depth);
	
	renderer->init();
	renderer->bindFrameBuffer(frameBuffer);
	renderer->bindVertexBuffer(vertexBuffer);
	renderer->bindIndexBuffer(indexBuffer);
	
	DemoVertexShader vertexShader;
	VaryingDescriptor vertexShaderLayout;
	vertexShaderLayout.setVaryingDescriptors({ TypeDescriptors.FLOAT4 });
	renderer->bindVertexShader(vertexShader, vertexShaderLayout);
	DemoFragmentShader fragmentShader;
	renderer->bindFragmentShader(fragmentShader);


	float ang = 0;
	if(presentEngine==PE_CONSOLE){
		TerminalAsciiBackend backend(139, 40);
		while (true) {
			renderer->render(true);
			backend.updateTexture(*image);
			backend.draw();
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
			renderer->render(true);
			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			*coreTime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			printf("PassDone %d\n", *coreTime);
			backend.updateTexture(*image);
			backend.draw();
		});
	}
	return 0;
}

int mainGpu() {

	WavefrontLoader loader;
	std::vector<ifloat3> pos;
	std::vector<ifloat3> normal;
	std::vector<ifloat2> uv;
	std::vector<uint32_t> index;
	std::vector<ifloat3> procNormal;

	loader.loadObject(IFRIT_ASSET_PATH"/sponza2.obj", pos, normal, uv, index);
	procNormal = loader.remapNormals(normal, index, pos.size());


	std::shared_ptr<ImageF32> image1 = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 4,true);
	std::shared_ptr<ImageF32> depth = std::make_shared<ImageF32>(DEMO_RESOLUTION, DEMO_RESOLUTION, 1);
	std::shared_ptr<TileRasterRendererCuda> renderer = std::make_shared<TileRasterRendererCuda>();
	FrameBuffer frameBuffer;

	VertexBuffer vertexBuffer;
	vertexBuffer.setLayout({ TypeDescriptors.FLOAT4,TypeDescriptors.FLOAT4 });

	vertexBuffer.setVertexCount(4);
	vertexBuffer.allocateBuffer(4);
	vertexBuffer.setValue(0, 0, ifloat4(0.2, 0, -0.1, 1));
	vertexBuffer.setValue(1, 0, ifloat4(-0.5, -0.5, -0.1, 1));
	vertexBuffer.setValue(2, 0, ifloat4(0.5, -0.15, 5.51, 1));
	vertexBuffer.setValue(3, 0, ifloat4(0.5, 0.15, 5.51, 1));
	vertexBuffer.setValue(0, 1, ifloat4(0.93, 0, 0.3, 0));
	vertexBuffer.setValue(1, 1, ifloat4(0.93, 0, 0.3, 0));
	vertexBuffer.setValue(2, 1, ifloat4(0.93, 0, 0.3, 0));
	vertexBuffer.setValue(3, 1, ifloat4(0.93, 0, 0.3, 0));

	
	vertexBuffer.allocateBuffer(pos.size());

	for (int i = 0; i < pos.size(); i++) {
		vertexBuffer.setValue(i, 0, ifloat4(pos[i].x, pos[i].y, pos[i].z, 1));
		vertexBuffer.setValue(i, 1, ifloat4(procNormal[i].x, procNormal[i].y, procNormal[i].z, 0));
	}

	std::vector<int> indexBuffer = { 2,3,0 };
	
	indexBuffer.resize(index.size() /3);
	for (int i = 0; i < index.size(); i += 3) { //index.size()
		indexBuffer[i / 3] = index[i];
	}
	printf("Total Triangles:%d\n", index.size() / 3);

	frameBuffer.setColorAttachments({ image1 });
	frameBuffer.setDepthAttachment(depth);

	renderer->init();
	renderer->bindFrameBuffer(frameBuffer);
	renderer->bindVertexBuffer(vertexBuffer);
	renderer->bindIndexBuffer(indexBuffer);

	DemoVertexShaderCuda vertexShader;
	VaryingDescriptor vertexShaderLayout;
	vertexShaderLayout.setVaryingDescriptors({ TypeDescriptors.FLOAT4 });
	DemoFragmentShaderCuda fragmentShader;

	
	auto dVertexShader = vertexShader.getCudaClone();
	auto dFragmentShader = fragmentShader.getCudaClone();
	renderer->bindFragmentShader(dFragmentShader);
	renderer->bindVertexShader(dVertexShader, vertexShaderLayout);

	printf("Start\n");
	GLFWWindowProvider windowProvider;
	windowProvider.setup(1920, 1080);
	windowProvider.setTitle("Ifrit-v2 CUDA");

	OpenGLBackend backend;
	backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());
	renderer->setAggressiveRatio(1);
	windowProvider.loop([&](int* coreTime) {
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		renderer->render();
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		*coreTime = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		backend.updateTexture(*image1);
		backend.draw();
	});
	return 0;
}


int miscTest() {
	int x = 1;
	printf("%d\n", 31 - __lzcnt((x - 1) | ((1 << 9) - 1)));
	return 0;
}
int main() {
	return mainGpu();
}
