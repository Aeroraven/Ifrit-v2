#include <iostream>
#include "presentation/window/GLFWWindowProvider.h"
#include "presentation/backend/OpenGLBackend.h"
#include "core/data/Image.h"
#include "engine/tileraster/TileRasterWorker.h"
#include "engine/tileraster/TileRasterRenderer.h"
#include "utility/loader/WavefrontLoader.h"
#include "engine/math/ShaderOps.h"

using namespace std;
using namespace Ifrit::Core::Data;
using namespace Ifrit::Presentation::Window;
using namespace Ifrit::Presentation::Backend;
using namespace Ifrit::Engine::TileRaster;
using namespace Ifrit::Utility::Loader;
using namespace Ifrit::Engine::Math::ShaderOps;

float4x4 view = (lookAt({ 0,0.75,-1.0}, { 0,0.75,0 }, { 0,1,0 }));
float4x4 proj = (perspective(90*3.14159/180, 1920.0 / 1080.0, 0.1, 1000));
float4x4 mvp = multiply(proj, view);

class DemoVertexShader : public VertexShader {
public:
	void execute(const int id) override {
		auto s = vertexBuffer->getValue<float4>(id, 0);
		auto p = multiply(mvp,s);
		//p.x = -p.x;
		varyingBuffer->getPositionBuffer()[id] = p;
		varyingBuffer->getVaryingBuffer<float4>(0)[id] = vertexBuffer->getValue<float4>(id, 1);
	}
};

class DemoFragmentShader : public FragmentShader {
public:
	void execute(const std::vector<std::any>& varyings, std::vector<float4>& colorOutput) override {
		float4 result = { 1,1,1,1 };
		result.x *= 255;
		result.y *= 255;
		result.z *= 255;
		result.w *= 255;
		colorOutput[0] = result;
	}
};


int main() {
	WavefrontLoader loader;
	std::vector<float3> pos;
	std::vector<float3> normal;
	std::vector<float2> uv;
	std::vector<uint32_t> index;

	loader.loadObject(IFRIT_ASSET_PATH"/yomiya.obj",pos,normal,uv,index);


	GLFWWindowProvider windowProvider;
	windowProvider.setup(1920, 1080);

	OpenGLBackend backend;
	backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());

	std::shared_ptr<ImageU8> image = std::make_shared<ImageU8>(1600, 900, 4);
	std::shared_ptr<ImageF32> depth = std::make_shared<ImageF32>(1600, 900, 1);
	std::shared_ptr<TileRasterRenderer> renderer = std::make_shared<TileRasterRenderer>();
	FrameBuffer frameBuffer;

	VertexBuffer vertexBuffer;
	vertexBuffer.setLayout({ {sizeof(float4)},{sizeof(float4)} });
	vertexBuffer.allocateBuffer(pos.size());

	
	vertexBuffer.setValue(0, 0, float4{ -0.5,-0.5,0,1 });
	vertexBuffer.setValue(1, 0, float4{ 0.5,-1.95,0,1 });
	vertexBuffer.setValue(2, 0, float4{ 0.5,0.5,0,1 });
	vertexBuffer.setValue(3, 0, float4{ -0.5,0.5,0,1 });

	vertexBuffer.setValue(0, 1, float4{ 1,0,0,1 });
	vertexBuffer.setValue(1, 1, float4{ 0,1,0,1 });
	vertexBuffer.setValue(2, 1, float4{ 0,0,1,1 });
	vertexBuffer.setValue(3, 1, float4{ 1,1,1,1 });

	for (int i = 0; i < pos.size(); i++) {
		vertexBuffer.setValue(i, 0, float4(pos[i].x, pos[i].y, pos[i].z, 1));
		vertexBuffer.setValue(i, 1, float4(1, 0, 0, 1));
	}


	std::vector<int> indexBuffer = { 0,1,2,2,3,0 };

	indexBuffer.resize(index.size()/3);
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
	vertexShader.setVaryingDescriptors({ TypeDescriptors.FLOAT4 });
	renderer->bindVertexShader(vertexShader);
	DemoFragmentShader fragmentShader;
	renderer->bindFragmentShader(fragmentShader);

	ifritLog2("Start Rendering");
	windowProvider.loop([&]() {
		renderer->clear();
		renderer->render();
		backend.updateTexture(*image);
		backend.draw();
	});
	return 0;
}

