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

//float4x4 view = (lookAt({ 0,0.1,0.25}, { 0,0.1,0.0 }, { 0,1,0 })); //Bunny
float4x4 view = (lookAt({ 0,3000,1500}, { 0,0.1,-500.0 }, { 0,1,0 })); //Sponza
float4x4 proj = (perspective(60*3.14159/180, 1920.0 / 1080.0, 0.1, 4000));
float4x4 mvp = multiply(proj, view);

class DemoVertexShader : public VertexShader {
public:
	void execute(const std::vector<const void*>& input, float4& outPos, std::vector<VaryingStore*>& outVaryings) override{
		auto s = *reinterpret_cast<const float4*>(input[0]);
		auto p = multiply(mvp,s);
		outPos = p;
		outVaryings[0]->vf4 = *reinterpret_cast<const float4*>(input[1]);
	}
};

class DemoFragmentShader : public FragmentShader {
public:
	void execute(const std::vector<VaryingStore>& varyings, std::vector<float4>& colorOutput) override {
		float4 result = varyings[0].vf4;
		result.x = 0.5 * result.x + 0.5;
		result.y = 0.5 * result.y + 0.5;
		result.z = 0.5 * result.z + 0.5;
		result.w = 0.5 * result.w + 0.5;

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
	std::vector<float3> procNormal;

	loader.loadObject(IFRIT_ASSET_PATH"/sponza2.obj",pos,normal,uv,index);
	procNormal = loader.remapNormals(normal, index, pos.size());

	GLFWWindowProvider windowProvider;
	windowProvider.setup(1920, 1080);

	OpenGLBackend backend;
	backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());

	std::shared_ptr<ImageU8> image = std::make_shared<ImageU8>(1600, 900, 4);
	std::shared_ptr<ImageF32> depth = std::make_shared<ImageF32>(1600, 900, 1);
	std::shared_ptr<TileRasterRenderer> renderer = std::make_shared<TileRasterRenderer>();
	FrameBuffer frameBuffer;

	VertexBuffer vertexBuffer;
	vertexBuffer.setLayout({ TypeDescriptors.FLOAT4,TypeDescriptors.FLOAT4 });
	vertexBuffer.allocateBuffer(pos.size());

	for (int i = 0; i < pos.size(); i++) {
		vertexBuffer.setValue(i, 0, float4(pos[i].x, pos[i].y, pos[i].z, 1));
		vertexBuffer.setValue(i, 1, float4(procNormal[i].x, procNormal[i].y, procNormal[i].z, 0));
	}


	std::vector<int> indexBuffer;

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

