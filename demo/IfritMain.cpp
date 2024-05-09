#include <iostream>
#include "presentation/window/GLFWWindowProvider.h"
#include "presentation/backend/OpenGLBackend.h"
#include "core/data/Image.h"
#include "engine/tileraster/TileRasterWorker.h"
#include "engine/tileraster/TileRasterRenderer.h"

using namespace std;
using namespace Ifrit::Core::Data;
using namespace Ifrit::Presentation::Window;
using namespace Ifrit::Presentation::Backend;
using namespace Ifrit::Engine::TileRaster;


class DemoVertexShader : public VertexShader {
public:
	void execute(const int id) override {
		varyingBuffer->getPositionBuffer()[id] = vertexBuffer->getValue<float4>(id, 0);
	}
};


int main() {
	GLFWWindowProvider windowProvider;
	windowProvider.setup(1920, 1080);
	OpenGLBackend backend;
	backend.setViewport(0, 0, windowProvider.getWidth(), windowProvider.getHeight());

	std::shared_ptr<ImageU8> image = std::make_shared<ImageU8>(800, 600, 4);
	std::shared_ptr<TileRasterRenderer> renderer = std::make_shared<TileRasterRenderer>();
	FrameBuffer frameBuffer;

	VertexBuffer vertexBuffer;
	vertexBuffer.setLayout({ {sizeof(float4)} });
	vertexBuffer.allocateBuffer(3);

	vertexBuffer.setValue<float4>(0, 0, { 0,1,0 });
	vertexBuffer.setValue<float4>(1, 0, { -1,0,0 });
	vertexBuffer.setValue<float4>(2, 0, { 1,0,0 });

	std::vector<int> indexBuffer = { 0,1,2 };

	frameBuffer.setColorAttachments({ image });
	renderer->init();
	renderer->bindFrameBuffer(frameBuffer);
	renderer->bindVertexBuffer(vertexBuffer);
	renderer->bindIndexBuffer(indexBuffer);
	
	DemoVertexShader vertexShader;
	renderer->bindVertexShader(vertexShader);

	windowProvider.loop([&]() {
		renderer->clear();
		renderer->render();
		backend.updateTexture(*image);
		backend.draw();
	});
	return 0;
}
