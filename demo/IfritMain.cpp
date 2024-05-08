#include <iostream>
#include "presentation/window/GLFWWindowProvider.h"
#include "presentation/backend/OpenGLBackend.h"
#include "core/data/Image.h"
using namespace std;

int main() {
	using namespace Ifrit::Core::Data;
	using namespace Ifrit::Presentation::Window;
	using namespace Ifrit::Presentation::Backend;

	ImageU8 image(64, 64, 3);
	for (size_t y = 0; y < image.getHeight(); y++) {
		for (size_t x = 0; x < image.getWidth(); x++) {
			image(x, y, 0) = 255;
			image(x, y, 1) = 0;
			image(x, y, 2) = 0;
		}
	}

	GLFWWindowProvider windowProvider;
	windowProvider.setup(1920, 1080);

	OpenGLBackend backend;
	backend.setViewport(0,0,windowProvider.getWidth(), windowProvider.getHeight());
	windowProvider.loop([&]() {
		backend.updateTexture(image);
		backend.draw();
	});
	return 0;
}
