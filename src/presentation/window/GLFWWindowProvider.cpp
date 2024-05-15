#include "presentation/window/GLFWWindowProvider.h"
#include "core/definition/CoreExports.h"

namespace Ifrit::Presentation::Window {
	bool GLFWWindowProvider::setup(size_t width, size_t height) {
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

		window = glfwCreateWindow(width, height, "Ifrit", nullptr, nullptr);
		if (!window) {
			glfwTerminate();
			ifritAssert(false, "Failed to create GLFW window");
			return false;
		}
		glfwMakeContextCurrent(window);
		ifritAssert(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress), "Failed to initialize GLAD");

		this->width = width;
		this->height = height;
		return true;
	}
	void GLFWWindowProvider::loop(const std::function<void()>& funcs) {
		static int frameCount = 0;
		while (!glfwWindowShouldClose(window)) {
			auto start = std::chrono::high_resolution_clock::now();
			funcs();
			glfwSwapBuffers(window);
			glfwPollEvents();
			auto end = std::chrono::high_resolution_clock::now();
			frameTimes.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
			totalFrameTime += frameTimes.back();
			if (frameTimes.size() > 100) {
				frameTimes.pop_front();
				totalFrameTime -= frameTimes.front();
			}
			frameCount++;
			frameCount %= 100;
			if (frameCount % 100 == 0) {
				std::stringstream ss;
				ss << "Ifrit-V2";
				ss << " [FPS: " << 1000.0 / (totalFrameTime / 100.0)<<"]";
				glfwSetWindowTitle(window, ss.str().c_str());
			}
		}
		glfwTerminate();
	}
}