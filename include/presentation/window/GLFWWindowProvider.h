#pragma once
#include "./dependency/GLAD/glad/glad.h"
#include "./core/definition/CoreDefs.h"
#include "./presentation/window/WindowProvider.h"
#include <GLFW/glfw3.h>

namespace Ifrit::Presentation::Window {
	class GLFWWindowProvider:public WindowProvider {
	protected:
		GLFWwindow* window;
		std::deque<int> frameTimes;
		int totalFrameTime = 0;
	public:
		virtual bool setup(size_t width, size_t height);
		void loop(const std::function<void()>& func);
	};
}