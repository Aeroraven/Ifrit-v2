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
		std::deque<int> frameTimesCore;
		int totalFrameTime = 0;
		int totalFrameTimeCore = 0;
		std::string title = "Ifrit-v2";
	public:
		virtual bool setup(size_t width, size_t height) override;
		virtual void loop(const std::function<void(int*)>& func) override;
		virtual void setTitle(const std::string& title) override;
	};
}