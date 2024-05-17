#pragma once
#include "./core/definition/CoreDefs.h"
#include "./presentation/window/WindowProvider.h"

namespace Ifrit::Presentation::Window {
	class TerminalProvider :public WindowProvider {
	protected:
		std::deque<int> frameTimes;
		std::deque<int> frameTimesCore;
		int totalFrameTime = 0;
		int totalFrameTimeCore = 0;
	public:
		virtual bool setup(size_t width, size_t height) { return true; }
		void loop(const std::function<void(int*)>& func);
	};
}