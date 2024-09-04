#pragma once
#include "presentation/backend/AbstractTerminalBackend.h"

namespace Ifrit::Presentation::Backend {
	class TerminalAsciiBackend :public AbstractTerminalBackend {
	private:
		int consoleWidth;
		int consoleHeight;
		constexpr static const char* ramp = R"($@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:," ^ `'. )";
		std::string resultBuffer;
	public:
		TerminalAsciiBackend(int cWid,int cHeight);
		virtual void updateTexture(const Ifrit::Core::Data::ImageF32& image) override;
		virtual void draw() override;
		virtual void setViewport(int32_t x, int32_t y, int32_t width, int32_t height) override {}
	};
}