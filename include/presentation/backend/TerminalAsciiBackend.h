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
		void updateTexture(const Ifrit::Core::Data::ImageF32& image) override;
		void draw();
	};
}