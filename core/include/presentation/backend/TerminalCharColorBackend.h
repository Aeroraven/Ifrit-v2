#pragma once
#include "presentation/backend/AbstractTerminalBackend.h"

namespace Ifrit::Presentation::Backend {
	class TerminalCharColorBackend :public AbstractTerminalBackend {
	private:
		int consoleWidth;
		int consoleHeight;
		std::string resultBuffer;
	public:
		TerminalCharColorBackend(int cWid, int cHeight);
		virtual void updateTexture(const Ifrit::Core::Data::ImageF32& image) override;
		virtual void draw() override;
		virtual void setViewport(int32_t , int32_t , int32_t , int32_t ) override {}
	};
}