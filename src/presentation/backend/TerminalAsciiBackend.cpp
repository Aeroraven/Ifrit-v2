#include "presentation/backend/TerminalAsciiBackend.h"


namespace Ifrit::Presentation::Backend {
	TerminalAsciiBackend::TerminalAsciiBackend(int cWid, int cHeight) {
		this->consoleWidth = cWid;
		this->consoleHeight = cHeight;
	}
	void TerminalAsciiBackend::updateTexture(const Ifrit::Core::Data::ImageF32& image) {
		std::string res = "";
		for (int i = consoleHeight-1; i >=0; i--) {
			for (int j = 0; j < consoleWidth; j++) {
				int samplePtX = (int)(j * (image.getWidth() / (float)consoleWidth));
				int samplePtY = (int)(i * (image.getHeight() / (float)consoleHeight));
				auto colR = image(samplePtX, samplePtY, 0);
				auto colG = image(samplePtX, samplePtY, 1);
				auto colB = image(samplePtX, samplePtY, 2);
				auto luminance = 1-(0.2126 * colR + 0.7152 * colG + 0.0722 * colB);
				luminance = (luminance * 71 + 0.5);
				auto luminInt = (int)luminance;

				res += ramp[luminInt];
			}
			res += "\n";
		}
		resultBuffer = res;
	}
	void TerminalAsciiBackend::draw() {
		this->setCursor(0, 0, resultBuffer);
		std::cout << resultBuffer;
	}

}