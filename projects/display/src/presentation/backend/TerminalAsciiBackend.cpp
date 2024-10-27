#include "ifrit/display/presentation/backend/TerminalAsciiBackend.h"
#include <iomanip>
#include <iostream>

namespace Ifrit::Presentation::Backend {
IFRIT_APIDECL TerminalAsciiBackend::TerminalAsciiBackend(int cWid,
                                                         int cHeight) {
  this->consoleWidth = cWid;
  this->consoleHeight = cHeight;
}
IFRIT_APIDECL void TerminalAsciiBackend::updateTexture(const float *image,
                                                       int channels, int width,
                                                       int height) {
  std::string res = "";
  for (int i = consoleHeight - 1; i >= 0; i--) {
    for (int j = 0; j < consoleWidth; j++) {
      int samplePtX = (int)(j * (width / (float)consoleWidth));
      int samplePtY = (int)(i * (height / (float)consoleHeight));
      auto colR =
          image[samplePtX * width * channels + samplePtY * channels + 0];
      auto colG =
          image[samplePtX * width * channels + samplePtY * channels + 1];
      auto colB =
          image[samplePtX * width * channels + samplePtY * channels + 2];
      auto luminance = 1 - (0.2126 * colR + 0.7152 * colG + 0.0722 * colB);
      luminance = (luminance * 71 + 0.5);
      auto luminInt = (int)luminance;

      res += ramp[luminInt];
    }
    res += "\n";
  }
  resultBuffer = res;
}
IFRIT_APIDECL void TerminalAsciiBackend::draw() {
  this->setCursor(0, 0, resultBuffer);
  std::cout << resultBuffer;
}

} // namespace Ifrit::Presentation::Backend