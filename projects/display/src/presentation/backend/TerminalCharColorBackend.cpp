#include "ifrit/display/presentation/backend/TerminalCharColorBackend.h"
#include <iostream>
#include <sstream>

namespace Ifrit::Presentation::Backend {
IFRIT_APIDECL TerminalCharColorBackend::TerminalCharColorBackend(int cWid,
                                                                 int cHeight) {
  this->consoleWidth = cWid;
  this->consoleHeight = cHeight;
  resultBuffer = "";
}
IFRIT_APIDECL void TerminalCharColorBackend::updateTexture(const float *image,
                                                           int channels,
                                                           int width,
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

      std::stringstream ss;
      ss << "\033[38;2;" << (int)(colR * 255) << ";" << (int)(colG * 255) << ";"
         << (int)(colB * 255) << "m��\033[0m";
      res += ss.str();
    }
    res += "\n";
  }
  resultBuffer = res;
}
IFRIT_APIDECL void TerminalCharColorBackend::draw() {
  this->setCursor(0, 0, resultBuffer);
  std::cout << resultBuffer;
}

} // namespace Ifrit::Presentation::Backend