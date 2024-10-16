#include "presentation/backend/TerminalCharColorBackend.h"

namespace Ifrit::Presentation::Backend {
TerminalCharColorBackend::TerminalCharColorBackend(int cWid, int cHeight) {
  this->consoleWidth = cWid;
  this->consoleHeight = cHeight;
  resultBuffer = "";
}
void TerminalCharColorBackend::updateTexture(
    const Ifrit::Engine::SoftRenderer::Core::Data::ImageF32 &image) {
  std::string res = "";
  for (int i = consoleHeight - 1; i >= 0; i--) {
    for (int j = 0; j < consoleWidth; j++) {
      int samplePtX = (int)(j * (image.getWidth() / (float)consoleWidth));
      int samplePtY = (int)(i * (image.getHeight() / (float)consoleHeight));
      auto colR = image(samplePtX, samplePtY, 0);
      auto colG = image(samplePtX, samplePtY, 1);
      auto colB = image(samplePtX, samplePtY, 2);

      std::stringstream ss;
      ss << "\033[38;2;" << (int)(colR * 255) << ";" << (int)(colG * 255) << ";"
         << (int)(colB * 255) << "m¨€\033[0m";
      res += ss.str();
    }
    res += "\n";
  }
  resultBuffer = res;
}
void TerminalCharColorBackend::draw() {
  this->setCursor(0, 0, resultBuffer);
  std::cout << resultBuffer;
}

} // namespace Ifrit::Presentation::Backend