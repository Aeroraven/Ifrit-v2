#pragma once
#include "ifrit/display/presentation/backend/AbstractTerminalBackend.h"

namespace Ifrit::Display::Backend {
class IFRIT_APIDECL TerminalAsciiBackend : public AbstractTerminalBackend {
private:
  int consoleWidth;
  int consoleHeight;
  constexpr static const char *ramp =
      R"($@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:," ^ `'. )";
  std::string resultBuffer;

public:
  TerminalAsciiBackend(int cWid, int cHeight);
  virtual void updateTexture(const float *image, int channels, int width,
                             int height) override;
  virtual void draw() override;
  virtual void setViewport(int32_t, int32_t, int32_t, int32_t) override {}
};
} // namespace Ifrit::Display::Backend