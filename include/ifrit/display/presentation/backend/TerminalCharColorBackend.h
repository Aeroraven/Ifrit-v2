#pragma once
#include "ifrit/display/presentation/backend/AbstractTerminalBackend.h"

namespace Ifrit::Presentation::Backend {
class IFRIT_APIDECL TerminalCharColorBackend : public AbstractTerminalBackend {
private:
  int consoleWidth;
  int consoleHeight;
  std::string resultBuffer;

public:
  TerminalCharColorBackend(int cWid, int cHeight);
  virtual void updateTexture(const float *image, int channels, int width,
                             int height) override;
  virtual void draw() override;
  virtual void setViewport(int32_t, int32_t, int32_t, int32_t) override {}
};
} // namespace Ifrit::Presentation::Backend