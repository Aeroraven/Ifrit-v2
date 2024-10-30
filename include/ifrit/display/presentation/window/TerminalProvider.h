#pragma once
#include "ifrit/display/presentation/window/WindowProvider.h"
#include <deque>

namespace Ifrit::Display::Window {
class IFRIT_APIDECL TerminalProvider : public WindowProvider {
protected:
  std::deque<int> frameTimes;
  std::deque<int> frameTimesCore;
  int totalFrameTime = 0;
  int totalFrameTimeCore = 0;

public:
  virtual bool setup(size_t argWidth, size_t argHeight) override {
    return true;
  }
  virtual void loop(const std::function<void(int *)> &func) override;
  virtual void setTitle(const std::string &) override{};
};
} // namespace Ifrit::Display::Window