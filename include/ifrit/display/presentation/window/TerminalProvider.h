
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


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