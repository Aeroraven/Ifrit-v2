
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


#include "ifrit/display/presentation/window/TerminalProvider.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace Ifrit::Display::Window {
IFRIT_APIDECL void
TerminalProvider::loop(const std::function<void(int *)> &funcs) {
  static int frameCount = 0;
  while (true) {
    int repCore;
    auto start = std::chrono::high_resolution_clock::now();
    funcs(&repCore);
    auto end = std::chrono::high_resolution_clock::now();
    using durationType =
        decltype(std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count());
    frameTimes.push_back(std::max(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count(),
        static_cast<durationType>(1ll)));
    frameTimesCore.push_back(repCore);

    totalFrameTime += frameTimes.back();
    totalFrameTimeCore += frameTimesCore.back();

    if (frameTimes.size() > 150) {
      totalFrameTime -= frameTimes.front();
      totalFrameTimeCore -= frameTimesCore.front();
      frameTimes.pop_front();
      frameTimesCore.pop_front();
    }
    frameCount++;
    frameCount %= 150;

    std::stringstream ss;
    ss << " Total FPS: " << std::setw(10) << 1000.0 / (totalFrameTime / 150.0)
       << ",";
    ss << " Render FPS: " << std::setw(10)
       << 1000.0 / (totalFrameTimeCore / 150.0) << ",";

    auto presentationTime = totalFrameTime - totalFrameTimeCore;
    ss << " Presentation Delay: " << std::setw(10) << presentationTime / 150.0
       << "ms";
    std::cout << "\n" << ss.str() << std::endl;
  }
}
} // namespace Ifrit::Display::Window