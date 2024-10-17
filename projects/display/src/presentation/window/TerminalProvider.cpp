#include "presentation/window/TerminalProvider.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
namespace Ifrit::Presentation::Window {
IFRIT_APIDECL void TerminalProvider::loop(const std::function<void(int *)> &funcs) {
  static int frameCount = 0;
  while (true) {
    int repCore;
    auto start = std::chrono::high_resolution_clock::now();
    funcs(&repCore);
    auto end = std::chrono::high_resolution_clock::now();
    using durationType = decltype(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
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
} // namespace Ifrit::Presentation::Window