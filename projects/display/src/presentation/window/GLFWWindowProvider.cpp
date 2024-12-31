
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

#include "ifrit/display/presentation/window/GLFWWindowProvider.h"
#include "ifrit/common/util/TypingUtil.h"
#include <chrono>
#include <iostream>
#include <sstream>
#ifndef _WIN32
#include <X11/Xlib.h>
#endif
namespace Ifrit::Display::Window {

void displayAssert(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << std::endl;
    exit(1);
  }
}
IFRIT_APIDECL bool GLFWWindowProvider::setup(size_t argWidth,
                                             size_t argHeight) {
  callGlfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  if (m_args.vulkanMode) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  } else {
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  }
  using Ifrit::Common::Utility::size_cast;
  window = glfwCreateWindow(size_cast<int>(argWidth), size_cast<int>(argHeight),
                            "Ifrit", nullptr, nullptr);
  glfwSetWindowUserPointer(window, this);
  auto keyFunc = [](GLFWwindow *window, int key, int scancode, int action,
                    int mods) {
    auto s =
        static_cast<GLFWWindowProvider *>(glfwGetWindowUserPointer(window));
    auto func = s->getKeyCallBack();
    if (func) {
      func(key, scancode, action, mods);
    }
  };
  glfwSetKeyCallback(window, keyFunc);
  if (!window) {
    glfwTerminate();
    displayAssert(false, "Failed to create GLFW window");
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);
  if (!m_args.vulkanMode)
    displayAssert(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress),
                  "Failed to initialize GLAD");
  this->width = argWidth;
  this->height = argHeight;
  return true;
}
IFRIT_APIDECL void GLFWWindowProvider::setTitle(const std::string &titleName) {
  this->title = titleName;
}
IFRIT_APIDECL void
GLFWWindowProvider::loop(const std::function<void(int *)> &funcs) {
  static int frameCount = 0;
  while (!glfwWindowShouldClose(window)) {
    int repCore = -1;
    auto start = std::chrono::high_resolution_clock::now();
    funcs(&repCore);

    auto end = std::chrono::high_resolution_clock::now();
    using durationType =
        decltype(std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count());
    frameTimes.push_back(Ifrit::Common::Utility::size_cast<int>(std::max(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count(),
        static_cast<durationType>(1ll))));
    if (repCore != -1)
      frameTimesCore.push_back(repCore);
    else
      frameTimesCore.push_back(1);

    totalFrameTime += frameTimes.back();
    totalFrameTimeCore += frameTimesCore.back();

    if (frameTimes.size() > 100) {
      totalFrameTime -= frameTimes.front();
      totalFrameTimeCore -= frameTimesCore.front();
      frameTimes.pop_front();
      frameTimesCore.pop_front();
    }
    frameCount++;
    frameCount %= 100;
    if (frameCount % 100 == 0 && repCore != -1) {
      std::stringstream ss;
      ss << this->title;
      ss << " [Total FPS: " << 1000.0 / (totalFrameTime / 100.0) << ",";
      ss << " Render FPS: " << 1000.0 / (totalFrameTimeCore / 100.0) << ",";
      ss << " Frame Time: " << (totalFrameTimeCore / 100.0) << "ms,";

      auto presentationTime = totalFrameTime - totalFrameTimeCore;
      ss << " Presentation Delay: " << presentationTime / 100.0 << "ms]";
      glfwSetWindowTitle(window, ss.str().c_str());
    } else if (frameCount % 100 == 0) {
      std::stringstream ss;
      ss << this->title;
      ss << " [Total FPS: " << 1000.0 / (totalFrameTime / 100.0) << ",";
      ss << " Frame Time: " << (totalFrameTime / 100.0) << "ms]";
      glfwSetWindowTitle(window, ss.str().c_str());
    }
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glfwTerminate();
}
IFRIT_APIDECL const char **
GLFWWindowProvider::getVkRequiredInstanceExtensions(uint32_t *count) {
  return glfwGetRequiredInstanceExtensions(count);
}
IFRIT_APIDECL void *GLFWWindowProvider::getWindowObject() {
#ifdef _WIN32
  return glfwGetWin32Window(window);
#else

  static_assert((sizeof(unsigned long long) == sizeof(void *)),
                "Window size is not equal to void* size");
  return (void *)glfwGetX11Window(window);
#endif
}

IFRIT_APIDECL void GLFWWindowProvider::registerKeyCallback(
    std::function<void(int, int, int, int)> x) {
  keyCallBack = x;
}

IFRIT_APIDECL std::pair<uint32_t, uint32_t>
GLFWWindowProvider::getFramebufferSize() {
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  return {width, height};
}
IFRIT_APIDECL void *GLFWWindowProvider::getGLFWWindow() { return window; }
} // namespace Ifrit::Display::Window