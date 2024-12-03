
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
#include "ifrit/display/dependencies/GLAD/glad/glad.h"
#include "ifrit/display/presentation/window/WindowProvider.h"
#include <deque>

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#else
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3native.h>

namespace Ifrit::Display::Window {
struct GLFWWindowProviderInitArgs {
  bool vulkanMode = false;
};
class IFRIT_APIDECL GLFWWindowProvider : public WindowProvider {
protected:
  GLFWwindow *window = nullptr;
  std::deque<int> frameTimes;
  std::deque<int> frameTimesCore;
  int totalFrameTime = 0;
  int totalFrameTimeCore = 0;
  std::string title = "Ifrit-v2";
  GLFWWindowProviderInitArgs m_args;
  std::function<void(int, int, int, int)> keyCallBack;

public:
  GLFWWindowProvider() = default;
  GLFWWindowProvider(const GLFWWindowProviderInitArgs &args) : m_args(args) {}
  virtual bool setup(size_t width, size_t height) override;
  virtual void loop(const std::function<void(int *)> &func) override;
  virtual void setTitle(const std::string &title) override;

  // For Vulkan
  const char **getVkRequiredInstanceExtensions(uint32_t *count) override;
  void *getWindowObject() override;
  std::pair<uint32_t, uint32_t> getFramebufferSize();
  void *getGLFWWindow() override;

  void callGlfwInit() {
    auto x = glfwInit();
    if (!x) {
      printf("GLFW fails\n");
    }
    printf("GLFW init:%d\n", x);
  }
  virtual void
      registerKeyCallback(std::function<void(int, int, int, int)>) override;
  inline std::function<void(int, int, int, int)> getKeyCallBack() {
    return keyCallBack;
  }
};
} // namespace Ifrit::Display::Window