#pragma once
#include <deque>
#include "./dependencies/GLAD/glad/glad.h"
#include "./presentation/window/WindowProvider.h"
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#else
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3native.h>

namespace Ifrit::Presentation::Window {
struct GLFWWindowProviderInitArgs{
  bool vulkanMode = false;
};
class IFRIT_APIDECL GLFWWindowProvider : public WindowProvider {
protected:
  GLFWwindow *window;
  std::deque<int> frameTimes;
  std::deque<int> frameTimesCore;
  int totalFrameTime = 0;
  int totalFrameTimeCore = 0;
  std::string title = "Ifrit-v2";
  GLFWWindowProviderInitArgs m_args;

public:
  GLFWWindowProvider() = default;
  GLFWWindowProvider(const GLFWWindowProviderInitArgs &args) : m_args(args) {}
  virtual bool setup(size_t width, size_t height) override;
  virtual void loop(const std::function<void(int *)> &func) override;
  virtual void setTitle(const std::string &title) override;

  // For Vulkan
  const char** getVkRequiredInstanceExtensions(uint32_t *count);
  void* getWindowObject();
};
} // namespace Ifrit::Presentation::Window