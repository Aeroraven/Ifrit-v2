#include "presentation/window/GLFWWindowProvider.h"
#include <chrono>
#include <iostream>
#include <sstream>

namespace Ifrit::Presentation::Window {

void displayAssert(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << std::endl;
    exit(1);
  }
}
IFRIT_APIDECL bool GLFWWindowProvider::setup(size_t argWidth, size_t argHeight) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  if(m_args.vulkanMode){
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  }else{
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  }

  window = glfwCreateWindow(argWidth, argHeight, "Ifrit", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    displayAssert(false, "Failed to create GLFW window");
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);
  if(!m_args.vulkanMode)
    displayAssert(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress),
              "Failed to initialize GLAD");
  this->width = argWidth;
  this->height = argHeight;
  return true;
}
IFRIT_APIDECL void GLFWWindowProvider::setTitle(const std::string &titleName) {
  this->title = titleName;
}
IFRIT_APIDECL void GLFWWindowProvider::loop(const std::function<void(int *)> &funcs) {
  static int frameCount = 0;
  while (!glfwWindowShouldClose(window)) {
    int repCore;
    auto start = std::chrono::high_resolution_clock::now();
    funcs(&repCore);
    glfwSwapBuffers(window);
    glfwPollEvents();
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

    if (frameTimes.size() > 100) {
      totalFrameTime -= frameTimes.front();
      totalFrameTimeCore -= frameTimesCore.front();
      frameTimes.pop_front();
      frameTimesCore.pop_front();
    }
    frameCount++;
    frameCount %= 100;
    if (frameCount % 100 == 0) {
      std::stringstream ss;
      ss << this->title;
      ss << " [Total FPS: " << 1000.0 / (totalFrameTime / 100.0) << ",";
      ss << " Render FPS: " << 1000.0 / (totalFrameTimeCore / 100.0) << ",";
      ss << " Frame Time: " << (totalFrameTimeCore / 100.0) << "ms,";

      auto presentationTime = totalFrameTime - totalFrameTimeCore;
      ss << " Presentation Delay: " << presentationTime / 100.0 << "ms]";
      glfwSetWindowTitle(window, ss.str().c_str());
    }
  }
  glfwTerminate();
}
const char **GLFWWindowProvider::getVkRequiredInstanceExtensions(uint32_t *count) {
  return glfwGetRequiredInstanceExtensions(count);
}
void *GLFWWindowProvider::getWindowObject() {
#ifdef _WIN32
  return glfwGetWin32Window(window);
#else
  return glfwGetX11Window(window); 
#endif
}
std::pair<uint32_t,uint32_t> GLFWWindowProvider::getFramebufferSize(){
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  return {width, height};
}
} // namespace Ifrit::Presentation::Window