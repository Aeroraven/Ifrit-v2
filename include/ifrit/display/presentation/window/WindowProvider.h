#pragma once
#include "ifrit/common/util/ApiConv.h"
#include <cstdint>
#include <functional>
#include <string>


namespace Ifrit::Display::Window {
class IFRIT_APIDECL WindowProvider {
protected:
  size_t width;
  size_t height;

public:
  virtual ~WindowProvider() = default;
  virtual bool setup(size_t width, size_t height) = 0;
  virtual size_t getWidth() const;
  virtual size_t getHeight() const;
  virtual void loop(const std::function<void(int *)> &func) = 0;
  virtual void setTitle(const std::string &title) = 0;
  virtual const char **getVkRequiredInstanceExtensions(uint32_t *count) {
    return nullptr;
  };
  virtual void *getWindowObject() { return nullptr; };
  virtual void *getGLFWWindow() { return nullptr; };
};
} // namespace Ifrit::Display::Window