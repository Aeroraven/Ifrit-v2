
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
  virtual void registerKeyCallback(std::function<void(int, int, int, int)>){}
};
} // namespace Ifrit::Display::Window