
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

namespace Ifrit::Display::Backend {
class IFRIT_APIDECL BackendProvider {
public:
  virtual ~BackendProvider() = default;
  virtual void draw() = 0;
  virtual void updateTexture(const float *image, int channels, int width,
                             int height) = 0;
  virtual void setViewport(int32_t x, int32_t y, int32_t width,
                           int32_t height) = 0;
};
} // namespace Ifrit::Display::Backend