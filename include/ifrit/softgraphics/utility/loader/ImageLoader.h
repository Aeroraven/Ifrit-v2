
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
#include "ifrit/softgraphics/core/definition/CoreExports.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader {
class ImageLoader {
public:
  void loadRGBA(const char *fileName, std::vector<float> *bufferOut,
                int *height, int *width);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader