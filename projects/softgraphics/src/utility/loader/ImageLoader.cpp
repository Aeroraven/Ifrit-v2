
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


#include "ifrit/softgraphics/utility/loader/ImageLoader.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader {
void ImageLoader::loadRGBA(const char *fileName, std::vector<float> *bufferOut,
                           int *height, int *width) {
  int h, c, w;
  stbi_uc *data = stbi_load(fileName, &w, &h, &c, 4);
  if (c != 4) {
    // printf("ERROR\n");
    // std::abort();
  }
  if (data == nullptr) {
    printf("Cannot load image %s", fileName);
    throw std::runtime_error("Failed to load image");
  }
  bufferOut->resize(w * h * 4);
  for (int i = 0; i < w * h * 4; i++) {
    (*bufferOut)[i] = data[i] / 255.0f;
  }
  stbi_image_free(data);
  *height = h;
  *width = w;
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Utility::Loader