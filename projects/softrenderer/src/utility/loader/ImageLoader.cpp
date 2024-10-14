#include "utility/loader/ImageLoader.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

namespace Ifrit::Utility::Loader {
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
} // namespace Ifrit::Utility::Loader