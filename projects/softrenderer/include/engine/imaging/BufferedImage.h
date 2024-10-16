#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/Structures.h"

namespace Ifrit::Engine::SoftRenderer::Imaging {

enum class BufferedImageTiling { TILING_LINEAR, TILING_TILED };
enum class BufferedImageFormat {
  FORMAT_R8G8B8A8_UINT,
  FORMAT_R32G32B32A32_SFLOAT
};

struct BufferedImageConfig {
  static constexpr uint32_t TILE_SIZE = 16;
};

struct BufferedImageCreateInfo {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t channels;
  BufferedImageTiling tiling;
  BufferedImageFormat format;
};

class BufferedImage {
public:
  BufferedImage() = default;
  BufferedImage(const BufferedImageCreateInfo &pCI){};
  ~BufferedImage() = default;

  virtual uint32_t getWidth() const = 0;
  virtual uint32_t getHeight() const = 0;
  virtual uint32_t getDepth() const = 0;
  virtual uint32_t getChannels() const = 0;

  virtual void setPixel3D(uint32_t x, uint32_t y, uint32_t z,
                          const void *pixel) = 0;
  virtual void setPixel2D(uint32_t x, uint32_t y, const void *pixel) = 0;
  virtual void getPixel3D(uint32_t x, uint32_t y, uint32_t z,
                          void *pixel) const = 0;
  virtual void getPixel2D(uint32_t x, uint32_t y, void *pixel) const = 0;
};

template <class Tp, class Allocator = std::allocator<Tp>>
class BufferedImageImpl : public BufferedImage {
protected:
  Tp *data;
  const uint32_t width;
  const uint32_t height;
  const uint32_t depth;
  const uint32_t channels;
  const BufferedImageTiling tiling;
  std::allocator<Tp> allocator;

public:
  BufferedImageImpl() {}
  BufferedImageImpl(const BufferedImageImpl &) = delete;
  BufferedImageImpl &operator=(const BufferedImageImpl &) = delete;

  inline uint32_t getPixelOffset3DLinear(uint32_t x, uint32_t y,
                                         uint32_t z) const {
    return (z * width * height + y * width + x) * channels;
  }

  inline uint32_t getPixelOffset2DLinear(uint32_t x, uint32_t y) const {
    return (y * width + x) * channels;
  }

  inline uint32_t getPixelOffset3DTiled(uint32_t x, uint32_t y,
                                        uint32_t z) const {
    const uint32_t tileX = x / BufferedImageConfig::TILE_SIZE;
    const uint32_t tileY = y / BufferedImageConfig::TILE_SIZE;
    const uint32_t tileZ = z / BufferedImageConfig::TILE_SIZE;
    const uint32_t tileOffset =
        (tileZ * (width / BufferedImageConfig::TILE_SIZE) *
             (height / BufferedImageConfig::TILE_SIZE) +
         tileY * (width / BufferedImageConfig::TILE_SIZE) + tileX) *
        (BufferedImageConfig::TILE_SIZE * BufferedImageConfig::TILE_SIZE *
         BufferedImageConfig::TILE_SIZE) *
        channels;
    const uint32_t localX = x % BufferedImageConfig::TILE_SIZE;
    const uint32_t localY = y % BufferedImageConfig::TILE_SIZE;
    const uint32_t localZ = z % BufferedImageConfig::TILE_SIZE;
    return tileOffset + (localZ * BufferedImageConfig::TILE_SIZE *
                             BufferedImageConfig::TILE_SIZE +
                         localY * BufferedImageConfig::TILE_SIZE + localX) *
                            channels;
  }

  inline uint32_t getPixelOffset2DTiled(uint32_t x, uint32_t y) const {
    const uint32_t tileX = x / BufferedImageConfig::TILE_SIZE;
    const uint32_t tileY = y / BufferedImageConfig::TILE_SIZE;
    const uint32_t tileOffset =
        (tileY * (width / BufferedImageConfig::TILE_SIZE) + tileX) *
        (BufferedImageConfig::TILE_SIZE * BufferedImageConfig::TILE_SIZE) *
        channels;
    const uint32_t localX = x % BufferedImageConfig::TILE_SIZE;
    const uint32_t localY = y % BufferedImageConfig::TILE_SIZE;
    return tileOffset +
           (localY * BufferedImageConfig::TILE_SIZE + localX) * channels;
  }

  inline uint32_t getRequiredBufferSizeLinear() const {
    return width * height * depth * channels;
  }

  inline uint32_t getRequiredBufferSizeTiled() const {
    const uint32_t numTilesX = (width + BufferedImageConfig::TILE_SIZE - 1) /
                               BufferedImageConfig::TILE_SIZE;
    const uint32_t numTilesY = (height + BufferedImageConfig::TILE_SIZE - 1) /
                               BufferedImageConfig::TILE_SIZE;
    const uint32_t numTilesZ = (depth + BufferedImageConfig::TILE_SIZE - 1) /
                               BufferedImageConfig::TILE_SIZE;
    return numTilesX * numTilesY * numTilesZ * BufferedImageConfig::TILE_SIZE *
           BufferedImageConfig::TILE_SIZE * BufferedImageConfig::TILE_SIZE *
           channels;
  }

  inline uint32_t getRequiredBufferSize() const {
    return tiling == BufferedImageTiling::TILING_LINEAR
               ? getRequiredBufferSizeLinear()
               : getRequiredBufferSizeTiled();
  }

public:
  BufferedImageImpl(const BufferedImageCreateInfo &pCI)
      : width(pCI.width), height(pCI.height), depth(pCI.depth),
        channels(pCI.channels), tiling(pCI.tiling) {
    data = allocator.allocate(getRequiredBufferSize());
  }

  ~BufferedImageImpl() { allocator.deallocate(data, getRequiredBufferSize()); }

  inline Tp *getData() { return data; }

  inline const Tp *getData() const { return data; }

  inline uint32_t getWidth() const { return width; }

  inline uint32_t getHeight() const { return height; }

  inline uint32_t getDepth() const { return depth; }

  inline uint32_t getChannels() const { return channels; }

  template <class Tp2> inline void loadFromPlainImage(const Tp2 *plainImage) {
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        for (int z = 0; z < depth; z++) {
          const Tp2 *pixel =
              plainImage + (z * width * height + y * width + x) * channels;
          setPixel3D(x, y, z, pixel);
        }
      }
    }
  }
};

// Linear Image
template <class Tp, class Allocator = std::allocator<Tp>>
class BufferedImageLinear : public BufferedImageImpl<Tp, Allocator> {
public:
  BufferedImageLinear(const BufferedImageCreateInfo &pCI)
      : BufferedImageImpl<Tp, Allocator>(pCI) {}

public:
  ~BufferedImageLinear() = default;

  inline void setPixel3D(uint32_t x, uint32_t y, uint32_t z,
                         const void *pixel) {
    const uint32_t offset = this->getPixelOffset3DLinear(x, y, z);
    memcpy(this->data + offset, pixel, this->channels * sizeof(Tp));
  }

  inline void setPixel2D(uint32_t x, uint32_t y, const void *pixel) {
    const uint32_t offset = this->getPixelOffset2DLinear(x, y);
    memcpy(this->data + offset, pixel, this->channels * sizeof(Tp));
  }

  inline void getPixel3D(uint32_t x, uint32_t y, uint32_t z,
                         void *pixel) const {
    const uint32_t offset = this->getPixelOffset3DLinear(x, y, z);
    memcpy(pixel, this->data + offset, this->channels * sizeof(Tp));
  }

  inline void getPixel2D(uint32_t x, uint32_t y, void *pixel) const {
    const uint32_t offset = this->getPixelOffset2DLinear(x, y);
    memcpy(pixel, this->data + offset, this->channels * sizeof(Tp));
  }
};

// Tiled Image
template <class Tp, class Allocator = std::allocator<Tp>>
class BufferedImageTiled : public BufferedImageImpl<Tp, Allocator> {
public:
  BufferedImageTiled(const BufferedImageCreateInfo &pCI)
      : BufferedImageImpl<Tp, Allocator>(pCI) {}

public:
  ~BufferedImageTiled() = default;

  inline void setPixel3D(uint32_t x, uint32_t y, uint32_t z,
                         const void *pixel) {
    const uint32_t offset = this->getPixelOffset3DTiled(x, y, z);
    memcpy(this->data + offset, pixel, this->channels * sizeof(Tp));
  }

  inline void setPixel2D(uint32_t x, uint32_t y, const void *pixel) {
    const uint32_t offset = this->getPixelOffset2DTiled(x, y);
    memcpy(this->data + offset, pixel, this->channels * sizeof(Tp));
  }

  inline void getPixel3D(uint32_t x, uint32_t y, uint32_t z,
                         void *pixel) const {
    const uint32_t offset = this->getPixelOffset3DTiled(x, y, z);
    memcpy(pixel, this->data + offset, this->channels * sizeof(Tp));
  }

  inline void getPixel2D(uint32_t x, uint32_t y, void *pixel) const {
    const uint32_t offset = this->getPixelOffset2DTiled(x, y);
    memcpy(pixel, this->data + offset, this->channels * sizeof(Tp));
  }
};

} // namespace Ifrit::Engine::SoftRenderer::Imaging