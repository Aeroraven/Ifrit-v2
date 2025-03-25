
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/Structures.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging {

enum class BufferedImageTiling { TILING_LINEAR, TILING_TILED };
enum class BufferedImageFormat { FORMAT_R8G8B8A8_UINT, FORMAT_R32G32B32A32_SFLOAT };

struct BufferedImageConfig {
  static IF_CONSTEXPR u32 TILE_SIZE = 16;
};

struct BufferedImageCreateInfo {
  u32 width;
  u32 height;
  u32 depth;
  u32 channels;
  BufferedImageTiling tiling;
  BufferedImageFormat format;
};

class BufferedImage {
public:
  BufferedImage() = default;
  BufferedImage(const BufferedImageCreateInfo &pCI){};
  ~BufferedImage() = default;

  virtual u32 getWidth() const = 0;
  virtual u32 getHeight() const = 0;
  virtual u32 getDepth() const = 0;
  virtual u32 getChannels() const = 0;

  virtual void setPixel3D(u32 x, u32 y, u32 z, const void *pixel) = 0;
  virtual void setPixel2D(u32 x, u32 y, const void *pixel) = 0;
  virtual void getPixel3D(u32 x, u32 y, u32 z, void *pixel) const = 0;
  virtual void getPixel2D(u32 x, u32 y, void *pixel) const = 0;
};

template <class Tp, class Allocator = std::allocator<Tp>> class BufferedImageImpl : public BufferedImage {
protected:
  Tp *data;
  const u32 width;
  const u32 height;
  const u32 depth;
  const u32 channels;
  const BufferedImageTiling tiling;
  std::allocator<Tp> allocator;

public:
  BufferedImageImpl() {}
  BufferedImageImpl(const BufferedImageImpl &) = delete;
  BufferedImageImpl &operator=(const BufferedImageImpl &) = delete;

  inline u32 getPixelOffset3DLinear(u32 x, u32 y, u32 z) const {
    return (z * width * height + y * width + x) * channels;
  }

  inline u32 getPixelOffset2DLinear(u32 x, u32 y) const { return (y * width + x) * channels; }

  inline u32 getPixelOffset3DTiled(u32 x, u32 y, u32 z) const {
    const u32 tileX = x / BufferedImageConfig::TILE_SIZE;
    const u32 tileY = y / BufferedImageConfig::TILE_SIZE;
    const u32 tileZ = z / BufferedImageConfig::TILE_SIZE;
    const u32 tileOffset =
        (tileZ * (width / BufferedImageConfig::TILE_SIZE) * (height / BufferedImageConfig::TILE_SIZE) +
         tileY * (width / BufferedImageConfig::TILE_SIZE) + tileX) *
        (BufferedImageConfig::TILE_SIZE * BufferedImageConfig::TILE_SIZE * BufferedImageConfig::TILE_SIZE) * channels;
    const u32 localX = x % BufferedImageConfig::TILE_SIZE;
    const u32 localY = y % BufferedImageConfig::TILE_SIZE;
    const u32 localZ = z % BufferedImageConfig::TILE_SIZE;
    return tileOffset + (localZ * BufferedImageConfig::TILE_SIZE * BufferedImageConfig::TILE_SIZE +
                         localY * BufferedImageConfig::TILE_SIZE + localX) *
                            channels;
  }

  inline u32 getPixelOffset2DTiled(u32 x, u32 y) const {
    const u32 tileX = x / BufferedImageConfig::TILE_SIZE;
    const u32 tileY = y / BufferedImageConfig::TILE_SIZE;
    const u32 tileOffset = (tileY * (width / BufferedImageConfig::TILE_SIZE) + tileX) *
                           (BufferedImageConfig::TILE_SIZE * BufferedImageConfig::TILE_SIZE) * channels;
    const u32 localX = x % BufferedImageConfig::TILE_SIZE;
    const u32 localY = y % BufferedImageConfig::TILE_SIZE;
    return tileOffset + (localY * BufferedImageConfig::TILE_SIZE + localX) * channels;
  }

  inline u32 getRequiredBufferSizeLinear() const { return width * height * depth * channels; }

  inline u32 getRequiredBufferSizeTiled() const {
    const u32 numTilesX = (width + BufferedImageConfig::TILE_SIZE - 1) / BufferedImageConfig::TILE_SIZE;
    const u32 numTilesY = (height + BufferedImageConfig::TILE_SIZE - 1) / BufferedImageConfig::TILE_SIZE;
    const u32 numTilesZ = (depth + BufferedImageConfig::TILE_SIZE - 1) / BufferedImageConfig::TILE_SIZE;
    return numTilesX * numTilesY * numTilesZ * BufferedImageConfig::TILE_SIZE * BufferedImageConfig::TILE_SIZE *
           BufferedImageConfig::TILE_SIZE * channels;
  }

  inline u32 getRequiredBufferSize() const {
    return tiling == BufferedImageTiling::TILING_LINEAR ? getRequiredBufferSizeLinear() : getRequiredBufferSizeTiled();
  }

public:
  BufferedImageImpl(const BufferedImageCreateInfo &pCI)
      : width(pCI.width), height(pCI.height), depth(pCI.depth), channels(pCI.channels), tiling(pCI.tiling) {
    data = allocator.allocate(getRequiredBufferSize());
  }

  ~BufferedImageImpl() { allocator.deallocate(data, getRequiredBufferSize()); }

  inline Tp *getData() { return data; }

  inline const Tp *getData() const { return data; }

  inline u32 getWidth() const { return width; }

  inline u32 getHeight() const { return height; }

  inline u32 getDepth() const { return depth; }

  inline u32 getChannels() const { return channels; }

  template <class Tp2> inline void loadFromPlainImage(const Tp2 *plainImage) {
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        for (int z = 0; z < depth; z++) {
          const Tp2 *pixel = plainImage + (z * width * height + y * width + x) * channels;
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
  BufferedImageLinear(const BufferedImageCreateInfo &pCI) : BufferedImageImpl<Tp, Allocator>(pCI) {}

public:
  ~BufferedImageLinear() = default;

  inline void setPixel3D(u32 x, u32 y, u32 z, const void *pixel) {
    const u32 offset = this->getPixelOffset3DLinear(x, y, z);
    memcpy(this->data + offset, pixel, this->channels * sizeof(Tp));
  }

  inline void setPixel2D(u32 x, u32 y, const void *pixel) {
    const u32 offset = this->getPixelOffset2DLinear(x, y);
    memcpy(this->data + offset, pixel, this->channels * sizeof(Tp));
  }

  inline void getPixel3D(u32 x, u32 y, u32 z, void *pixel) const {
    const u32 offset = this->getPixelOffset3DLinear(x, y, z);
    memcpy(pixel, this->data + offset, this->channels * sizeof(Tp));
  }

  inline void getPixel2D(u32 x, u32 y, void *pixel) const {
    const u32 offset = this->getPixelOffset2DLinear(x, y);
    memcpy(pixel, this->data + offset, this->channels * sizeof(Tp));
  }
};

// Tiled Image
template <class Tp, class Allocator = std::allocator<Tp>>
class BufferedImageTiled : public BufferedImageImpl<Tp, Allocator> {
public:
  BufferedImageTiled(const BufferedImageCreateInfo &pCI) : BufferedImageImpl<Tp, Allocator>(pCI) {}

public:
  ~BufferedImageTiled() = default;

  inline void setPixel3D(u32 x, u32 y, u32 z, const void *pixel) {
    const u32 offset = this->getPixelOffset3DTiled(x, y, z);
    memcpy(this->data + offset, pixel, this->channels * sizeof(Tp));
  }

  inline void setPixel2D(u32 x, u32 y, const void *pixel) {
    const u32 offset = this->getPixelOffset2DTiled(x, y);
    memcpy(this->data + offset, pixel, this->channels * sizeof(Tp));
  }

  inline void getPixel3D(u32 x, u32 y, u32 z, void *pixel) const {
    const u32 offset = this->getPixelOffset3DTiled(x, y, z);
    memcpy(pixel, this->data + offset, this->channels * sizeof(Tp));
  }

  inline void getPixel2D(u32 x, u32 y, void *pixel) const {
    const u32 offset = this->getPixelOffset2DTiled(x, y);
    memcpy(pixel, this->data + offset, this->channels * sizeof(Tp));
  }
};

} // namespace Ifrit::GraphicsBackend::SoftGraphics::Imaging