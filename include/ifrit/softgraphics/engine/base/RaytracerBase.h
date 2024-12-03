
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
#include "ifrit/common/math/VectorOps.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include <vector>

// Order required
#include "ifrit/common/math/simd/SimdVectors.h"

namespace Ifrit::GraphicsBackend::SoftGraphics {
struct Ray {
  Ifrit::Math::SIMD::vfloat3 o;
  Ifrit::Math::SIMD::vfloat3 r;
};

struct RayInternal {
  Ifrit::Math::SIMD::vfloat3 o;
  Ifrit::Math::SIMD::vfloat3 r;
  Ifrit::Math::SIMD::vfloat3 invr;
};

struct RayHit {
  ifloat3 p;
  float t;
  int id;
};

template <class T> class BufferredAccelerationStructure {
public:
  virtual RayHit queryIntersection(const RayInternal &ray, float tmin,
                                   float tmax) const = 0;
  virtual void buildAccelerationStructure() = 0;
  virtual void bufferData(const std::vector<T> &data) = 0;
};

} // namespace Ifrit::GraphicsBackend::SoftGraphics