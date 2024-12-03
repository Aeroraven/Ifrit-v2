
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
#include "ifrit/softgraphics/engine/base/RaytracerBase.h"
// Order required
#include "ifrit/common/math/simd/SimdVectors.h"

int getProfileCnt();

namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer {

struct BoundingBox {
  Ifrit::Math::SIMD::vfloat3 bmin, bmax;
};

struct BVHNode {
  BoundingBox bbox;
  std::unique_ptr<BVHNode> left = nullptr, right = nullptr;
  int elementSize = 0;
  int startPos = 0;
};

class IFRIT_APIDECL BoundingVolumeHierarchyBottomLevelAS {
private:
  std::vector<Ifrit::Math::SIMD::vfloat3> data;
  std::unique_ptr<BVHNode> root;
  std::vector<BoundingBox> bboxes;
  std::vector<Ifrit::Math::SIMD::vfloat3> centers;
  std::vector<int> belonging;
  std::vector<int> indices;
  int size;

  std::vector<Ifrit::Math::SIMD::vfloat4> balwinTmat1, balwinTmat2, balwinTmat3;

public:
  BoundingVolumeHierarchyBottomLevelAS() = default;
  ~BoundingVolumeHierarchyBottomLevelAS() = default;
  void bufferData(const std::vector<ifloat3> &data);
  RayHit queryIntersection(const RayInternal &ray, float tmin,
                           float tmax) const;
  void buildAccelerationStructure();

  inline BoundingBox getRootBbox() { return root->bbox; }
};

class IFRIT_APIDECL BoundingVolumeHierarchyTopLevelAS {
private:
  std::vector<BoundingVolumeHierarchyBottomLevelAS *> data;
  std::unique_ptr<BVHNode> root;
  std::vector<BoundingBox> bboxes;
  std::vector<Ifrit::Math::SIMD::vfloat3> centers;
  std::vector<int> belonging;
  std::vector<int> indices;
  int size;

public:
  BoundingVolumeHierarchyTopLevelAS() = default;
  ~BoundingVolumeHierarchyTopLevelAS() = default;
  void
  bufferData(const std::vector<BoundingVolumeHierarchyBottomLevelAS *> &data);
  RayHit queryIntersection(const RayInternal &ray, float tmin,
                           float tmax) const;
  void buildAccelerationStructure();
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer