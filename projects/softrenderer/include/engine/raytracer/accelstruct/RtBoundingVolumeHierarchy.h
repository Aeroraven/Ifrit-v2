#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/RaytracerBase.h"
#include "math/simd/SimdVectors.h"

int getProfileCnt();

namespace Ifrit::Engine::Raytracer {

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
} // namespace Ifrit::Engine::Raytracer