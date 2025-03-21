
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
#include "ifrit/meshproc/engine/meshsdf/MeshSDFConverter.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/common/util/Parallel.h"
#include "ifrit/common/util/TypingUtil.h"

// TODO: Use algo in following paper
// Reference: https://www2.imm.dtu.dk/pubdb/edoc/imm1289.pdf

using namespace Ifrit::Math::SIMD;
namespace Ifrit::MeshProcLib::MeshSDFProcess {

IF_CONSTEXPR u32 cMinBvhChilds = 2;

struct BVHNode : public Common::Utility::NonCopyableStruct {
  vfloat3 bboxMin;
  vfloat3 bboxMax;
  Uref<BVHNode> left;
  Uref<BVHNode> right;
  u32 startIdx;
  u32 endIdx;
};

struct Mesh2SDFTempData : public Common::Utility::NonCopyableStruct {
  vfloat3 bboxMin;
  vfloat3 bboxMax;
  f32 *meshVxBuffer;
  u32 *meshIxBuffer;
  u32 meshVxStride;
  u32 meshNumVerices;
  u32 meshNumIndices;

  Vec<u32> asTriIndices;
  Uref<BVHNode> asRoot;
  Vec<vfloat3> asTriBboxMin;
  Vec<vfloat3> asTriBboxMax;
  Vec<vfloat3> asTriBboxMid;
  Vec<vfloat3> asTriNormals;
};

IF_FORCEINLINE void computeMeshBoundingBox(Mesh2SDFTempData &data) {
  const u32 vertexCount = data.meshNumVerices;
  for (u32 i = 0; i < vertexCount; i++) {
    f32 vX = data.meshVxBuffer[i * data.meshVxStride + 0];
    f32 vY = data.meshVxBuffer[i * data.meshVxStride + 1];
    f32 vZ = data.meshVxBuffer[i * data.meshVxStride + 2];
    vfloat3 vP = vfloat3(vX, vY, vZ);
    data.bboxMin = min(data.bboxMin, vP);
    data.bboxMax = max(data.bboxMax, vP);
  }
}

IF_FORCEINLINE vfloat3 getTriangleNormal(const vfloat3 &a, const vfloat3 &b, const vfloat3 &c) {
  return normalize(cross(b - a, c - a));
}

IF_FORCEINLINE void computeTriangleBoundingBox(Mesh2SDFTempData &data) {
  const u32 indexCount = data.meshNumIndices;
  data.asTriBboxMin.resize(indexCount / 3);
  data.asTriBboxMax.resize(indexCount / 3);
  data.asTriBboxMid.resize(indexCount / 3);
  data.asTriNormals.resize(indexCount / 3);
  for (u32 i = 0; i < indexCount; i += 3) {
    u32 i0 = data.meshIxBuffer[i + 0];
    u32 i1 = data.meshIxBuffer[i + 1];
    u32 i2 = data.meshIxBuffer[i + 2];
    f32 v0X = data.meshVxBuffer[i0 * data.meshVxStride + 0];
    f32 v0Y = data.meshVxBuffer[i0 * data.meshVxStride + 1];
    f32 v0Z = data.meshVxBuffer[i0 * data.meshVxStride + 2];
    f32 v1X = data.meshVxBuffer[i1 * data.meshVxStride + 0];
    f32 v1Y = data.meshVxBuffer[i1 * data.meshVxStride + 1];
    f32 v1Z = data.meshVxBuffer[i1 * data.meshVxStride + 2];
    f32 v2X = data.meshVxBuffer[i2 * data.meshVxStride + 0];
    f32 v2Y = data.meshVxBuffer[i2 * data.meshVxStride + 1];
    f32 v2Z = data.meshVxBuffer[i2 * data.meshVxStride + 2];
    vfloat3 v0 = vfloat3(v0X, v0Y, v0Z);
    vfloat3 v1 = vfloat3(v1X, v1Y, v1Z);
    vfloat3 v2 = vfloat3(v2X, v2Y, v2Z);
    vfloat3 triMin = min(min(v0, v1), v2);
    vfloat3 triMax = max(max(v0, v1), v2);
    data.asTriBboxMin[i / 3] = triMin;
    data.asTriBboxMax[i / 3] = triMax;
    data.asTriBboxMid[i / 3] = (triMin + triMax) * 0.5f;
    data.asTriNormals[i / 3] = getTriangleNormal(v0, v1, v2);
  }
}
IF_FORCEINLINE vfloat3 pointDistToTriangle(const vfloat3 &p, const vfloat3 &a, const vfloat3 &b, const vfloat3 &c) {
  // Code from: https://github.com/RenderKit/embree/blob/master/tutorials/common/math/closest_point.h
  const auto ab = b - a;
  const auto ac = c - a;
  const auto ap = p - a;

  const f32 d1 = dot(ab, ap);
  const f32 d2 = dot(ac, ap);
  if (d1 <= 0.f && d2 <= 0.f)
    return a;

  const auto bp = p - b;
  const f32 d3 = dot(ab, bp);
  const f32 d4 = dot(ac, bp);
  if (d3 >= 0.f && d4 <= d3)
    return b;

  const auto cp = p - c;
  const f32 d5 = dot(ab, cp);
  const f32 d6 = dot(ac, cp);
  if (d6 >= 0.f && d5 <= d6)
    return c;

  const f32 vc = d1 * d4 - d3 * d2;
  if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
    const f32 v = d1 / (d1 - d3);
    return a + ab * v;
  }

  const f32 vb = d5 * d2 - d1 * d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
    const f32 v = d2 / (d2 - d6);
    return a + ac * v;
  }

  const f32 va = d3 * d6 - d5 * d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
    const f32 v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return b + (c - b) * v;
  }

  const f32 denom = 1.f / (va + vb + vc);
  const f32 v = vb * denom;
  const f32 w = vc * denom;
  return a + ab * v + ac * w;
}

void calculateAsChildBbox(const Mesh2SDFTempData &data, Uref<BVHNode> &node) {
  node->bboxMin = vfloat3(FLT_MAX);
  node->bboxMax = vfloat3(-FLT_MAX);
  for (u32 i = node->startIdx; i < node->endIdx; i++) {
    auto triId = data.asTriIndices[i];
    node->bboxMin = min(node->bboxMin, data.asTriBboxMin[triId]);
    node->bboxMax = max(node->bboxMax, data.asTriBboxMax[triId]);
  }
}

void buildAccelStructRecur(Mesh2SDFTempData &data, Uref<BVHNode> &node) {
  auto numChildren = node->endIdx - node->startIdx;
  if (numChildren <= cMinBvhChilds) {
    return;
  }
  auto longestAxis = 0;
  auto axisLength = node->bboxMax.x - node->bboxMin.x;
  if (node->bboxMax.y - node->bboxMin.y > axisLength) {
    longestAxis = 1;
    axisLength = node->bboxMax.y - node->bboxMin.y;
  }
  if (node->bboxMax.z - node->bboxMin.z > axisLength) {
    longestAxis = 2;
  }
  auto mid = (node->bboxMin + node->bboxMax) * 0.5f;

  // Inplace rearrange
  auto leftIdx = node->startIdx;
  auto rightIdx = node->endIdx - 1;
  while (leftIdx < rightIdx) {
    while (leftIdx < rightIdx && elementAt(data.asTriBboxMid[leftIdx], longestAxis) < elementAt(mid, longestAxis)) {
      leftIdx++;
    }
    while (leftIdx < rightIdx && elementAt(data.asTriBboxMid[rightIdx], longestAxis) >= elementAt(mid, longestAxis)) {
      rightIdx--;
    }
    if (leftIdx < rightIdx) {
      std::swap(data.asTriIndices[leftIdx], data.asTriIndices[rightIdx]);
      leftIdx++;
      rightIdx--;
    }
  }
  if (leftIdx == node->startIdx || leftIdx == node->endIdx) {
    return;
  }

  node->left = std::make_unique<BVHNode>();
  node->left->startIdx = node->startIdx;
  node->left->endIdx = leftIdx;
  calculateAsChildBbox(data, node->left);

  node->right = std::make_unique<BVHNode>();
  node->right->startIdx = leftIdx;
  node->right->endIdx = node->endIdx;
  calculateAsChildBbox(data, node->right);

  buildAccelStructRecur(data, node->left);
  buildAccelStructRecur(data, node->right);
}

void buildAccelStruct(Mesh2SDFTempData &data) {
  data.asRoot = std::make_unique<BVHNode>();
  data.asRoot->bboxMin = data.bboxMin;
  data.asRoot->bboxMax = data.bboxMax;
  data.asRoot->startIdx = 0;
  data.asRoot->endIdx = data.asTriIndices.size();
  buildAccelStructRecur(data, data.asRoot);
}

IF_FORCEINLINE f32 getDistanceToBbox(const vfloat3 &p, const vfloat3 &bboxMin, const vfloat3 &bboxMax) {
  vfloat3 dxMin = bboxMin - p;
  vfloat3 dxMax = p - bboxMax;
  vfloat3 dxZero = vfloat3(0.0f);
  vfloat3 dx = max(max(dxMin, dxMax), dxZero);
  f32 dist = length(dx);
  return dist;
}

void getSignedDistanceToMeshRecur(const Mesh2SDFTempData &data, const vfloat3 &p, BVHNode *node, f32 &tgtDist) {
  auto leftChild = node->left.get();
  auto rightChild = node->right.get();
  if (leftChild && rightChild) {
    // Non child node, check distance to bbox
    auto leftChildDist = getDistanceToBbox(p, leftChild->bboxMin, leftChild->bboxMax);
    auto rightChildDist = getDistanceToBbox(p, rightChild->bboxMin, rightChild->bboxMax);
    if (leftChildDist <= rightChildDist && leftChildDist < std::abs(tgtDist)) {
      getSignedDistanceToMeshRecur(data, p, leftChild, tgtDist);
      if (std::abs(tgtDist) > rightChildDist) {
        getSignedDistanceToMeshRecur(data, p, rightChild, tgtDist);
      }
    } else if (rightChildDist < leftChildDist && rightChildDist < std::abs(tgtDist)) {
      getSignedDistanceToMeshRecur(data, p, rightChild, tgtDist);
      if (std::abs(tgtDist) > leftChildDist) {
        getSignedDistanceToMeshRecur(data, p, leftChild, tgtDist);
      }
    }
  } else {
    // child node, check distance to triangle
    for (u32 i = node->startIdx; i < node->endIdx; i++) {
      auto triId = data.asTriIndices[i];
      auto i0 = data.meshIxBuffer[triId * 3 + 0];
      auto i1 = data.meshIxBuffer[triId * 3 + 1];
      auto i2 = data.meshIxBuffer[triId * 3 + 2];

      auto v0 = vfloat3(data.meshVxBuffer[i0 * data.meshVxStride + 0], data.meshVxBuffer[i0 * data.meshVxStride + 1],
                        data.meshVxBuffer[i0 * data.meshVxStride + 2]);
      auto v1 = vfloat3(data.meshVxBuffer[i1 * data.meshVxStride + 0], data.meshVxBuffer[i1 * data.meshVxStride + 1],
                        data.meshVxBuffer[i1 * data.meshVxStride + 2]);
      auto v2 = vfloat3(data.meshVxBuffer[i2 * data.meshVxStride + 0], data.meshVxBuffer[i2 * data.meshVxStride + 1],
                        data.meshVxBuffer[i2 * data.meshVxStride + 2]);

      auto vNormal = data.asTriNormals[triId];
      auto nearestPt = pointDistToTriangle(p, v0, v1, v2);
      auto vDist = p - nearestPt;
      auto sign = dot(vDist, vNormal) > 0.0f ? 1.0f : -1.0f;
      auto dist = length(vDist) * sign;
      if (std::abs(dist) < std::abs(tgtDist)) {
        tgtDist = dist;
      }
    }
  }
}

f32 getSignedDistanceToMesh(const Mesh2SDFTempData &data, const vfloat3 &p) {
  f32 tgtDist = FLT_MAX;
  getSignedDistanceToMeshRecur(data, p, data.asRoot.get(), tgtDist);
  return tgtDist;
}

IFRIT_MESHPROC_API void convertMeshToSDF(const MeshDescriptor &meshDesc, SignedDistanceField &sdf, u32 sdfWidth,
                                         u32 sdfHeight, u32 sdfDepth) {
  Mesh2SDFTempData data;
  data.bboxMin = vfloat3(FLT_MAX);
  data.bboxMax = vfloat3(-FLT_MAX);
  data.meshVxBuffer = reinterpret_cast<f32 *>(meshDesc.vertexData);
  data.meshIxBuffer = reinterpret_cast<u32 *>(meshDesc.indexData);
  data.meshVxStride = meshDesc.vertexStride / sizeof(f32);
  data.meshNumVerices = meshDesc.vertexCount;
  data.meshNumIndices = meshDesc.indexCount;
  computeMeshBoundingBox(data);
  computeTriangleBoundingBox(data);

  // build accel structure
  data.asTriIndices.resize(meshDesc.indexCount / 3);
  for (u32 i = 0; i < meshDesc.indexCount / 3; i++) {
    data.asTriIndices[i] = i;
  }
  buildAccelStruct(data);

  // then, for each voxel, calculate the distance to the mesh
  sdf.width = sdfWidth;
  sdf.height = sdfHeight;
  sdf.depth = sdfDepth;
  sdf.sdfData.resize(sdfWidth * sdfHeight * sdfDepth);
  auto totalVoxels = sdfWidth * sdfHeight * sdfDepth;

  Common::Utility::unordered_for<u32>(0, totalVoxels, [&](u32 el) {
    auto depth = el / (sdfWidth * sdfHeight);
    auto height = (el % (sdfWidth * sdfHeight)) / sdfWidth;
    auto width = (el % (sdfWidth * sdfHeight)) % sdfWidth;
    f32 x = (f32)width / (f32)sdfWidth + 0.5f;
    f32 y = (f32)height / (f32)sdfHeight + 0.5f;
    f32 z = (f32)depth / (f32)sdfDepth + 0.5f;
    f32 lx = std::lerp(data.bboxMin.x, data.bboxMax.x, x);
    f32 ly = std::lerp(data.bboxMin.y, data.bboxMax.y, y);
    f32 lz = std::lerp(data.bboxMin.z, data.bboxMax.z, z);
    vfloat3 p = vfloat3(lx, ly, lz);
    f32 dist = getSignedDistanceToMesh(data, p);
    sdf.sdfData[el] = dist;
  });
  sdf.bboxMin = ifloat3(data.bboxMin.x, data.bboxMin.y, data.bboxMin.z);
  sdf.bboxMax = ifloat3(data.bboxMax.x, data.bboxMax.y, data.bboxMax.z);
}

} // namespace Ifrit::MeshProcLib::MeshSDFProcess