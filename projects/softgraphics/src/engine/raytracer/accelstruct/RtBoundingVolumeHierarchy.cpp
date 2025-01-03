
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

#include "ifrit/softgraphics/engine/raytracer/accelstruct/RtBoundingVolumeHierarchy.h"
#include "ifrit/common/math/VectorOps.h"
#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/common/util/TypingUtil.h"
#include <queue>


constexpr bool PROFILE_CNT = false;

using namespace Ifrit::Math::SIMD;
using Ifrit::Common::Utility::size_cast;

static auto totalTime = 0;
namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl {
static std::atomic<int> intersect = 0;
static std::atomic<int> validIntersect = 0;
static std::atomic<int> competeIntersect = 0;
static std::atomic<int> boxIntersect = 0;
static std::atomic<int> earlyReject = 0;

enum BVHSplitType { BST_TRIVIAL, BST_SAH };

enum BVHRayTriangleIntersectAlgo { BVH_RAYTRI_MOLLER, BVH_RAYTRI_BALDWIN };
static constexpr BVHSplitType splitType = BST_SAH;
static constexpr int maxDepth = 32;
static constexpr int sahBuckets = 15;

inline float procRayBoxIntersection(const RayInternal &ray,
                                    const BoundingBox &bbox) {
  using namespace Ifrit::Math;
  auto t1 = (bbox.bmin - ray.o) * ray.invr;
  auto t2 = (bbox.bmax - ray.o) * ray.invr;
  auto v1 = min(t1, t2);
  auto v2 = max(t1, t2);
  float tmin = std::max(v1.x, std::max(v1.y, v1.z));
  float tmax = std::min(v2.x, std::min(v2.y, v2.z));
  if (tmin > tmax)
    return -1;
  return tmin;
}

int procFindSplit(int start, int end, int axis, float mid,
                  std::vector<vfloat3> &centers, std::vector<int> &belonging) {
  using namespace Ifrit::Math;
  int l = start, r = end;

  while (l < r) {
    while (l < r && elementAt(centers[belonging[l]], axis) < mid)
      l++;
    while (l < r && elementAt(centers[belonging[r]], axis) >= mid)
      r--;
    if (l < r) {
      std::swap(belonging[l], belonging[r]);
      l++;
      r--;
    }
  }
  auto pivot = elementAt(centers[belonging[l]], axis) < mid ? l : l - 1;
  return pivot;
}

void procBuildBvhNode(int size, BVHNode *root,
                      const std::vector<BoundingBox> &bboxes,
                      std::vector<vfloat3> &centers,
                      std::vector<int> &belonging, std::vector<int> &indices) {

  using namespace Ifrit::Math;
  std::queue<std::tuple<BVHNode *, int, int>> q;
  q.push({root, 0, 0});
  int profNodes = 0;

  while (!q.empty()) {
    profNodes++;
    auto largestBBox = vfloat3(-std::numeric_limits<float>::max(),
                               -std::numeric_limits<float>::max(),
                               -std::numeric_limits<float>::max());
    auto &[node, depth, start] = q.front();
    BoundingBox &bbox = node->bbox;
    bbox.bmax = vfloat3(-std::numeric_limits<float>::max(),
                        -std::numeric_limits<float>::max(),
                        -std::numeric_limits<float>::max());
    bbox.bmin = vfloat3(std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max());

    for (int i = 0; i < node->elementSize; i++) {
      bbox.bmax = max(bbox.bmax, bboxes[belonging[start + i]].bmax);
      bbox.bmin = min(bbox.bmin, bboxes[belonging[start + i]].bmin);
      largestBBox = max(largestBBox, bboxes[belonging[start + i]].bmax -
                                         bboxes[belonging[start + i]].bmin);
    }
    node->startPos = start;
    if (depth >= maxDepth || node->elementSize <= 1) {
      q.pop();
      continue;
    }

    int pivot = 0;

    float bestPivot = 0.0;
    auto bestPivotI = 0;
    int bestAxis = -1;

    if (splitType == BST_TRIVIAL) {
      auto diff = bbox.bmax - bbox.bmin;
      auto midv = (bbox.bmax + bbox.bmin) * 0.5f;
      int axis = 0;
      if (diff.y > diff.x)
        axis = 1;
      if (diff.z > diff.y && diff.z > diff.x)
        axis = 2;
      float midvp = elementAt(midv, axis);
      pivot = procFindSplit(start, start + node->elementSize - 1, axis, midvp,
                            centers, belonging);
    } else if (splitType == BST_SAH) {
      auto diff = bbox.bmax - bbox.bmin;
      constexpr float unbalancedLeafPenalty = 80.0f;
      auto minCost = (node->elementSize == 2)
                         ? 1e30
                         : diff.x * diff.y * diff.z * 2.0 * node->elementSize +
                               unbalancedLeafPenalty;

      int baxis = 0;
      if (diff.y > diff.x)
        baxis = 1;
      if (diff.z > diff.y && diff.z > diff.x)
        baxis = 2;

      for (int axis = 0; axis < 3; axis++) {
        for (int i = 1; i < sahBuckets; i++) {
          auto midv = lerp(bbox.bmin, bbox.bmax, 1.0f * i / sahBuckets);
          float midvp = elementAt(midv, axis);
          pivot = procFindSplit(start, start + node->elementSize - 1, axis,
                                midvp, centers, belonging);

          BoundingBox bLeft, bRight;
          // Bounding boxes
          bLeft.bmax = vfloat3(-std::numeric_limits<float>::max(),
                               -std::numeric_limits<float>::max(),
                               -std::numeric_limits<float>::max());
          bLeft.bmin = vfloat3(std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max());
          bRight.bmax = vfloat3(-std::numeric_limits<float>::max(),
                                -std::numeric_limits<float>::max(),
                                -std::numeric_limits<float>::max());
          bRight.bmin = vfloat3(std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max());

          for (int j = start; j <= pivot; j++) {
            auto idx = belonging[j];
            bLeft.bmax = max(bLeft.bmax, bboxes[idx].bmax);
            bLeft.bmin = min(bLeft.bmin, bboxes[idx].bmin);
          }
          for (int j = pivot + 1; j < start + node->elementSize; j++) {
            auto idx = belonging[j];
            bRight.bmax = max(bRight.bmax, bboxes[idx].bmax);
            bRight.bmin = min(bRight.bmin, bboxes[idx].bmin);
          }
          auto dLeft = bLeft.bmax - bLeft.bmin;
          auto dRight = bRight.bmax - bRight.bmin;
          auto spLeft = dLeft.x * dLeft.y * dLeft.z * 2.0f;
          auto spRight = dRight.x * dRight.y * dRight.z * 2.0f;
          auto rnc = (node->elementSize - (pivot - start + 1));
          auto lnc = pivot - start + 1;
          auto rcost = spRight * rnc;
          auto lcost = spLeft * lnc;
          auto penaltyUnbalancedLeaf =
              ((lnc <= 1 && rnc > 2) || (rnc <= 1 && lnc > 2))
                  ? unbalancedLeafPenalty
                  : 0.0f;
          auto cost = lcost + rcost + penaltyUnbalancedLeaf;

          if (cost < minCost && !std::isnan(lcost) && !std::isnan(rcost)) {
            minCost = cost;
            bestAxis = axis;
            bestPivot = midvp;
          }
        }
      }

      if (bestAxis == -1) {
        pivot = -2;
      } else {
        pivot = procFindSplit(start, start + node->elementSize - 1, bestAxis,
                              bestPivot, centers, belonging);
      }
    }

    node->left = std::make_unique<BVHNode>();
    node->right = std::make_unique<BVHNode>();
    node->left->elementSize = pivot - start + 1;
    node->right->elementSize = node->elementSize - node->left->elementSize;

    bool isUnbalancedLeaf =
        pivot > 0 &&
        abs(node->left->elementSize - node->right->elementSize) > 2 &&
        (node->left->elementSize <= 1 || node->right->elementSize <= 1);
    auto cA = (node->left->elementSize > 1 || node->right->elementSize > 1);
    auto cB = (node->left->elementSize == 1 && node->right->elementSize == 1);

    if (isUnbalancedLeaf) {
      q.push({node->left.get(), depth + 1, start});
      q.push({node->right.get(), depth + 1, pivot + 1});
    } else if (pivot > 0 && (cA || cB)) {
      q.push({node->left.get(), depth + 1, start});
      q.push({node->right.get(), depth + 1, pivot + 1});
    } else {
      node->left = nullptr;
      node->right = nullptr;
    }
    q.pop();
  }
  for (int i = 0; i < size; i++) {
    indices[belonging[i]] = i;
  }
  ifritLog1("BVH Built, Total Nodes:", profNodes);
}

vfloat3 getElementCenterBLAS(int index, const std::vector<vfloat3> &data) {
  using namespace Ifrit::Math;
  vfloat3 v0 = (data)[index * 3];
  vfloat3 v1 = (data)[index * 3 + 1];
  vfloat3 v2 = (data)[index * 3 + 2];
  BoundingBox bbox;
  bbox.bmin = min(min(v0, v1), v2);
  bbox.bmax = max(max(v0, v1), v2);
  return (bbox.bmin + bbox.bmax) * 0.5f;
}

BoundingBox getElementBboxBLAS(int index, const std::vector<vfloat3> &data) {
  using namespace Ifrit::Math;
  vfloat3 v0 = (data)[index * 3];
  vfloat3 v1 = (data)[index * 3 + 1];
  vfloat3 v2 = (data)[index * 3 + 2];
  BoundingBox bbox;
  bbox.bmin = min(min(v0, v1), v2);
  bbox.bmax = max(max(v0, v1), v2);
  return bbox;
}

BoundingBox getElementBboxTLAS(
    int index,
    const std::vector<BoundingVolumeHierarchyBottomLevelAS *> &data) {
  auto x = data[index]->getRootBbox();
  return x;
}
vfloat3 getElementCenterTLAS(
    int index,
    const std::vector<BoundingVolumeHierarchyBottomLevelAS *> &data) {
  using namespace Ifrit::Math;
  auto bbox = data[index]->getRootBbox();
  auto cx = (bbox.bmax + bbox.bmin) * 0.5f;
  return cx;
}

void procBuildBvhTLAS(
    std::unique_ptr<BVHNode> &root, int size, std::vector<BoundingBox> &bboxes,
    std::vector<int> &indices, std::vector<vfloat3> &centers,
    std::vector<int> &belonging,
    const std::vector<BoundingVolumeHierarchyBottomLevelAS *> &data) {
  root = std::make_unique<BVHNode>();

  bboxes = std::vector<BoundingBox>(size);
  indices = std::vector<int>(size);
  centers = std::vector<vfloat3>(size);
  belonging = std::vector<int>(size);
  for (int i = 0; i < size; i++) {
    bboxes[i] = getElementBboxTLAS(i, data);
    centers[i] = getElementCenterTLAS(i, data);
    belonging[i] = i;
  }
  root->elementSize = size;
  procBuildBvhNode(size, root.get(), bboxes, centers, belonging, indices);
}
inline RayHit procRayElementIntersectionBalwin(
    const RayInternal &ray, int index, const std::vector<vfloat4> &tmat1,
    const std::vector<vfloat4> &tmat2, const std::vector<vfloat4> &tmat3) {
  RayHit rh;
  rh.id = -1;
  rh.t = std::numeric_limits<float>::max();
  vfloat4 ro = vfloat4(ray.o, 1.0f);
  vfloat4 rd = vfloat4(ray.r, 0.0f);
  float s = dot(tmat3[index], ro);
  float d = dot(tmat3[index], rd);
  float t = -s / d;
  if (t < 0)
    return rh;
  vfloat4 p = fma(rd, t, ro);
  p.w = 1;
  float u = dot(tmat1[index], p);
  float v = dot(tmat2[index], p);
  if (u < 0 || v < 0 || u + v > 1)
    return rh;

  rh.id = index;
  rh.p = {u, v, 1 - u - v};
  rh.t = t;
  return rh;
}
inline RayHit
procRayElementIntersectionMoller(const RayInternal &ray, int index,
                                 const std::vector<vfloat3> &data) {
  RayHit proposal;
  proposal.id = -1;
  proposal.t = std::numeric_limits<float>::max();
  vfloat3 v0 = data[index * 3];
  vfloat3 e1 = data[index * 3 + 1];
  vfloat3 e2 = data[index * 3 + 2];
  vfloat3 p = cross(ray.r, e2);
  float det = dot(e1, p);
  using namespace Ifrit::Math;
  if (det > std::numeric_limits<float>::epsilon()) {
    vfloat3 t = ray.o - v0;
    float u = dot(t, p);
    if (u < 0 || u > det) {
      return proposal;
    }
    vfloat3 q = cross(t, e1);
    float v = dot(ray.r, q);
    if (v < 0 || u + v > det) {
      return proposal;
    }
    float dist = dot(e2, q);
    proposal.id = index;
    proposal.p = {u, v, det};
    proposal.t = dist / det;
    return proposal;
  } else if (det < -std::numeric_limits<float>::epsilon()) {
    vfloat3 t = ray.o - v0;
    float u = dot(t, p);
    if (u > 0 || u < det) {
      return proposal;
    }
    vfloat3 q = cross(t, e1);
    float v = dot(ray.r, q);
    if (v > 0 || u + v < det) {
      return proposal;
    }
    float dist = dot(e2, q);
    proposal.id = index;
    proposal.p = {u, v, det};
    proposal.t = dist / det;
    return proposal;
  }
  return proposal;
}

inline RayHit procQueryRayIntersectionTLAS(
    const RayInternal &ray, float tmin, float tmax,
    const std::vector<int> &belonging, bool doRootBoxIgnore, BVHNode *root,
    const std::vector<BoundingVolumeHierarchyBottomLevelAS *> &data)
    IFRIT_AP_NOTHROW {
  return {};
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl

namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer {
void BoundingVolumeHierarchyBottomLevelAS::bufferData(
    const std::vector<ifloat3> &vecData) {
  this->data = std::vector<vfloat3>(vecData.size());
  for (int i = 0; i < vecData.size(); i++) {
    this->data[i] = vfloat3(vecData[i].x, vecData[i].y, vecData[i].z);
  }
  size = size_cast<int>(vecData.size()) / 3;
}
RayHit BoundingVolumeHierarchyBottomLevelAS::queryIntersection(
    const RayInternal &ray, float tmin, float tmax) const {
  using namespace Ifrit::Math;
  RayHit prop;
  prop.id = -1;
  prop.t = std::numeric_limits<float>::max();
  constexpr auto dfsStackSize = Impl::maxDepth * 2 + 1;
  std::tuple<BVHNode *, float> q[dfsStackSize];
  int qPos = 0;

  q[qPos++] = {root.get(), tmax / 2};
  float minDist = tmax;

  while (qPos) {
    auto p = q[--qPos];
    auto &[node, cmindist] = p;
    if (cmindist >= minDist)
      continue;
    float leftIntersect = -1, rightIntersect = -1;
    const auto nLeft = node->left.get();
    const auto nRight = node->right.get();
    const auto nSize = node->elementSize;
    const auto nStartPos = node->startPos;

    if (nLeft == nullptr || nRight == nullptr) {
      for (int i = 0; i < nSize; i++) {
        int index = belonging[i + nStartPos];
        auto dist = Impl::procRayElementIntersectionMoller(ray, index, data);
        if (dist.t > tmin && dist.t < minDist) {
          minDist = dist.t;
          prop = dist;
        }
      }
    } else {
      leftIntersect = Impl::procRayBoxIntersection(ray, nLeft->bbox);
      rightIntersect = Impl::procRayBoxIntersection(ray, nRight->bbox);
      if (leftIntersect > minDist)
        leftIntersect = -1;
      if (rightIntersect > minDist)
        rightIntersect = -1;
      if (leftIntersect > 0 && rightIntersect > 0) {
        if (leftIntersect < rightIntersect) {
          q[qPos++] = {nRight, rightIntersect};
          q[qPos++] = {nLeft, leftIntersect};
        } else {
          q[qPos++] = {nLeft, leftIntersect};
          q[qPos++] = {nRight, rightIntersect};
        }
      } else if (leftIntersect > 0)
        q[qPos++] = {nLeft, leftIntersect};
      else if (rightIntersect > 0)
        q[qPos++] = {nRight, rightIntersect};
    }
  }
  return prop;
}
void BoundingVolumeHierarchyBottomLevelAS::buildAccelerationStructure() {
  root = std::make_unique<BVHNode>();

  bboxes = std::vector<BoundingBox>(size);
  indices = std::vector<int>(size);
  centers = std::vector<vfloat3>(size);
  belonging = std::vector<int>(size);
  for (int i = 0; i < size; i++) {
    bboxes[i] = Impl::getElementBboxBLAS(i, data);
    centers[i] = Impl::getElementCenterBLAS(i, data);
    belonging[i] = i;
  }
  root->elementSize = size;
  Impl::procBuildBvhNode(size, root.get(), bboxes, centers, belonging, indices);

  // Balwin precompute
  // https://www.shadertoy.com/view/wttyR4
  balwinTmat1.resize(size);
  balwinTmat2.resize(size);
  balwinTmat3.resize(size);
  for (int i = 0; i < size * 3; i += 3) {
    auto id = i / 3;
    vfloat3 v0 = vfloat3(data[i]);
    vfloat3 v1 = vfloat3(data[i + 1]);
    vfloat3 v2 = vfloat3(data[i + 2]);
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto n = cross(e1, e2);
    auto an = abs(n);
    auto num = -dot(n, v0);
    if (an.x > an.y && an.x > an.z) {
      balwinTmat1[id] =
          vfloat4(0, e2.z, -e2.y, v2.y * v0.z - v2.z * v0.y) / n.x;
      balwinTmat2[id] =
          vfloat4(0, -e1.z, e1.y, v1.z * v0.y - v1.y * v0.z) / n.x;
      balwinTmat3[id] = vfloat4(n.x, n.y, n.z, num) / n.x;
    } else if (an.y > an.z) {
      balwinTmat1[id] =
          vfloat4(-e2.z, 0, e2.x, v2.z * v0.x - v2.x * v0.z) / n.y;
      balwinTmat2[id] =
          vfloat4(e1.z, 0, -e1.x, v1.x * v0.z - v1.z * v0.x) / n.y;
      balwinTmat3[id] = vfloat4(n.x, n.y, n.z, num) / n.y;
    } else {
      balwinTmat1[id] =
          vfloat4(e2.y, -e2.x, 0, v2.x * v0.y - v2.y * v0.x) / n.z;
      balwinTmat2[id] =
          vfloat4(-e1.y, e1.x, 0, v1.y * v0.x - v1.x * v0.y) / n.z;
      balwinTmat3[id] = vfloat4(n.x, n.y, n.z, num) / n.z;
    }
  }

  // Precompute edge
  for (int i = 0; i < size * 3; i += 3) {
    vfloat3 v0 = vfloat3(data[i]);
    vfloat3 v1 = vfloat3(data[i + 1]);
    vfloat3 v2 = vfloat3(data[i + 2]);
    data[i + 1] = v1 - v0;
    data[i + 2] = v2 - v0;
  }
}

void BoundingVolumeHierarchyTopLevelAS::bufferData(
    const std::vector<BoundingVolumeHierarchyBottomLevelAS *> &data) {
  this->data = data;
  this->size = size_cast<int>(data.size());
}
RayHit BoundingVolumeHierarchyTopLevelAS::queryIntersection(
    const RayInternal &ray, float tmin, float tmax) const {
  using namespace Ifrit::Math;
  RayHit prop;
  prop.id = -1;
  prop.t = std::numeric_limits<float>::max();
  float rootHit;

  rootHit = Impl::procRayBoxIntersection(ray, root->bbox);
  if (rootHit < 0) {
    return prop;
  }
  constexpr auto dfsStackSize = Impl::maxDepth * 2 + 1;
  std::tuple<BVHNode *, float> q[dfsStackSize];
  int qPos = 0;

  q[qPos++] = {root.get(), rootHit};
  float minDist = tmax;
  int ds = 0;
  while (qPos) {
    auto p = q[--qPos];
    auto &[node, cmindist] = p;
    if (cmindist >= minDist)
      continue;
    float leftIntersect = -1, rightIntersect = -1;
    const auto nLeft = node->left.get();
    const auto nRight = node->right.get();
    const auto nSize = node->elementSize;
    const auto nStartPos = node->startPos;

    if (nLeft == nullptr || nRight == nullptr) {
      for (int i = 0; i < nSize; i++) {
        ds++;
        int index = belonging[i + nStartPos];
        auto dist = data[index]->queryIntersection(ray, tmin, tmax);
        if (dist.t > tmin && dist.t < minDist) {
          minDist = dist.t;
          prop = dist;
        }
      }
    } else {
      leftIntersect = Impl::procRayBoxIntersection(ray, nLeft->bbox);
      rightIntersect = Impl::procRayBoxIntersection(ray, nRight->bbox);
      if (leftIntersect > minDist)
        leftIntersect = -1;
      if (rightIntersect > minDist)
        rightIntersect = -1;
      if (leftIntersect > 0 && rightIntersect > 0) {
        if (leftIntersect < rightIntersect) {
          q[qPos++] = {nRight, rightIntersect};
          q[qPos++] = {nLeft, leftIntersect};
        } else {
          q[qPos++] = {nLeft, leftIntersect};
          q[qPos++] = {nRight, rightIntersect};
        }
      } else if (leftIntersect > 0)
        q[qPos++] = {nLeft, leftIntersect};
      else if (rightIntersect > 0)
        q[qPos++] = {nRight, rightIntersect};
    }
  }
  return prop;
}
void BoundingVolumeHierarchyTopLevelAS::buildAccelerationStructure() {
  Impl::procBuildBvhTLAS(root, size, bboxes, indices, centers, belonging, data);
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer

int getProfileCnt() {
  return totalTime;
  if constexpr (PROFILE_CNT) {
    int v = Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::intersect;
    int vv =
        Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::validIntersect;
    int bv =
        Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::boxIntersect;
    int er = Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::earlyReject;
    int cv =
        Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::competeIntersect;
    printf("Total Intersect:%d, Valid Intersect:%d , Overtest Rate:%f\n", v, vv,
           1.0f * vv / v);
    printf("Compete Intersect:%d \n", cv);
    printf("Total Box Intersect:%d Box/Triangle Ratio: %f\n", bv,
           1.0f * bv / v);
    printf("Early Reject: %d\n\n", er);

    Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::intersect.store(0);
    Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::competeIntersect
        .store(0);
    Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::validIntersect.store(
        0);
    Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::boxIntersect.store(
        0);
    Ifrit::GraphicsBackend::SoftGraphics::Raytracer::Impl::earlyReject.store(0);
    return v;
  }
  return 0;
}