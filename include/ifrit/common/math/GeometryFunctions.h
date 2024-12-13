
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
#include "../util/ApiConv.h"
#include "./LinalgOps.h"
#include "VectorOps.h"
#include <cmath>

namespace Ifrit::Math {

inline ifloat4 getFrustumBoundingSphere(float fovy, float aspect, float fNear,
                                        float fFar, ifloat3 apex) {
  float halfFov = fovy / 2.0f;
  printf("HalfFov %f\n", std::tan(halfFov));
  auto halfHeightNear = fNear * std::tan(halfFov);
  auto halfWidthNear = halfHeightNear * aspect;
  auto halfHeightFar = fFar * std::tan(halfFov);
  auto halfWidthFar = halfHeightFar * aspect;
  auto radiusNear = std::sqrt(halfHeightNear * halfHeightNear +
                              halfWidthNear * halfWidthNear);
  auto radiusFar =
      std::sqrt(halfHeightFar * halfHeightFar + halfWidthFar * halfWidthFar);

  // let x be the distance from the apex to the center of the sphere
  // We have ((x-zn)^2+nr^2 = r^2) and ((x-zf)^2+fr^2 = r^2)
  // So, x^2 - 2x*zn + zn^2 + nr^2 = r^2 and x^2 - 2x*zf + zf^2 + fr^2 = r^2
  // So, x^2 - 2x*zn + zn^2 + nr^2 = x^2 - 2x*zf + zf^2 + fr^2
  // So, 2x(zf - zn) = fr^2 - nr^2 + zf^2 - zn^2
  // => x = (fr^2 - nr^2 + zf^2 - zn^2) / 2(zf - zn)
  auto l = fFar - fNear;
  auto x =
      (radiusFar * radiusFar - radiusNear * radiusNear + l * (fFar + fNear)) /
      (2 * l);
  auto zCenter = x;
  auto sphereRadius =
      std::sqrt(radiusNear * radiusNear + (x - fNear) * (x - fNear));
  ifloat3 center = {apex.x, apex.y, zCenter + apex.z};
  return {center.x, center.y, center.z, sphereRadius};
}

inline void getFrustumBoundingBoxWithRay(float fovy, float aspect, float zNear,
                                         float zFar, ifloat3 apex,
                                         ifloat3 rayDir, float reqResultZNear,
                                         float &resZFar, float &resOrthoSize,
                                         ifloat3 &resCenter) {
  float halfFov = fovy / 2.0f;
  auto halfHeightNear = zNear * std::tan(halfFov);
  auto halfWidthNear = halfHeightNear * aspect;
  auto halfHeightFar = zFar * std::tan(halfFov);
  auto halfWidthFar = halfHeightFar * aspect;

  std::vector<ifloat4> worldSpacePts;
  worldSpacePts.push_back({halfWidthNear, halfHeightNear, zNear, 1.0f});
  worldSpacePts.push_back({-halfWidthNear, halfHeightNear, zNear, 1.0f});
  worldSpacePts.push_back({halfWidthNear, -halfHeightNear, zNear, 1.0f});
  worldSpacePts.push_back({-halfWidthNear, -halfHeightNear, zNear, 1.0f});
  worldSpacePts.push_back({halfWidthFar, halfHeightFar, zFar, 1.0f});
  worldSpacePts.push_back({-halfWidthFar, halfHeightFar, zFar, 1.0f});
  worldSpacePts.push_back({halfWidthFar, -halfHeightFar, zFar, 1.0f});
  worldSpacePts.push_back({-halfWidthFar, -halfHeightFar, zFar, 1.0f});

  for (auto &pt : worldSpacePts) {
    pt = {apex.x + pt.x, apex.y + pt.y, apex.z + pt.z, 1.0f};
  }

  float projMinX = std::numeric_limits<float>::max();
  float projMaxX = std::numeric_limits<float>::min();
  float projMinY = std::numeric_limits<float>::max();
  float projMaxY = std::numeric_limits<float>::min();
  float projMinZ = std::numeric_limits<float>::max();
  float projMaxZ = std::numeric_limits<float>::min();

  ifloat3 dLookAtCenter = ifloat3{0.0f, 0.0f, 0.0f};
  ifloat3 dUp = ifloat3{0.0f, 1.0f, 0.0f};
  ifloat3 dRay = rayDir;

  float4x4 dTestView = lookAt(dLookAtCenter, dRay, dUp);
  float4x4 dTestViewToWorld = inverse4(dTestView);

  std::vector<ifloat4> viewSpacePts;
  for (auto &pt : worldSpacePts) {
    auto viewSpacePt = matmul(dTestView, pt);
    viewSpacePts.push_back(viewSpacePt);
    projMinX = std::min(projMinX, viewSpacePt.x);
    projMaxX = std::max(projMaxX, viewSpacePt.x);
    projMinY = std::min(projMinY, viewSpacePt.y);
    projMaxY = std::max(projMaxY, viewSpacePt.y);
    projMinZ = std::min(projMinZ, viewSpacePt.z);
    projMaxZ = std::max(projMaxZ, viewSpacePt.z);
  }

  // AABB center in viewspace
  auto center =
      ifloat3{(projMinX + projMaxX) / 2.0f, (projMinY + projMaxY) / 2.0f,
              (projMinZ + projMaxZ) / 2.0f};
  auto orthoSize = std::max(projMaxX - projMinX, projMaxY - projMinY);
  auto orthoSizeZ = (projMaxZ - projMinZ) * 0.5f + reqResultZNear;
  auto orthoSizeZFar = (projMaxZ - projMinZ) + reqResultZNear;
  auto worldCenter =
      matmul(dTestViewToWorld, ifloat4{center.x, center.y, center.z, 1.0f});
  auto reqCamPos = ifloat3{worldCenter.x - orthoSizeZ * dRay.x,
                           worldCenter.y - orthoSizeZ * dRay.y,
                           worldCenter.z - orthoSizeZ * dRay.z};

  // Return
  resZFar = orthoSizeZFar;
  resOrthoSize = orthoSize;
  resCenter = {reqCamPos.x, reqCamPos.y, reqCamPos.z};
}
} // namespace Ifrit::Math