
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
#include "VectorOps.h"
#include <cmath>

namespace Ifrit::Math {

inline ifloat4 getFrustumBoundingSphere(float fovy, float aspect, float fNear,
                                        float fFar, ifloat3 apex) {
  float halfFov = fovy / 2.0f;
  auto halfHeightNear = fNear * std::tan(halfFov);
  auto halfWidthNear = halfHeightNear * aspect;
  auto halfHeightFar = fFar * std::tan(halfFov);
  auto halfWidthFar = halfHeightFar * aspect;
  auto radiusNear = std::sqrt(halfHeightNear * halfHeightNear +
                              halfWidthNear * halfWidthNear);
  auto radiusFar = std::sqrt(halfHeightFar * halfHeightFar +
                             halfWidthFar * halfWidthFar);

  // We have (x^2+nr^2 = r^2) and ((l-x)^2+fr^2 = r^2)
  // where, l = (zFar-zNear)
  // so, x^2 + nr^2 = (l-x)^2 + fr^2 = x^2 -2lx + l^2 + fr^2
  // => 2lx = fr^2 - nr^2 + l^2 => x = (fr^2 - nr^2 + l^2) / 2l
  auto l = fFar - fNear;
  auto x = (radiusFar * radiusFar - radiusNear * radiusNear + l * l) / (2 * l);
  auto zCenter = fNear + x;
  auto sphereRadius = std::sqrt(radiusNear * radiusNear + x * x);
  ifloat3 center = {apex.x, apex.y, zCenter + apex.z};
  return {center.x, center.y, center.z, sphereRadius};
}
} // namespace Ifrit::Math