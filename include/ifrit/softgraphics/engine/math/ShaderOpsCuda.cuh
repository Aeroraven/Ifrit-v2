
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
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/softgraphics/core/definition/CoreDefs.h"
#include "ifrit/softgraphics/core/definition/CoreTypes.h"

#define IFRIT_InvoGetThreadBlocks(tasks, blockSize)                            \
  ((tasks) / (blockSize)) + ((tasks) % (blockSize) != 0)
#define IFRIT_InvoCeilRshift(x, y) (((x) + ((1 << (y)) - 1)) >> (y))

#ifdef IFRIT_FEATURE_CUDA
namespace Ifrit::SoftRenderer::Math::ShaderOps::CUDA {
template <class T>
IFRIT_DUAL inline T clamp(const T &x, const T &mi, const T &ma) {
  return (x >= ma) ? ma : ((x <= mi) ? mi : x);
}
template <class T>
IFRIT_DUAL inline T mirrorclamp(const T &x, const T &mi, const T &ma) {
  return (x >= ma) ? mi : ((x <= mi) ? ma : x);
}
IFRIT_DUAL inline float4x4 multiply(const float4x4 a, const float4x4 b) {
  float4x4 result;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] +
                     a[i][3] * b[3][j];
    }
  }
  return result;
}
IFRIT_DUAL inline float4 abs(float4 x) {
  return {fabs(x.x), fabs(x.y), fabs(x.z), fabs(x.w)};
}
IFRIT_DUAL inline float4 normalize(float4 a) {
  float length = sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
  return {a.x / length, a.y / length, a.z / length, a.w / length};
}
IFRIT_DUAL inline float3 cross(float3 a, float3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
IFRIT_DUAL inline float3 normalize(float3 a) {
  float length = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  return {a.x / length, a.y / length, a.z / length};
}

IFRIT_DUAL inline float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
IFRIT_DUAL inline float4x4 transpose(float4x4 a) {
  float4x4 result;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result[i][j] = a[j][i];
    }
  }
  return result;
}
IFRIT_DUAL inline float dot(const float4 &a, const float4 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
IFRIT_DUAL inline float4 sub(const float4 &a, const float4 &b) {
  return float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
IFRIT_DUAL inline float3 sub(float3 a, float3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
IFRIT_DUAL inline float4 multiply(const float4 &a, const float &b) {
  return float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
IFRIT_DUAL inline float3 multiply(const float3 &a, const float &b) {
  return float3(a.x * b, a.y * b, a.z * b);
}
IFRIT_DUAL inline float4 add(const float4 &a, const float4 &b) {
  return float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
IFRIT_DUAL inline float3 add(const float3 &a, const float3 &b) {
  return float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

IFRIT_DUAL inline float4 lerp(const float4 &a, const float4 &b,
                              const float &t) {
  return add(a, multiply(sub(b, a), t));
}
IFRIT_DUAL inline float3 lerp(const float3 &a, const float3 &b,
                              const float &t) {
  return add(a, multiply(sub(b, a), t));
}
template <typename T>
IFRIT_DUAL inline T lerp(const T &a, const T &b, const T &t) {
  return a + (b - a) * t;
}

template <typename T>
IFRIT_DUAL inline T multiply(const float4x4 a, const T b) {
  T result;
  result.x = a[0][0] * b.x + a[0][1] * b.y + a[0][2] * b.z + a[0][3] * b.w;
  result.y = a[1][0] * b.x + a[1][1] * b.y + a[1][2] * b.z + a[1][3] * b.w;
  result.z = a[2][0] * b.x + a[2][1] * b.y + a[2][2] * b.z + a[2][3] * b.w;
  result.w = a[3][0] * b.x + a[3][1] * b.y + a[3][2] * b.z + a[3][3] * b.w;
  return result;
}
IFRIT_DUAL inline float4x4 lookAt(float3 eye, float3 center, float3 up) {
  float3 f = normalize(sub(center, eye));
  float3 s = normalize(cross(f, up));
  float3 u = cross(s, f);
  float4x4 result;
  result[0][0] = s.x;
  result[0][1] = s.y;
  result[0][2] = s.z;
  result[0][3] = 0;
  result[1][0] = u.x;
  result[1][1] = u.y;
  result[1][2] = u.z;
  result[1][3] = 0;
  result[2][0] = f.x;
  result[2][1] = f.y;
  result[2][2] = f.z;
  result[2][3] = 0;
  result[3][0] = 0;
  result[3][1] = 0;
  result[3][2] = 0;
  result[3][3] = 1;
  float4x4 trans;
  for (int i = 0; i <= 3; i++) {
    trans[i][0] = 0;
    trans[i][1] = 0;
    trans[i][2] = 0;
    trans[i][3] = 0;
  }
  trans[0][3] = -eye.x;
  trans[1][3] = -eye.y;
  trans[2][3] = -eye.z;
  trans[3][3] = 1;
  trans[0][0] = 1;
  trans[1][1] = 1;
  trans[2][2] = 1;
  return multiply(result, trans);
}
IFRIT_DUAL inline float4x4 perspective(float fovy, float aspect, float zNear,
                                       float zFar) {
  float4x4 result;
  float halfFovy = fovy / 2.0f;
  float nTop = zNear * tan(halfFovy);
  float nRight = nTop * aspect;
  float nLeft = -nRight;
  float nBottom = -nTop;
  result[0][0] = 2 * zNear / (nRight - nLeft);
  result[1][0] = 0;
  result[2][0] = 0;
  result[3][0] = 0;
  result[0][1] = 0;
  result[1][1] = 2 * zNear / (nTop - nBottom);
  result[2][1] = 0;
  result[3][1] = 0;
  result[0][2] = 0;
  result[1][2] = 0;
  result[2][2] = (zFar) / (zFar - zNear);
  result[3][2] = 1;
  result[0][3] = 0;
  result[1][3] = 0;
  result[2][3] = -(zFar * zNear) / (zFar - zNear);
  result[3][3] = 0;
  return result;
}

} // namespace Ifrit::SoftRenderer::Math::ShaderOps::CUDA
#endif