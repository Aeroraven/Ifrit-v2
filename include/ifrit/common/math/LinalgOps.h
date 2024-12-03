
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
inline float4x4 transpose(const float4x4 &a) {
  float4x4 result;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result[i][j] = a[j][i];
    }
  }
  return result;
}
inline ifloat4 matmul(const float4x4 &a, const ifloat4 &b) {
  ifloat4 result;
  result.x = a[0][0] * b.x + a[0][1] * b.y + a[0][2] * b.z + a[0][3] * b.w;
  result.y = a[1][0] * b.x + a[1][1] * b.y + a[1][2] * b.z + a[1][3] * b.w;
  result.z = a[2][0] * b.x + a[2][1] * b.y + a[2][2] * b.z + a[2][3] * b.w;
  result.w = a[3][0] * b.x + a[3][1] * b.y + a[3][2] * b.z + a[3][3] * b.w;
  return result;
}
inline float4x4 matmul(const float4x4 &a, const float4x4 &b) {
  float4x4 result;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] +
                     a[i][3] * b[3][j];
    }
  }
  return result;
}
inline float4x4 axisAngleRotation(const ifloat3 &axis, float angle) {
  float c = cos(angle), s = sin(angle);
  float t = 1 - c, x = axis.x, y = axis.y, z = axis.z;
  float4x4 ret;
  ret[0][0] = t * x * x + c;
  ret[0][1] = t * x * y + s * z;
  ret[0][2] = t * x * z - s * y;
  ret[0][3] = 0;
  ret[1][0] = t * x * y - s * z;
  ret[1][1] = t * y * y + c;
  ret[1][2] = t * y * z + s * x;
  ret[1][3] = 0;
  ret[2][0] = t * x * z + s * y;
  ret[2][1] = t * y * z - s * x;
  ret[2][2] = t * z * z + c;
  ret[2][3] = 0;
  ret[3][0] = 0;
  ret[3][1] = 0;
  ret[3][2] = 0;
  ret[3][3] = 1;
  return ret;
}
inline float4x4 lookAt(ifloat3 eye, ifloat3 center, ifloat3 up) {
  ifloat3 f = normalize((center - eye));
  ifloat3 s = normalize(cross(f, up));
  ifloat3 u = cross(s, f);
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
  return matmul(result, trans);
}
inline float4x4 perspective(float fovy, float aspect, float zNear, float zFar) {
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
inline float4x4 perspectiveNegateY(float fovy, float aspect, float zNear,
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
  result[1][1] = -2 * zNear / (nTop - nBottom);
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
inline float4x4 eulerAngleToMatrix(const ifloat3 &euler) {
  float4x4 result;
  float cx = cos(euler.x), sx = sin(euler.x);
  float cy = cos(euler.y), sy = sin(euler.y);
  float cz = cos(euler.z), sz = sin(euler.z);
  result[0][0] = cy * cz;
  result[0][1] = cy * sz;
  result[0][2] = -sy;
  result[0][3] = 0;
  result[1][0] = sx * sy * cz - cx * sz;
  result[1][1] = sx * sy * sz + cx * cz;
  result[1][2] = sx * cy;
  result[1][3] = 0;
  result[2][0] = cx * sy * cz + sx * sz;
  result[2][1] = cx * sy * sz - sx * cz;
  result[2][2] = cx * cy;
  result[2][3] = 0;
  result[3][0] = 0;
  result[3][1] = 0;
  result[3][2] = 0;
  result[3][3] = 1;
  return result;
}
inline float4x4 identity() {
  float4x4 result;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result[i][j] = i == j ? 1 : 0;
    }
  }
  return result;
}

inline float4x4 translate3D(const ifloat3 &t) {
  float4x4 result = identity();
  result[0][3] = t.x;
  result[1][3] = t.y;
  result[2][3] = t.z;
  return result;
}

inline float4x4 scale3D(const ifloat3 &s) {
  float4x4 result = identity();
  result[0][0] = s.x;
  result[1][1] = s.y;
  result[2][2] = s.z;
  return result;
}

inline float4x4 inverse4(const float4x4 &p) {
  // From: https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
  // Translated by copilot (AI)
  auto a2323 = p[2][2] * p[3][3] - p[2][3] * p[3][2];
  auto a1323 = p[2][1] * p[3][3] - p[2][3] * p[3][1];
  auto a1223 = p[2][1] * p[3][2] - p[2][2] * p[3][1];
  auto a0323 = p[2][0] * p[3][3] - p[2][3] * p[3][0];
  auto a0223 = p[2][0] * p[3][2] - p[2][2] * p[3][0];
  auto a0123 = p[2][0] * p[3][1] - p[2][1] * p[3][0];
  auto a2313 = p[1][2] * p[3][3] - p[1][3] * p[3][2];
  auto a1313 = p[1][1] * p[3][3] - p[1][3] * p[3][1];
  auto a1213 = p[1][1] * p[3][2] - p[1][2] * p[3][1];
  auto a2312 = p[1][2] * p[2][3] - p[1][3] * p[2][2];
  auto a1312 = p[1][1] * p[2][3] - p[1][3] * p[2][1];
  auto a1212 = p[1][1] * p[2][2] - p[1][2] * p[2][1];
  auto a0313 = p[1][0] * p[3][3] - p[1][3] * p[3][0];
  auto a0213 = p[1][0] * p[3][2] - p[1][2] * p[3][0];
  auto a0312 = p[1][0] * p[2][3] - p[1][3] * p[2][0];
  auto a0212 = p[1][0] * p[2][2] - p[1][2] * p[2][0];
  auto a0113 = p[1][0] * p[3][1] - p[1][1] * p[3][0];
  auto a0112 = p[1][0] * p[2][1] - p[1][1] * p[2][0];
  auto det = p[0][0] * (p[1][1] * a2323 - p[1][2] * a1323 + p[1][3] * a1223) -
             p[0][1] * (p[1][0] * a2323 - p[1][2] * a0323 + p[1][3] * a0223) +
             p[0][2] * (p[1][0] * a1323 - p[1][1] * a0323 + p[1][3] * a0123) -
             p[0][3] * (p[1][0] * a1223 - p[1][1] * a0223 + p[1][2] * a0123);
  auto invdet = 1 / det;
  float4x4 inv;
  inv[0][0] = invdet * (p[1][1] * a2323 - p[1][2] * a1323 + p[1][3] * a1223);
  inv[0][1] = -invdet * (p[0][1] * a2323 - p[0][2] * a1323 + p[0][3] * a1223);
  inv[0][2] = invdet * (p[0][1] * a2313 - p[0][2] * a1313 + p[0][3] * a1213);
  inv[0][3] = -invdet * (p[0][1] * a2312 - p[0][2] * a1312 + p[0][3] * a1212);
  inv[1][0] = -invdet * (p[1][0] * a2323 - p[1][2] * a0323 + p[1][3] * a0223);
  inv[1][1] = invdet * (p[0][0] * a2323 - p[0][2] * a0323 + p[0][3] * a0223);
  inv[1][2] = -invdet * (p[0][0] * a2313 - p[0][2] * a0313 + p[0][3] * a0213);
  inv[1][3] = invdet * (p[0][0] * a2312 - p[0][2] * a0312 + p[0][3] * a0212);
  inv[2][0] = invdet * (p[1][0] * a1323 - p[1][1] * a0323 + p[1][3] * a0123);
  inv[2][1] = -invdet * (p[0][0] * a1323 - p[0][1] * a0323 + p[0][3] * a0123);
  inv[2][2] = invdet * (p[0][0] * a1313 - p[0][1] * a0313 + p[0][3] * a0113);
  inv[2][3] = -invdet * (p[0][0] * a1312 - p[0][1] * a0312 + p[0][3] * a0112);
  inv[3][0] = -invdet * (p[1][0] * a1223 - p[1][1] * a0223 + p[1][2] * a0123);
  inv[3][1] = invdet * (p[0][0] * a1223 - p[0][1] * a0223 + p[0][2] * a0123);
  inv[3][2] = -invdet * (p[0][0] * a1213 - p[0][1] * a0213 + p[0][2] * a0113);
  inv[3][3] = invdet * (p[0][0] * a1212 - p[0][1] * a0212 + p[0][2] * a0112);
  return inv;
}

} // namespace Ifrit::Math