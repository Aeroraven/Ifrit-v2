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

} // namespace Ifrit::Math