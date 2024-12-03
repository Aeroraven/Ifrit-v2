
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
#include "../util/CudaBaseDefs.h"
#include <cstdint>

template <class T> struct CoreVec2 {
  T x, y;
};

template <class T> struct CoreVec3 {
  T x, y, z;
};

template <class T> struct CoreVec4 {
  T x, y, z, w;
};

template <class T> struct Rect2D {
  T x, y, w, h;
};

template <class T> struct Rect3D {
  T x, y, z, w, h, d;
};

template <class T, int U> struct CoreVec4Shared {
  T x;
  T xp[U - 1];
  T y;
  T yp[U - 1];
  T z;
  T zp[U - 1];
  T w;
  T wp[U - 1];
};

#define ifloat4 CoreVec4<float>
#define ifloat3 CoreVec3<float>
#define ifloat2 CoreVec2<float>
#define idouble4 CoreVec4<double>
#define idouble3 CoreVec3<double>
#define idouble2 CoreVec2<double>
#define iint4 CoreVec4<int>
#define iint3 CoreVec3<int>
#define iint2 CoreVec2<int>
#define ishort4 CoreVec4<short>
#define ishort3 CoreVec3<short>
#define ishort2 CoreVec2<short>
#define iuint4 CoreVec4<unsigned int>
#define iuint3 CoreVec3<unsigned int>
#define iuint2 CoreVec2<unsigned int>

#define irect2Df Rect2D<float>
#define irect2Di Rect2D<int>
#define irect2Dui Rect2D<unsigned int>
#define irect3Df Rect3D<float>
#define irect3Di Rect3D<int>
#define irect3Dui Rect3D<unsigned int>

#define ifloat4s256 CoreVec4Shared<float, 256>
#define ifloat4s128 CoreVec4Shared<float, 128>

#define igvec2 CoreVec2
#define igvec3 CoreVec3
#define igvec4 CoreVec4

template <class T> struct CoreMat4 {
  T data[4][4];
  IFRIT_DUAL const T *operator[](int i) const { return data[i]; }
  IFRIT_DUAL T *operator[](int i) { return data[i]; }
};
template struct CoreMat4<float>;
#define float4x4 CoreMat4<float>

extern "C" {
template struct IFRIT_APIDECL ifloat2;
template struct IFRIT_APIDECL ifloat3;
template struct IFRIT_APIDECL ifloat4;
template struct IFRIT_APIDECL idouble2;
template struct IFRIT_APIDECL idouble3;
template struct IFRIT_APIDECL idouble4;
template struct IFRIT_APIDECL iint2;
template struct IFRIT_APIDECL iint3;
template struct IFRIT_APIDECL iint4;
template struct IFRIT_APIDECL ishort2;
template struct IFRIT_APIDECL ishort3;
template struct IFRIT_APIDECL ishort4;
template struct IFRIT_APIDECL iuint2;
template struct IFRIT_APIDECL iuint3;
template struct IFRIT_APIDECL iuint4;

template struct IFRIT_APIDECL irect2Df;
template struct IFRIT_APIDECL irect2Di;
template struct IFRIT_APIDECL irect2Dui;
template struct IFRIT_APIDECL irect3Df;
template struct IFRIT_APIDECL irect3Di;
template struct IFRIT_APIDECL irect3Dui;
}