
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

template <class T>
struct CoreVec2
{
    T x, y;
};

template <class T>
struct CoreVec3
{
    T x, y, z;
};

template <class T>
struct CoreVec4
{
    T x, y, z, w;
};

template <class T>
struct Rect2D
{
    T x, y, w, h;
};

template <class T>
struct Rect3D
{
    T x, y, z, w, h, d;
};

template <class T, int U>
struct CoreVec4Shared
{
    T x;
    T xp[U - 1];
    T y;
    T yp[U - 1];
    T z;
    T zp[U - 1];
    T w;
    T wp[U - 1];
};

#define Vector4f CoreVec4<float>
#define Vector3f CoreVec3<float>
#define Vector2f CoreVec2<float>
#define Vector4d CoreVec4<double>
#define Vector3d CoreVec3<double>
#define Vector2d CoreVec2<double>
#define Vector4i CoreVec4<int>
#define Vector3i CoreVec3<int>
#define Vector2i CoreVec2<int>
#define Vector4s CoreVec4<short>
#define Vector3s CoreVec3<short>
#define Vector2s CoreVec2<short>
#define Vector4u CoreVec4<unsigned int>
#define Vector3u CoreVec3<unsigned int>
#define Vector2u CoreVec2<unsigned int>

#define irect2Df Rect2D<float>
#define irect2Di Rect2D<int>
#define irect2Dui Rect2D<unsigned int>
#define irect3Df Rect3D<float>
#define irect3Di Rect3D<int>
#define irect3Dui Rect3D<unsigned int>

#define Vector4fs256 CoreVec4Shared<float, 256>
#define Vector4fs128 CoreVec4Shared<float, 128>

#define igvec2 CoreVec2
#define igvec3 CoreVec3
#define igvec4 CoreVec4

template <class T>
struct CoreMat4
{
    T                   data[4][4];
    IFRIT_DUAL const T* operator[](int i) const { return data[i]; }
    IFRIT_DUAL T*       operator[](int i) { return data[i]; }
};
template struct CoreMat4<float>;
#define Matrix4x4f CoreMat4<float>

extern "C"
{
    template struct IFRIT_APIDECL Vector2f;
    template struct IFRIT_APIDECL Vector3f;
    template struct IFRIT_APIDECL Vector4f;
    template struct IFRIT_APIDECL Vector2d;
    template struct IFRIT_APIDECL Vector3d;
    template struct IFRIT_APIDECL Vector4d;
    template struct IFRIT_APIDECL Vector2i;
    template struct IFRIT_APIDECL Vector3i;
    template struct IFRIT_APIDECL Vector4i;
    template struct IFRIT_APIDECL Vector2s;
    template struct IFRIT_APIDECL Vector3s;
    template struct IFRIT_APIDECL Vector4s;
    template struct IFRIT_APIDECL Vector2u;
    template struct IFRIT_APIDECL Vector3u;
    template struct IFRIT_APIDECL Vector4u;

    template struct IFRIT_APIDECL irect2Df;
    template struct IFRIT_APIDECL irect2Di;
    template struct IFRIT_APIDECL irect2Dui;
    template struct IFRIT_APIDECL irect3Df;
    template struct IFRIT_APIDECL irect3Di;
    template struct IFRIT_APIDECL irect3Dui;
}