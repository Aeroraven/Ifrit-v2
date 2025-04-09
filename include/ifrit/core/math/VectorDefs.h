
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
#include "../platform/ApiConv.h"
#include "../cuda/CudaBaseDefs.h"
#include "../base/IfritBase.h"
#include <cstdint>

#define IF_VECDECL3(a, b, c) \
    IF_FORCEINLINE IF_CONSTEXPR CoreVec3<T> a##b##c() const { return CoreVec3<T>(a, b, c); }

#define IF_VECDECL3_3P1(a, b) \
    IF_VECDECL3(a, b, x)      \
    IF_VECDECL3(a, b, y)      \
    IF_VECDECL3(a, b, z)

#define IF_VECDECL3_3P2(a) \
    IF_VECDECL3_3P1(a, x)  \
    IF_VECDECL3_3P1(a, y)  \
    IF_VECDECL3_3P1(a, z)

#define IF_VECDECL3_3P3() \
    IF_VECDECL3_3P2(x)    \
    IF_VECDECL3_3P2(y)    \
    IF_VECDECL3_3P2(z)

#define IF_VECDECL3_P1(a, b) \
    IF_VECDECL3(a, b, x)     \
    IF_VECDECL3(a, b, y)     \
    IF_VECDECL3(a, b, z)     \
    IF_VECDECL3(a, b, w)

#define IF_VECDECL3_P2(a) \
    IF_VECDECL3_P1(a, x)  \
    IF_VECDECL3_P1(a, y)  \
    IF_VECDECL3_P1(a, z)  \
    IF_VECDECL3_P1(a, w)

#define IF_VECDECL3_P3() \
    IF_VECDECL3_P2(x)    \
    IF_VECDECL3_P2(y)    \
    IF_VECDECL3_P2(z)    \
    IF_VECDECL3_P2(w)

#define IF_VECDECL4(a, b, c, d) \
    IF_FORCEINLINE IF_CONSTEXPR CoreVec4<T> a##b##c##d() const { return CoreVec4<T>(a, b, c, d); }

#define IF_VECDECL4_P1(a, b, c) \
    IF_VECDECL4(a, b, c, x)     \
    IF_VECDECL4(a, b, c, y)     \
    IF_VECDECL4(a, b, c, z)     \
    IF_VECDECL4(a, b, c, w)

#define IF_VECDECL4_P2(a, b) \
    IF_VECDECL4_P1(a, b, x)  \
    IF_VECDECL4_P1(a, b, y)  \
    IF_VECDECL4_P1(a, b, z)  \
    IF_VECDECL4_P1(a, b, w)

#define IF_VECDECL4_P3(a) \
    IF_VECDECL4_P2(a, x)  \
    IF_VECDECL4_P2(a, y)  \
    IF_VECDECL4_P2(a, z)  \
    IF_VECDECL4_P2(a, w)

#define IF_VECDECL4_P4() \
    IF_VECDECL4_P3(x)    \
    IF_VECDECL4_P3(y)    \
    IF_VECDECL4_P3(z)    \
    IF_VECDECL4_P3(w)

template <class T> struct CoreVec2
{
    T                           x, y;
    IF_CONSTEXPR                CoreVec2() = default;
    IF_CONSTEXPR                CoreVec2(T x) : x(x), y(x) {}
    IF_CONSTEXPR                CoreVec2(T x, T y) : x(x), y(y) {}
    IF_CONSTEXPR                CoreVec2(const CoreVec2& v) : x(v.x), y(v.y) {}

    IF_FORCEINLINE T&           operator[](int i) { return ((T*)this)[i]; }
    IF_FORCEINLINE const T&     operator[](int i) const { return ((T*)this)[i]; }

    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> xx() const { return CoreVec2<T>(x, x); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> yy() const { return CoreVec2<T>(y, y); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> xy() const { return CoreVec2<T>(x, y); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> yx() const { return CoreVec2<T>(y, x); }
};

template <class T> struct CoreVec3
{
    T                           x, y, z;

    IF_CONSTEXPR                CoreVec3() = default;
    IF_CONSTEXPR                CoreVec3(T x) : x(x), y(x), z(x) {}
    IF_CONSTEXPR                CoreVec3(T x, T y, T z) : x(x), y(y), z(z) {}
    IF_CONSTEXPR                CoreVec3(const CoreVec3& v) : x(v.x), y(v.y), z(v.z) {}

    IF_CONSTEXPR                CoreVec3(const CoreVec2<T>& v, T z) : x(v.x), y(v.y), z(z) {}
    IF_CONSTEXPR                CoreVec3(T x, const CoreVec2<T>& v) : x(x), y(v.x), z(v.y) {}

    IF_FORCEINLINE T&           operator[](int i) { return ((T*)this)[i]; }
    IF_FORCEINLINE const T&     operator[](int i) const { return ((T*)this)[i]; }

    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> xy() const { return CoreVec2<T>(x, y); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> xx() const { return CoreVec2<T>(x, x); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> yy() const { return CoreVec2<T>(y, y); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> yx() const { return CoreVec2<T>(y, x); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> zx() const { return CoreVec2<T>(z, x); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> zy() const { return CoreVec2<T>(z, y); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> xz() const { return CoreVec2<T>(x, z); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> yz() const { return CoreVec2<T>(y, z); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> zz() const { return CoreVec2<T>(z, z); }

    IF_VECDECL3_3P3()
};

template <class T> struct CoreVec4
{
    T                       x, y, z, w;

    IF_CONSTEXPR            CoreVec4() = default;
    IF_CONSTEXPR            CoreVec4(T x) : x(x), y(x), z(x), w(x) {}
    IF_CONSTEXPR            CoreVec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    IF_CONSTEXPR            CoreVec4(const CoreVec4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    IF_CONSTEXPR            CoreVec4(const CoreVec3<T>& v, T w) : x(v.x), y(v.y), z(v.z), w(w) {}
    IF_CONSTEXPR            CoreVec4(const CoreVec2<T>& v, T z, T w) : x(v.x), y(v.y), z(z), w(w) {}
    IF_CONSTEXPR            CoreVec4(const CoreVec2<T>& v, const CoreVec2<T>& v2) : x(v.x), y(v.y), z(v2.x), w(v2.y) {}
    IF_CONSTEXPR            CoreVec4(T x, const CoreVec2<T>& v, T w) : x(x), y(v.x), z(v.y), w(w) {}
    IF_CONSTEXPR            CoreVec4(T x, T y, const CoreVec2<T>& v) : x(x), y(y), z(v.x), w(v.y) {}
    IF_CONSTEXPR            CoreVec4(T x, const CoreVec3<T>& v) : x(x), y(v.x), z(v.y), w(v.z) {}

    IF_FORCEINLINE T&       operator[](int i) { return ((T*)this)[i]; }
    IF_FORCEINLINE const T& operator[](int i) const { return ((T*)this)[i]; }

    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> xy() const { return CoreVec2<T>(x, y); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> xz() const { return CoreVec2<T>(x, z); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> xw() const { return CoreVec2<T>(x, w); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> yx() const { return CoreVec2<T>(y, x); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> yz() const { return CoreVec2<T>(y, z); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> yw() const { return CoreVec2<T>(y, w); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> zx() const { return CoreVec2<T>(z, x); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> zy() const { return CoreVec2<T>(z, y); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> zw() const { return CoreVec2<T>(z, w); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> wx() const { return CoreVec2<T>(w, x); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> wy() const { return CoreVec2<T>(w, y); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> wz() const { return CoreVec2<T>(w, z); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> xx() const { return CoreVec2<T>(x, x); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> yy() const { return CoreVec2<T>(y, y); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> zz() const { return CoreVec2<T>(z, z); }
    IF_FORCEINLINE IF_CONSTEXPR CoreVec2<T> ww() const { return CoreVec2<T>(w, w); }

    IF_VECDECL3_P3()
    IF_VECDECL4_P4()
};

#undef IF_VECDECL3_3P1
#undef IF_VECDECL3_3P2
#undef IF_VECDECL3_3P3

#undef IF_VECDECL3_P1
#undef IF_VECDECL3_P2
#undef IF_VECDECL3_P3

#undef IF_VECDECL4_P1
#undef IF_VECDECL4_P2
#undef IF_VECDECL4_P3
#undef IF_VECDECL4_P4

#undef IF_VECDECL4
#undef IF_VECDECL3

template <class T> struct Rect2D
{
    T x, y, w, h;
};

template <class T> struct Rect3D
{
    T x, y, z, w, h, d;
};

template <class T, int U> struct CoreVec4Shared
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

#define Vector2bool CoreVec2<bool>
#define Vector3bool CoreVec3<bool>
#define Vector4bool CoreVec4<bool>

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

template <class T> struct CoreMat4
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