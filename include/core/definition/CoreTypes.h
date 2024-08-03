#pragma once
#include "core/definition/CoreDefs.h"
#include <cstdint>

template<class T>
struct CoreVec2 {
	T x, y;
};

template<class T>
struct CoreVec3 {
	T x, y, z;
};

template<class T>
struct CoreVec4 {
	T x, y, z, w;
};

template<class T>
struct Rect2D {
	T x, y, w, h;
};

template<class T,int U>
struct CoreVec4Shared {
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

#define ifloat4s256 CoreVec4Shared<float,256>
#define ifloat4s128 CoreVec4Shared<float,128>

template<class T>
struct CoreMat4 {
	 T data[4][4];
	IFRIT_DUAL const T* operator[](int i) const {
		return data[i];
	}
	IFRIT_DUAL T* operator[](int i) {
		return data[i];
	}
};
template struct CoreMat4<float>;

#define float4x4 CoreMat4<float>
