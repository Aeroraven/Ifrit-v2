#pragma once
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

#define ifloat4 CoreVec4<float>
#define ifloat3 CoreVec3<float>
#define ifloat2 CoreVec2<float>
#define iint4 CoreVec4<int>
#define iint3 CoreVec3<int>
#define iint2 CoreVec2<int>
#define iuint4 CoreVec4<unsigned int>
#define iuint3 CoreVec3<unsigned int>
#define iuint2 CoreVec2<unsigned int>

#define irect2Df Rect2D<float>
#define irect2Di Rect2D<int>

template<class T>
struct CoreMat4 {
	T data[4][4];
	const T* operator[](int i) const {
		return data[i];
	}
	T* operator[](int i) {
		return data[i];
	}
};

#define float4x4 CoreMat4<float>
