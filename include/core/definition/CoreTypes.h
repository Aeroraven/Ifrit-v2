#pragma once

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

#define float4 CoreVec4<float>
#define float3 CoreVec3<float>
#define float2 CoreVec2<float>
#define int4 CoreVec4<int>
#define int3 CoreVec3<int>
#define int2 CoreVec2<int>
#define uint4 CoreVec4<unsigned int>
#define uint3 CoreVec3<unsigned int>
#define uint2 CoreVec2<unsigned int>

#define rect2Df Rect2D<float>
#define rect2Di Rect2D<int>

template<class T>
struct CoreMat4 {
	T data[4][4];
	T* operator[](int i) const {
		return data[i];
	}
	T* operator[](int i) {
		return data[i];
	}
};

#define float4x4 CoreMat4<float>
