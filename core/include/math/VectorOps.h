#pragma once
#include "./core/definition/CoreTypes.h"

namespace Ifrit::Math {
	// Element wise ops
#define ELEMENTWISE_VECTOR_OP(op) \
	template <typename T> \
	inline CoreVec2<T> operator op(const CoreVec2<T>& a, const CoreVec2<T>& b) { \
		return CoreVec2<T>{a.x op b.x, a.y op b.y}; \
	} \
	template <typename T> \
	inline CoreVec3<T> operator op(const CoreVec3<T>& a, const CoreVec3<T>& b) { \
		return CoreVec3<T>{a.x op b.x, a.y op b.y, a.z op b.z}; \
	} \
	template <typename T> \
	inline CoreVec4<T> operator op(const CoreVec4<T>& a, const CoreVec4<T>& b) { \
		return CoreVec4<T>{a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w}; \
	}
	ELEMENTWISE_VECTOR_OP(+);
	ELEMENTWISE_VECTOR_OP(-);
	ELEMENTWISE_VECTOR_OP(*);
	ELEMENTWISE_VECTOR_OP(/);
#undef ELEMENTWISE_VECTOR_OP

	// Scalar ops
#define SCALAR_VECTOR_OP(op) \
	template <typename T> \
	inline CoreVec2<T> operator op(const CoreVec2<T>& a, T b) { \
		return CoreVec2<T>{a.x op b, a.y op b}; \
	} \
	template <typename T> \
	inline CoreVec3<T> operator op(const CoreVec3<T>& a, T b) { \
		return CoreVec3<T>{a.x op b, a.y op b, a.z op b}; \
	} \
	template <typename T> \
	inline CoreVec4<T> operator op(const CoreVec4<T>& a, T b) { \
		return CoreVec4<T>{a.x op b, a.y op b, a.z op b, a.w op b}; \
	}
	SCALAR_VECTOR_OP(+);
	SCALAR_VECTOR_OP(-);
	SCALAR_VECTOR_OP(*);
	SCALAR_VECTOR_OP(/);
#undef SCALAR_VECTOR_OP

	// Element wise ops 2

#define ELEMENTWISE_VECTOR_OP2(op) \
	template <typename T> \
	inline CoreVec2<T>& operator op(CoreVec2<T>& a, const CoreVec2<T>& b) { \
		a.x op b.x; \
		a.y op b.y; \
		return a; \
	} \
	template <typename T> \
	inline CoreVec3<T>& operator op(CoreVec3<T>& a, const CoreVec3<T>& b) { \
		a.x op b.x; \
		a.y op b.y; \
		a.z op b.z; \
		return a; \
	} \
	template <typename T> \
	inline CoreVec4<T>& operator op(CoreVec4<T>& a, const CoreVec4<T>& b) { \
		a.x op b.x; \
		a.y op b.y; \
		a.z op b.z; \
		a.w op b.w; \
		return a; \
	}
	ELEMENTWISE_VECTOR_OP2(+=);
	ELEMENTWISE_VECTOR_OP2(-=);
	ELEMENTWISE_VECTOR_OP2(*=);
	ELEMENTWISE_VECTOR_OP2(/=);

#undef ELEMENTWISE_VECTOR_OP2

	// Scalar ops 2
#define SCALAR_VECTOR_OP2(op) \
	template <typename T> \
	inline CoreVec2<T>& operator op(CoreVec2<T>& a, T b) { \
		a.x op b; \
		a.y op b; \
		return a; \
	} \
	template <typename T> \
	inline CoreVec3<T>& operator op(CoreVec3<T>& a, T b) { \
		a.x op b; \
		a.y op b; \
		a.z op b; \
		return a; \
	} \
	template <typename T> \
	inline CoreVec4<T>& operator op(CoreVec4<T>& a, T b) { \
		a.x op b; \
		a.y op b; \
		a.z op b; \
		a.w op b; \
		return a; \
	}

	SCALAR_VECTOR_OP2(+=);
	SCALAR_VECTOR_OP2(-=);
	SCALAR_VECTOR_OP2(*=);
	SCALAR_VECTOR_OP2(/=);

#undef SCALAR_VECTOR_OP2

	// Address ops
	template <typename T>
	IFRIT_APIDECL inline T elementAt(CoreVec2<T>& a, int i) {
		return i == 0 ? a.x : a.y;
	}
	template <typename T>
	IFRIT_APIDECL inline T elementAt(CoreVec3<T>& a, int i) {
		return i == 0 ? a.x : i == 1 ? a.y : a.z;
	}
	template <typename T>
	IFRIT_APIDECL inline T elementAt(CoreVec4<T>& a, int i) {
		return i == 0 ? a.x : i == 1 ? a.y : i == 2 ? a.z : a.w;
	}

	// Normalize ops
	template <typename T>
	IFRIT_APIDECL inline CoreVec2<T> normalize(const CoreVec2<T>& a) {
		T len = sqrt(a.x * a.x + a.y * a.y);
		return CoreVec2<T>{a.x / len, a.y / len};
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec3<T> normalize(const CoreVec3<T>& a) {
		T len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
		return CoreVec3<T>{a.x / len, a.y / len, a.z / len};
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec4<T> normalize(const CoreVec4<T>& a) {
		T len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
		return CoreVec4<T>{a.x / len, a.y / len, a.z / len, a.w / len};
	}

	// Length ops
	template <typename T>
	IFRIT_APIDECL inline T length(const CoreVec2<T>& a) {
		return sqrt(a.x * a.x + a.y * a.y);
	}
	template <typename T>
	IFRIT_APIDECL inline T length(const CoreVec3<T>& a) {
		return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
	}
	template <typename T>
	IFRIT_APIDECL inline T length(const CoreVec4<T>& a) {
		return sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
	}

	// Dot ops
	template <typename T>
	IFRIT_APIDECL inline T dot(const CoreVec2<T>& a, const CoreVec2<T>& b) {
		return a.x * b.x + a.y * b.y;
	}
	template <typename T>
	IFRIT_APIDECL inline T dot(const CoreVec3<T>& a, const CoreVec3<T>& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	template <typename T>
	IFRIT_APIDECL inline T dot(const CoreVec4<T>& a, const CoreVec4<T>& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	// Lerp ops
	template <typename T>
	IFRIT_APIDECL inline CoreVec2<T> lerp(const CoreVec2<T>& a, const CoreVec2<T>& b, const T& t) {
		return a * (1 - t) + b * t;
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec3<T> lerp(const CoreVec3<T>& a, const CoreVec3<T>& b, const T& t) {
		return a * (1 - t) + b * t;
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec4<T> lerp(const CoreVec4<T>& a, const CoreVec4<T>& b, const T& t) {
		return a * (1 - t) + b * t;
	}

	// Clamp ops
	template <typename T>
	IFRIT_APIDECL inline CoreVec2<T> clamp(const CoreVec2<T>& a, const T& min, const T& max) {
		return CoreVec2<T>{std::clamp(a.x, min, max), std::clamp(a.y, min, max)};
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec3<T> clamp(const CoreVec3<T>& a, const T& min, const T& max) {
		return CoreVec3<T>{std::clamp(a.x, min, max), std::clamp(a.y, min, max), std::clamp(a.z, min, max)};
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec4<T> clamp(const CoreVec4<T>& a, const T& min, const T& max) {
		return CoreVec4<T>{std::clamp(a.x, min, max), std::clamp(a.y, min, max), std::clamp(a.z, min, max), std::clamp(a.w, min, max)};
	}

	// Cross ops
	template <typename T>
	IFRIT_APIDECL inline CoreVec3<T> cross(const CoreVec3<T>& a, const CoreVec3<T>& b) {
		return CoreVec3<T>{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
	}

	// Distance ops
	template <typename T>
	IFRIT_APIDECL inline T distance(const CoreVec2<T>& a, const CoreVec2<T>& b) {
		return length(a - b);
	}
	template <typename T>
	IFRIT_APIDECL inline T distance(const CoreVec3<T>& a, const CoreVec3<T>& b) {
		return length(a - b);
	}
	template <typename T>
	IFRIT_APIDECL inline T distance(const CoreVec4<T>& a, const CoreVec4<T>& b) {
		return length(a - b);
	}

	// Min ops
	template <typename T>
	IFRIT_APIDECL inline CoreVec2<T> min(const CoreVec2<T>& a, const CoreVec2<T>& b) {
		return CoreVec2<T>{std::min(a.x, b.x), std::min(a.y, b.y)};
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec3<T> min(const CoreVec3<T>& a, const CoreVec3<T>& b) {
		return CoreVec3<T>{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec4<T> min(const CoreVec4<T>& a, const CoreVec4<T>& b) {
		return CoreVec4<T>{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w)};
	}

	// Max ops
	template <typename T>
	IFRIT_APIDECL inline CoreVec2<T> max(const CoreVec2<T>& a, const CoreVec2<T>& b) {
		return CoreVec2<T>{std::max(a.x, b.x), std::max(a.y, b.y)};
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec3<T> max(const CoreVec3<T>& a, const CoreVec3<T>& b) {
		return CoreVec3<T>{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
	}
	template <typename T>
	IFRIT_APIDECL inline CoreVec4<T> max(const CoreVec4<T>& a, const CoreVec4<T>& b) {
		return CoreVec4<T>{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w)};
	}


}
