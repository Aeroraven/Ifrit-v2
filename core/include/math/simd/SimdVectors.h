#pragma once
#include "core/definition/CoreExports.h"

namespace Ifrit::Math::SIMD {
#ifdef IFRIT_FEATURE_SIMD
	constexpr const auto simdEnable = true;
#else
	constexpr const auto simdEnable = false;
#endif

	template<typename T>
	concept SimdElement32 = requires (T x) { sizeof(T) == 4; };

	template<typename T, typename S, int V>
	requires SimdElement32<T>
	struct alignas(16) SimdVector {
		union {
			struct { T x, y, z, w; };
			S dataf;
		};
		inline SimdVector() = default;
		inline SimdVector(const SimdVector& v) = default;

		inline SimdVector(S data) {
			dataf = data;
		}
		inline SimdVector(T x, T y, T z, T w) {
#ifdef IFRIT_FEATURE_SIMD
			if constexpr (std::is_same_v<S, __m128>) {
				dataf = _mm_set_ps(w, z, y, x);
			}
			else if constexpr (std::is_same_v<S, __m128i>) {
				dataf = _mm_set_epi32(w, z, y, x);
			}
			else {
				static_assert(false, "Unsupported SIMD type");
			}
#else
			this->w = w;
			this->z = z;
			this->y = y;
			this->x = x;
#endif
		
		}
		inline SimdVector(T x, T y, T z) {
#ifdef IFRIT_FEATURE_SIMD
			if constexpr (std::is_same_v<S, __m128>) {
				dataf = _mm_set_ps(0, z, y, x);
			}
			else if constexpr (std::is_same_v<S, __m128i>) {
				dataf = _mm_set_epi32(0, z, y, x);
			}
			else {
				static_assert(false, "Unsupported SIMD type");
			}
#else
			this->w = 0;
			this->z = z;
			this->y = y;
			this->x = x;
#endif
		}
		inline SimdVector(T x, T y) {
#ifdef IFRIT_FEATURE_SIMD
			if constexpr (std::is_same_v<S, __m128>) {
				dataf = _mm_set_ps(0, 0, y, x);
			}
			else if constexpr (std::is_same_v<S, __m128i>) {
				dataf = _mm_set_epi32(0, 0, y, x);
			}
			else {
				static_assert(false, "Unsupported SIMD type");
			}
#else
			this->w = 0;
			this->z = 0;
			this->y = y;
			this->x = x;
#endif

		}
		inline SimdVector(T x) {
#ifdef IFRIT_FEATURE_SIMD
			if constexpr (std::is_same_v<S, __m128>) {
				dataf = _mm_set_ps(x, x, x, x);
			}
			else if constexpr (std::is_same_v<S, __m128i>) {
				dataf = _mm_set_epi32(x, x, x, x);
			}
			else {
				static_assert(false, "Unsupported SIMD type");
			}
#else
			this->w = x;
			this->z = x;
			this->y = x;
			this->x = x;
#endif
		}

#ifdef IFRIT_FEATURE_SIMD
#define SIMD_VECTOR_OPERATOR_1(op,insName) \
		inline SimdVector operator op(const SimdVector& v) const { \
			if constexpr (std::is_same_v<S, __m128>) return _mm_##insName##_ps(dataf, v.dataf); \
			else if constexpr (std::is_same_v<S, __m128i>) return _mm_##insName##_epi32(dataf, v.dataf); \
			else  static_assert(false, "Unsupported SIMD type"); \
		}
#define SIMD_VECTOR_OPERATOR_2(op,insName) \
		inline SimdVector& operator op(const SimdVector& v) { \
			if constexpr (std::is_same_v<S, __m128>) dataf = _mm_##insName##_ps(dataf, v.dataf); \
			else if constexpr (std::is_same_v<S, __m128i>) dataf = _mm_##insName##_epi32(dataf, v.dataf); \
			else  static_assert(false, "Unsupported SIMD type"); \
			return *this; \
		}
#else
#define SIMD_VECTOR_OPERATOR_1(op,insName) \
		inline SimdVector operator op(const SimdVector& v) const { \
			SimdVector r; \
			if constexpr (V >= 4) r.w = w op v.w; \
			if constexpr (V >= 3) r.z = z op v.z; \
			if constexpr (V >= 2) r.y = y op v.y; \
			r.x = x op v.x; \
			return r; \
		}
#define SIMD_VECTOR_OPERATOR_2(op,insName) \
		inline SimdVector& operator op(const SimdVector& v) { \
			if constexpr (V >= 4) w op v.w; \
			if constexpr (V >= 3) z op v.z; \
			if constexpr (V >= 2) y op v.y; \
			x op v.x; \
			return *this; \
		}
#endif
		SIMD_VECTOR_OPERATOR_1(+, add);
		SIMD_VECTOR_OPERATOR_1(-, sub);
		SIMD_VECTOR_OPERATOR_1(*, mul)
		SIMD_VECTOR_OPERATOR_1(/, div);
		SIMD_VECTOR_OPERATOR_2(+=, add);
		SIMD_VECTOR_OPERATOR_2(-=, sub);
		SIMD_VECTOR_OPERATOR_2(*=, mul);
		SIMD_VECTOR_OPERATOR_2(/=, div);

		inline SimdVector& cross_(const SimdVector& v) {
#ifdef IFRIT_FEATURE_SIMD
			if constexpr (std::is_same_v<S, __m128>) {
				dataf = _mm_sub_ps(
					_mm_mul_ps(_mm_shuffle_ps(dataf, dataf, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(v.dataf, v.dataf, _MM_SHUFFLE(3, 1, 0, 2))),
					_mm_mul_ps(_mm_shuffle_ps(dataf, dataf, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(v.dataf, v.dataf, _MM_SHUFFLE(3, 0, 2, 1)))
				);
			}
			else {
				static_assert(false, "Unsupported SIMD type");
			}
#else
			T tx = x, ty = y, tz = z;
			x = ty * v.z - tz * v.y;
			y = tz * v.x - tx * v.z;
			z = tx * v.y - ty * v.x;
#endif
			return *this;
		}

		inline T dot(const SimdVector& v) const {
			T r = 0;
#ifdef IFRIT_FEATURE_SIMD
			if constexpr (std::is_same_v<S, __m128>) {
				if constexpr (V == 4) r = _mm_cvtss_f32(_mm_dp_ps(dataf, v.dataf, 0xF1));
				else if constexpr (V == 3) r = _mm_cvtss_f32(_mm_dp_ps(dataf, v.dataf, 0x71));
				else if constexpr (V == 2) r = _mm_cvtss_f32(_mm_dp_ps(dataf, v.dataf, 0x31));
				else r = _mm_cvtss_f32(_mm_dp_ps(dataf, v.dataf, 0x11));
			}
			else {
				static_assert(false, "Unsupported SIMD type");
			}
#else
			if constexpr (V >= 4) r += w * v.w;
			if constexpr (V >= 3) r += z * v.z;
			if constexpr (V >= 2) r += y * v.y;
			r += x * v.x;
#endif
			return r;
		}
	};

	// Crosss
	template<typename T, typename S, int V>
	inline SimdVector<T, S, V> cross(const SimdVector<T, S, V>& a, const SimdVector<T, S, V>& b){
		SimdVector<T, S, V> r;
		static_assert(V == 3, "Cross product is only defined for 3D vectors");
#ifdef IFRIT_FEATURE_SIMD
	if constexpr (std::is_same_v<S, __m128>) {
		r.dataf = _mm_sub_ps(
			_mm_mul_ps(_mm_shuffle_ps(a.dataf, a.dataf, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b.dataf, b.dataf, _MM_SHUFFLE(3, 1, 0, 2))),
			_mm_mul_ps(_mm_shuffle_ps(a.dataf, a.dataf, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b.dataf, b.dataf, _MM_SHUFFLE(3, 0, 2, 1)))
		);
	}
	else {
		static_assert(false, "Unsupported SIMD type");
	}
#else
		r.x = a.y * b.z - a.z * b.y;
		r.y = a.z * b.x - a.x * b.z;
		r.z = a.x * b.y - a.y * b.x;
#endif
	return r;
	}
	
	// FMA
	template<typename T, typename S, int V>
	inline SimdVector<T, S, V> fma(const SimdVector<T, S, V>& a, const SimdVector<T, S, V>& b, const SimdVector<T, S, V>& c) {
		SimdVector<T, S, V> r;
#ifdef IFRIT_FEATURE_SIMD
		if constexpr (std::is_same_v<S, __m128>) {
			r.dataf = _mm_fmadd_ps(a.dataf, b.dataf, c.dataf);
		}
		else {
			static_assert(false, "Unsupported SIMD type");
		}
#endif
		return r;
	}

	// Dot
	template<typename T, typename S, int V>
	inline T dot(const SimdVector<T, S, V>& a, const SimdVector<T, S, V>& b) {
		T r = 0;
#ifdef IFRIT_FEATURE_SIMD
		if constexpr (std::is_same_v<S, __m128>) {
			if constexpr (V == 4) r = _mm_cvtss_f32(_mm_dp_ps(a.dataf, b.dataf, 0xF1));
			else if constexpr (V == 3)r = _mm_cvtss_f32(_mm_dp_ps(a.dataf, b.dataf, 0x71));
			else if constexpr (V == 2) r = _mm_cvtss_f32(_mm_dp_ps(a.dataf, b.dataf, 0x31));
			else r = _mm_cvtss_f32(_mm_dp_ps(a.dataf, b.dataf, 0x11));
		}
		else {
			static_assert(false, "Unsupported SIMD type");
		}
#else
		if constexpr (V >= 4) r += a.w * b.w;
		if constexpr (V >= 3) r += a.z * b.z;
		if constexpr (V >= 2) r += a.y * b.y;
		r += a.x * b.x;
#endif
		return r;
	}

	// Element-wise max
	template<typename T, typename S, int V>
	inline SimdVector<T, S, V> max(const SimdVector<T, S, V>& a, const SimdVector<T, S, V>& b) {
		SimdVector<T, S, V> r;
#ifdef IFRIT_FEATURE_SIMD
		if constexpr (std::is_same_v<S, __m128>) {
			r.dataf = _mm_max_ps(a.dataf, b.dataf);
		}
		else {
			static_assert(false, "Unsupported SIMD type");
		}
#else
		if constexpr (V >= 4) r.w = std::max(a.w, b.w);
		if constexpr (V >= 3) r.z = std::max(a.z, b.z);
		if constexpr (V >= 2) r.y = std::max(a.y, b.y);
		r.x = std::max(a.x, b.x);
#endif
		return r;
	}

	// Element-wise min
	template<typename T, typename S, int V>
	inline SimdVector<T, S, V> min(const SimdVector<T, S, V>& a, const SimdVector<T, S, V>& b) {
		SimdVector<T, S, V> r;
#ifdef IFRIT_FEATURE_SIMD
		if constexpr (std::is_same_v<S, __m128>) {
			r.dataf = _mm_min_ps(a.dataf, b.dataf);
		}
		else {
			static_assert(false, "Unsupported SIMD type");
		}
#else
		if constexpr (V >= 4) r.w = std::min(a.w, b.w);
		if constexpr (V >= 3) r.z = std::min(a.z, b.z);
		if constexpr (V >= 2) r.y = std::min(a.y, b.y);
		r.x = std::min(a.x, b.x);
#endif
		return r;
	}

	// elementAt
	template<typename T, typename S, int V>
	inline T elementAt(const SimdVector<T, S, V>& a, int index) {
		if constexpr (V == 4) {
			if (index == 0) return a.x;
			if (index == 1) return a.y;
			if (index == 2) return a.z;
			if (index == 3) return a.w;
		}
		else if constexpr (V == 3) {
			if (index == 0) return a.x;
			if (index == 1) return a.y;
			if (index == 2) return a.z;
		}
		else if constexpr (V == 2) {
			if (index == 0) return a.x;
			if (index == 1) return a.y;
		}
		else if constexpr (V == 1) {
			if (index == 0) return a.x;
		}
		else {
			static_assert(false, "Unsupported SIMD type");
		}
	}

	// lerp
	template<typename T, typename S, int V>
	inline SimdVector<T, S, V> lerp(const SimdVector<T, S, V>& a, const SimdVector<T, S, V>& b, const T& t) {
		SimdVector<T, S, V> r;
#ifdef IFRIT_FEATURE_SIMD
		if constexpr (std::is_same_v<S, __m128>) {
			r.dataf = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(b.dataf, a.dataf), _mm_set1_ps(t)), a.dataf);
		}
		else {
			static_assert(false, "Unsupported SIMD type");
		}
#else
		if constexpr (V >= 4) r.w = a.w + (b.w - a.w) * t;
		if constexpr (V >= 3) r.z = a.z + (b.z - a.z) * t;
		if constexpr (V >= 2) r.y = a.y + (b.y - a.y) * t;
		r.x = a.x + (b.x - a.x) * t;
#endif
		return r;
	}

	// Exporting Types
	using vfloat3 = SimdVector<float, __m128, 3>;
	using vfloat4 = SimdVector<float, __m128, 4>;
	using vint3 = SimdVector<int, __m128i, 3>;
	using vint4 = SimdVector<int, __m128i, 4>;
}
