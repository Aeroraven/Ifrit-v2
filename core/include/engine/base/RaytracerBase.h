#pragma once
#include "math/VectorOps.h"
#include "math/simd/SimdVectors.h"

namespace Ifrit::Engine {
	struct Ray {
		Ifrit::Math::SIMD::vfloat3 o;
		Ifrit::Math::SIMD::vfloat3 r;
	};

	struct RayInternal {
		Ifrit::Math::SIMD::vfloat3 o;
		Ifrit::Math::SIMD::vfloat3 r;
		Ifrit::Math::SIMD::vfloat3 invr;
		Ifrit::Math::SIMD::vfloat3 neg_invr_o;
	};

	struct RayHit {
		ifloat3 p;
		float t;
		int id;
	};

	template <class T>
	class BufferredAccelerationStructure {
	public:
		virtual RayHit queryIntersection(const RayInternal& ray, float tmin, float tmax) const = 0;
		virtual void buildAccelerationStructure() = 0;
		virtual void bufferData(const std::vector<T>& data) = 0;
	};

}