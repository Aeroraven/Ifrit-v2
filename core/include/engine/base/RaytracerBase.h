#pragma once
#include "math/VectorOps.h"

namespace Ifrit::Engine {
	struct Ray {
		ifloat3 o, r;
	};

	struct RayInternal {
		ifloat3 o, r, invr;
	};

	struct RayHit {
		ifloat3 p;
		float t;
		int id;
	};
	
	class AccelerationStructure {
	public:
		virtual RayHit queryIntersection(const RayInternal& ray, float tmin, float tmax) const = 0;
		virtual void buildAccelerationStructure() = 0;
	};

	template <class T>
	class BufferredAccelerationStructure : public AccelerationStructure {
	public:
		virtual void bufferData(const std::vector<T>& data) = 0;
	};

	struct AccelerationStructureRef {
		AccelerationStructure* ref;
	};
}