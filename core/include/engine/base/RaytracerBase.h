#pragma once
#include "math/VectorOps.h"

namespace Ifrit::Engine {
	struct Ray {
		ifloat3 o, r;
	};

	struct RayHit {
		ifloat3 p;
		float t;
		int id;
	};
	
	class AccelerationStructure {
	public:
		virtual RayHit queryIntersection(const Ray& ray) const = 0;
		virtual void buildAccelerationStructure() = 0;
	};

	template <class T>
	class BufferredAccelerationStructure : public AccelerationStructure {
	public:
		virtual void bufferData(const std::vector<T>& data) = 0;
	};
}