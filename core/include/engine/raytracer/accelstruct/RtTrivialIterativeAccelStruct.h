#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/RaytracerBase.h"

namespace Ifrit::Engine::Raytracer {

	namespace Impl {
		class TrivialBottomLevelASImpl;
		class TrivialTopLevelASImpl;
	}

	class IFRIT_APIDECL TrivialBottomLevelAS : public BufferredAccelerationStructure<ifloat3> {
	private:
		Impl::TrivialBottomLevelASImpl* impl = nullptr;
	public:
		friend class Impl::TrivialTopLevelASImpl;
		TrivialBottomLevelAS();
		virtual void bufferData(const std::vector<ifloat3>& data);
		virtual RayHit queryIntersection(const RayInternal& ray, float tmin, float tmax) const;
		virtual void buildAccelerationStructure();
	};

	class IFRIT_APIDECL TrivialTopLevelAS : public BufferredAccelerationStructure<TrivialBottomLevelAS*> {
	private:
		Impl::TrivialTopLevelASImpl* impl = nullptr;
	public:
		TrivialTopLevelAS();
		virtual void bufferData(const std::vector<TrivialBottomLevelAS*>& data);
		virtual RayHit queryIntersection(const RayInternal& ray, float tmin, float tmax) const;
		virtual void buildAccelerationStructure();
	};
}