#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/RaytracerBase.h"

int getProfileCnt();

namespace Ifrit::Engine::Raytracer {

	namespace Impl {
		class BoundingVolumeHierarchyBottomLevelASImpl;
		class BoundingVolumeHierarchyTopLevelASImpl;
	}

	class IFRIT_APIDECL BoundingVolumeHierarchyBottomLevelAS : public BufferredAccelerationStructure<ifloat3> {
	private:
		Impl::BoundingVolumeHierarchyBottomLevelASImpl* impl = nullptr;
	public:
		friend class Impl::BoundingVolumeHierarchyTopLevelASImpl;
		BoundingVolumeHierarchyBottomLevelAS();
		~BoundingVolumeHierarchyBottomLevelAS();
		virtual void bufferData(const std::vector<ifloat3>& data);
		virtual RayHit queryIntersection(const RayInternal& ray, float tmin, float tmax) const;
		virtual RayHit queryIntersectionFromTLAS(const RayInternal& ray, float tmin, float tmax) const;
		virtual void buildAccelerationStructure();
	};

	class IFRIT_APIDECL BoundingVolumeHierarchyTopLevelAS : public BufferredAccelerationStructure<BoundingVolumeHierarchyBottomLevelAS*> {
	private:
		Impl::BoundingVolumeHierarchyTopLevelASImpl* impl = nullptr;
	public:
		BoundingVolumeHierarchyTopLevelAS();
		~BoundingVolumeHierarchyTopLevelAS();
		virtual void bufferData(const std::vector<BoundingVolumeHierarchyBottomLevelAS*>& data);
		virtual RayHit queryIntersection(const RayInternal& ray, float tmin, float tmax) const;
		virtual void buildAccelerationStructure();
	};
}