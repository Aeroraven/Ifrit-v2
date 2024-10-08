#include "engine/raytracer/shaderops/RtShaderOps.h"
#include "engine/raytracer/TrivialRaytracerWorker.h"
#include "engine/base/RaytracerBase.h"
#include "math/simd/SimdVectors.h"
using namespace Ifrit::Math::SIMD;
extern "C" {
	IFRIT_APIDECL_FORCED void ifritShaderOps_Raytracer_TraceRay(
		ifritShaderOps_Raytracer_Vec3 rayOrigin,
		void* accelStruct,
		int rayFlag,
		int cullMask,
		int sbtOffset,
		int sbtStride,
		int missIndex,
		float rayTmin,
		ifritShaderOps_Raytracer_Vec3 rayDirection,
		float rayTmax,
		void* payload,

		// contextual arguments
		void* context
	) {
		using namespace Ifrit::Engine;
		using namespace Ifrit::Engine::Raytracer;
		
		RayInternal ray;
		ray.o = vfloat3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
		ray.r = vfloat3(rayDirection.x, rayDirection.y, rayDirection.z);
		ray.invr = reciprocal(ray.r);
		auto worker = reinterpret_cast<TrivialRaytracerWorker*>(context);
		worker->tracingRecursiveProcess(ray, payload, worker->getTracingDepth() + 1, rayTmin, rayTmax);
	}
}