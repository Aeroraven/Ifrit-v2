#include "engine/raytracer/shaderops/RtShaderOps.h"
#include "engine/raytracer/TrivialRaytracerWorker.h"
#include "engine/base/RaytracerBase.h"
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
		
		Ray ray;
		ray.o.x = rayOrigin.x;
		ray.o.y = rayOrigin.y;
		ray.o.z = rayOrigin.z;
		ray.r.x = rayDirection.x;
		ray.r.y = rayDirection.y;
		ray.r.z = rayDirection.z;
		auto worker = reinterpret_cast<TrivialRaytracerWorker*>(context);
		//printf("Origin: %f %f %f\n", ray.o.x, ray.o.y, ray.o.z);
		//printf("Direction: %f %f %f\n", ray.r.x, ray.r.y, ray.r.z);
		//printf("RM: %f\n", rayTmax);
		worker->tracingRecursiveProcess(ray, payload, worker->getTracingDepth() + 1);
	}
}