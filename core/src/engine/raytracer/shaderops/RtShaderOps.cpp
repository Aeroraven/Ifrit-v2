#include "engine/raytracer/shaderops/RtShaderOps.h"
#include "engine/raytracer/TrivialRaytracerWorker.h"
#include "engine/base/RaytracerBase.h"
extern "C" {
	IFRIT_APIDECL_FORCED void ifritShaderOps_Raytracer_TraceRay(
		ifritShaderOps_Raytracer_Struct_AccelerationStructure accelStruct,
		int rayFlag,
		int cullMask,
		int sbtOffset,
		int sbtStride,
		int missIndex,
		ifritShaderOps_Raytracer_Vec3 rayOrigin,
		float rayTmin,
		ifritShaderOps_Raytracer_Vec3 rayDirection,
		float rayTmax,
		void* payload,

		// contextual arguments
		size_t payloadSize,
		void* context,
		int recurDepth
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
		worker->tracingRecursiveProcess(ray, payload, payloadSize, worker->getTracingDepth() + 1);
	}
}