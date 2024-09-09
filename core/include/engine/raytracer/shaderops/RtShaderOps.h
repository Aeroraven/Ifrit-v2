#pragma once
#include "core/definition/CoreExports.h"

extern "C" {
	struct IFRIT_APIDECL_FORCED ifritShaderOps_Raytracer_Vec3 {
		float x, y, z;
	};
	struct IFRIT_APIDECL_FORCED ifritShaderOps_Raytracer_Struct_AccelerationStructure {
		uint64_t accelStructRef;
	};
	
	// Arguments conform to https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_ray_tracing.asciidoc
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
	);
}
