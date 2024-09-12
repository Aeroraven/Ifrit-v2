#pragma once
#include "core/definition/CoreExports.h"

extern "C" {
	struct IFRIT_APIDECL_FORCED  alignas(16) ifritShaderOps_Raytracer_Vec3 {
		float x, y, z;
	};

	// Arguments conform to https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_ray_tracing.asciidoc
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
	);
}
