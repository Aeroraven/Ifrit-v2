#pragma once
#include "core/definition/CoreExports.h"
#include "engine/raytracer/TrivialRaytracerWorker.h"

#if defined(__GNUC__) || defined(__MINGW32__) || defined(__MINGW64__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wattributes"
#endif

extern "C" {
IFRIT_APIDECL_FORCED struct alignas(16) ifritShaderOps_Raytracer_Vec3 {
  float x, y, z;
};

// Arguments conform to
// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_ray_tracing.asciidoc
IFRIT_APIDECL_FORCED void ifritShaderOps_Raytracer_TraceRay(
    ifritShaderOps_Raytracer_Vec3 rayOrigin, void *accelStruct, int rayFlag,
    int cullMask, int sbtOffset, int sbtStride, int missIndex, float rayTmin,
    ifritShaderOps_Raytracer_Vec3 rayDirection, float rayTmax, void *payload,

    // contextual arguments
    void *context);
}

#if defined(__GNUC__) || defined(__MINGW32__) || defined(__MINGW64__)
#pragma GCC diagnostic pop
#endif