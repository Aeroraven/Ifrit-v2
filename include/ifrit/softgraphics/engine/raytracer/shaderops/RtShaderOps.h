
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/raytracer/TrivialRaytracerWorker.h"

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