
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

#include "ifrit/softgraphics/engine/raytracer/shaderops/RtShaderOps.h"
#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/softgraphics/engine/base/RaytracerBase.h"
#include "ifrit/softgraphics/engine/raytracer/TrivialRaytracerWorker.h"

using namespace Ifrit::Math::SIMD;
extern "C" {
IFRIT_APIDECL_FORCED void ifritShaderOps_Raytracer_TraceRay(
    ifritShaderOps_Raytracer_Vec3 rayOrigin, void *accelStruct, int rayFlag,
    int cullMask, int sbtOffset, int sbtStride, int missIndex, float rayTmin,
    ifritShaderOps_Raytracer_Vec3 rayDirection, float rayTmax, void *payload,

    // contextual arguments
    void *context) {
  using namespace Ifrit::GraphicsBackend::SoftGraphics;
  using namespace Ifrit::GraphicsBackend::SoftGraphics::Raytracer;

  RayInternal ray;
  ray.o = SVector3f(rayOrigin.x, rayOrigin.y, rayOrigin.z);
  ray.r = SVector3f(rayDirection.x, rayDirection.y, rayDirection.z);
  ray.invr = reciprocal(ray.r);
  auto worker = reinterpret_cast<TrivialRaytracerWorker *>(context);
  worker->tracingRecursiveProcess(ray, payload, worker->getTracingDepth() + 1,
                                  rayTmin, rayTmax);
}
}