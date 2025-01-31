
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

#ifndef IFRIT_SHADER_AMBIENT_OCCLUSION_SHARED_CONST_H
#define IFRIT_SHADER_AMBIENT_OCCLUSION_SHARED_CONST_H

#ifdef __cplusplus
#include <cstdint>
#define IFRIT_AO_DEFINE_UINT(x, y) constexpr uint32_t x = y;
#define IFRIT_AO_DEFINE_INT(x, y) constexpr int32_t x = y;
#define IFRIT_AO_DEFINE_FLOAT(x, y) constexpr float x = y;
namespace Ifrit::Core::Shaders::AmbientOcclusionConfig {
#else
#define IFRIT_AO_DEFINE_UINT(x, y) const uint x = y
#define IFRIT_AO_DEFINE_INT(x, y) const int x = y
#define IFRIT_AO_DEFINE_FLOAT(x, y) const float x = y
#endif

IFRIT_AO_DEFINE_UINT(cHBAOThreadGroupSizeX, 16);
IFRIT_AO_DEFINE_UINT(cHBAOThreadGroupSizeY, 16);
IFRIT_AO_DEFINE_UINT(cHBAODirections, 8);
IFRIT_AO_DEFINE_UINT(cHBAOSampleSteps, 6);

IFRIT_AO_DEFINE_UINT(cSSGIThreadGroupSizeX, 16);
IFRIT_AO_DEFINE_UINT(cSSGIThreadGroupSizeY, 16);
IFRIT_AO_DEFINE_UINT(cSSGIBounces, 1);
IFRIT_AO_DEFINE_UINT(cSSGISamples, 64);

#define SSGI_USE_HIERARCHICAL_Z 1
#define SSGI_RAY_MAX_DISTANCE 2.0f
#define SSGI_FALLBACK_TO_SSR 0

#ifdef __cplusplus
} // namespace Ifrit::Core::Shaders::AmbientOcclusionConfig
#endif

#endif