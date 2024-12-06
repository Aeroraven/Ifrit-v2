
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

// Some constants used for both cpp and shader code:

#ifndef SYARO_SHADER_SHARED_CONST_H
#define SYARO_SHADER_SHARED_CONST_H

#ifdef __cplusplus
#include <cstdint>
#define SYARO_DEFINE_UINT(x, y) constexpr uint32_t x = y;
#define SYARO_DEFINE_FLOAT(x, y) constexpr float x = y;

namespace Ifrit::Core::SyaroConfig {

#else
#define SYARO_DEFINE_UINT(x, y) const uint x = y
#define SYARO_DEFINE_FLOAT(x, y) const float x = y

#endif

SYARO_DEFINE_UINT(cPersistentCullThreadGroupSizeX, 128);
SYARO_DEFINE_UINT(cInstanceCullingThreadGroupSizeX, 64);

SYARO_DEFINE_UINT(cMeshRasterizeThreadGroupSizeX, 128);
SYARO_DEFINE_UINT(cMeshRasterizeMaxVertexSize, 128);
SYARO_DEFINE_UINT(cMeshRasterizeMaxPrimitiveSize, 128);
SYARO_DEFINE_UINT(cMeshRasterizeTaskPayloadSize, 32);
SYARO_DEFINE_UINT(cMeshRasterizeTaskThreadGroupSize, 128);

SYARO_DEFINE_UINT(cEmitDepthTargetThreadGroupSizeX, 16);
SYARO_DEFINE_UINT(cEmitDepthTargetThreadGroupSizeY, 16);

SYARO_DEFINE_UINT(cHiZThreadGroupSize, 256);

SYARO_DEFINE_UINT(cClassifyMaterialCountThreadGroupSizeX, 8);
SYARO_DEFINE_UINT(cClassifyMaterialCountThreadGroupSizeY, 8);

SYARO_DEFINE_UINT(cClassifyMaterialScatterThreadGroupSizeX, 8);
SYARO_DEFINE_UINT(cClassifyMaterialScatterThreadGroupSizeY, 8);

SYARO_DEFINE_UINT(cEmitGbufThreadGroupSizeX, 128);

SYARO_DEFINE_UINT(cAtmoRenderThreadGroupSizeX, 16);
SYARO_DEFINE_UINT(cAtmoRenderThreadGroupSizeY, 16);

#define SYARO_DEFERRED_SHADOW_MAPPING_HALTON_PCF_SAMPLING 1
#define SYARO_DEFERRED_SHADOW_MAPPING_HALTON_PCF_NUM_SAMPLES 32

#ifdef __cplusplus
} // namespace Ifrit::Core::Syaro
#undef SYARO_DEFINE_UINT
#undef SYARO_DEFINE_FLOAT
#endif

#endif