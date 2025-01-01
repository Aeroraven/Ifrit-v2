
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

// This macro enables the culling in triangle-level, which sacrifices
// max-occupancy (shared memory) for better culling efficiency. However, the
// trade-off is not always beneficial.
#define SYARO_SHADER_SHARED_VISBUFFER_ENABLE_PRIMCULL 0

// This macro allows the culling frustum to be smaller than the actual view
// frustum, for CSM, view frustum culling is larger than the actual view frustum
// to avoid flickering. However, this makes about 107% useless area to be
// rendered. This macro intends to reduce the culling frustum to the actual view
// frustum.
#define SYARO_SHADER_SHARED_EXPLICIT_ORTHO_FRUSTUM_CULL 1

// This macro determines whether to reuse the calculated triangle data during
// the GBuffer emission. A thread will process 4 pixels (quad) at a time.
// This sacrifices warp occupancy.
#define SYARO_SHADER_SHARED_EMIT_GBUFFER_TRIANGLE_REUSE 1

// Whether to move triangle culling from amplification shader to the persistent
// cull stage. This drops amplification shader.
#define SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL 1

// Whether to enable SW rasterizer
#define SYARO_ENABLE_SW_RASTERIZER 1

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

SYARO_DEFINE_UINT(cEmitGbufThreadGroupSizeX, 256);

SYARO_DEFINE_UINT(cAtmoRenderThreadGroupSizeX, 16);
SYARO_DEFINE_UINT(cAtmoRenderThreadGroupSizeY, 16);

SYARO_DEFINE_UINT(cPersistentCullParallelStg_PersistThread, 1);
SYARO_DEFINE_UINT(cPersistentCullParallelStg_StridedLoop_BVH, 2);
SYARO_DEFINE_UINT(cPersistentCullParallelStg_StridedLoop_ClusterGroup, 3);

SYARO_DEFINE_UINT(cPersistentCullParallelStg,
                  cPersistentCullParallelStg_StridedLoop_ClusterGroup);

SYARO_DEFINE_UINT(cCombineVisBufferThreadGroupSizeX, 16);
SYARO_DEFINE_UINT(cCombineVisBufferThreadGroupSizeY, 16);

#define SYARO_DEFERRED_SHADOW_MAPPING_HALTON_PCF_SAMPLING 1
#define SYARO_DEFERRED_SHADOW_MAPPING_HALTON_PCF_NUM_SAMPLES 32

#ifdef __cplusplus
} // namespace Ifrit::Core::Syaro
#undef SYARO_DEFINE_UINT
#undef SYARO_DEFINE_FLOAT
#endif

#endif