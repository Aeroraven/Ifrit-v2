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

#ifdef __cplusplus
} // namespace Ifrit::Core::Syaro
#undef SYARO_DEFINE_UINT
#undef SYARO_DEFINE_FLOAT
#endif

#endif