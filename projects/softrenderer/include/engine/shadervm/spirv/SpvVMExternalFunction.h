#pragma once
#include "core/data/Image.h"

extern "C" {
IFRIT_APIDECL_FORCED struct alignas(16) ifritShaderOps_Base_Vecf4 {
  float x, y, z, w;
};
IFRIT_APIDECL_FORCED struct alignas(16) ifritShaderOps_Base_Vecf3 {
  float x, y, z;
};
IFRIT_APIDECL_FORCED struct alignas(16) ifritShaderOps_Base_Vecf2 {
  float x, y;
};
IFRIT_APIDECL_FORCED struct alignas(16) ifritShaderOps_Base_Veci2 { int x, y; };

IFRIT_APIDECL_FORCED void
ifritShaderOps_Base_ImageWrite_v2i32_v4f32(void *pImage,
                                           ifritShaderOps_Base_Veci2 coord,
                                           ifritShaderOps_Base_Vecf4 color);

IFRIT_APIDECL_FORCED void ifritShaderOps_Base_ImageSampleExplicitLod_2d_v4f32(
    void *pSampledImage, ifritShaderOps_Base_Veci2 coord, float lod,
    ifritShaderOps_Base_Vecf4 *result);
}