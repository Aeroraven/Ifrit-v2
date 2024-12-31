
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
#include "ifrit/softgraphics/core/data/Image.h"

#if defined(__GNUC__) || defined(__MINGW32__) || defined(__MINGW64__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wattributes"
#endif

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
IFRIT_APIDECL_FORCED struct alignas(16) ifritShaderOps_Base_Veci2 {
  int x, y;
};

IFRIT_APIDECL_FORCED void
ifritShaderOps_Base_ImageWrite_v2i32_v4f32(void *pImage,
                                           ifritShaderOps_Base_Veci2 coord,
                                           ifritShaderOps_Base_Vecf4 color);

IFRIT_APIDECL_FORCED void ifritShaderOps_Base_ImageSampleExplicitLod_2d_v4f32(
    void *pSampledImage, ifritShaderOps_Base_Veci2 coord, float lod,
    ifritShaderOps_Base_Vecf4 *result);
}

#if defined(__GNUC__) || defined(__MINGW32__) || defined(__MINGW64__)
#pragma GCC diagnostic pop
#endif
