
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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

#ifndef AYANAMI_SHADER_SHARED_CONST_H
#define AYANAMI_SHADER_SHARED_CONST_H

#ifdef __cplusplus
    #include <cstdint>
    #include "ifrit/core/base/IfritBase.h"
    #define AYANAMI_DEFINE_UINT(x, y) IF_CONSTEXPR uint32_t x = y;
    #define AYANAMI_DEFINE_FLOAT(x, y) IF_CONSTEXPR float x = y;

namespace Ifrit::Runtime::Ayanami::Config
{

#else
    #define AYANAMI_DEFINE_UINT(x, y) const uint x = y
    #define AYANAMI_DEFINE_FLOAT(x, y) const float x = y
#endif

#define AYANAMI_OBJECT_GRID_CULL 0

    // SDF expanding. This is used to expand the SDF volume to avoid artifacts in thin geometry.
    AYANAMI_DEFINE_UINT(kAyanami_SDFExpand, 1);
    AYANAMI_DEFINE_FLOAT(kAyanami_SDFExpandRatio, 0.1f);

    // Object grid constants. Represents the alternative to the voxel lighting datastructure.
    AYANAMI_DEFINE_UINT(kAyanami_MaxObjectPerGridCell, 4);
    AYANAMI_DEFINE_FLOAT(kAyanami_ObjectGridCellQueryInterpolationRange, 3.0f);
    // might affect shared memory x_x! & 1 for atomic
    AYANAMI_DEFINE_UINT(kAyanami_ObjectGridCellMaxCullObjPerPass, 511);

    // Radiosity processing
    AYANAMI_DEFINE_UINT(kAyanami_CardTileWidth, 8);
    AYANAMI_DEFINE_UINT(kAyanami_RadiosityProbesPerCardTileWidth, 2);
    AYANAMI_DEFINE_UINT(kAyanami_RadiosityTracesPerCardTile, 64);
    AYANAMI_DEFINE_UINT(kAyanami_RadiosityTracesPerProbe, 16);
    AYANAMI_DEFINE_UINT(kAyanami_RadiosityTracesPerProbeSqrt, 4);
    AYANAMI_DEFINE_UINT(kAyanami_RadiosityProbHemiRes, 4);

    // Kernel Sizes
    AYANAMI_DEFINE_UINT(kAyanamiGlobalDFCompositeTileSize, 8);
    AYANAMI_DEFINE_UINT(kAyanamiGlobalDFRayMarchTileSize, 16);

    AYANAMI_DEFINE_UINT(kAyanamiShadowVisibilityObjectsPerBlock, 8);
    AYANAMI_DEFINE_UINT(kAyanamiShadowVisibilityCardSizePerBlock, 8);

    AYANAMI_DEFINE_UINT(kAyanamiObjectGridTileSize, 4);
    AYANAMI_DEFINE_UINT(kAyanamiRadiosityTraceKernelSize, kAyanami_RadiosityTracesPerCardTile);

    AYANAMI_DEFINE_UINT(kAyanamiReconFromSCTileSize, 8);
    AYANAMI_DEFINE_UINT(kAyanamiReconFromSCDepthTileSize, 8);

    AYANAMI_DEFINE_UINT(kAyanamiSCDirectLightObjectsPerBlock, 8);
    AYANAMI_DEFINE_UINT(kAyanamiSCDirectLightCardSizePerBlock, 8);

    AYANAMI_DEFINE_UINT(kAyanamiDbgObjGridTileSize, 8);

#ifdef __cplusplus
} // namespace Ifrit::Runtime::AYANAMI
    #undef AYANAMI_DEFINE_UINT
    #undef AYANAMI_DEFINE_FLOAT
#endif

#endif