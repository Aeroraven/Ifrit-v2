
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
#include "ifrit.shader.neo/Common.hlsli"

namespace IfritShader
{
    // SDF expanding. This is used to expand the SDF volume to avoid artifacts in thin geometry.
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_SDFExpand, 1);
    IFSHADER_DEFINE_CONST_FLOAT(kAyanami_SDFExpandRatio, 0.1f);

    // Object grid constants. Represents the alternative to the voxel lighting datastructure.
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_MaxObjectPerGridCell, 4);
    IFSHADER_DEFINE_CONST_FLOAT(kAyanami_ObjectGridCellQueryInterpolationRange, 3.0f);
    // might affect shared memory x_x! & 1 for atomic
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_ObjectGridCellMaxCullObjPerPass, 511);

    // Radiosity processing
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_CardTileWidth, 8);
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_RadiosityProbesPerCardTileWidth, 2);
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_RadiosityTracesPerCardTile, 64);
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_RadiosityTracesPerProbe, 16);
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_RadiosityTracesPerProbeSqrt, 4);
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_RadiosityProbHemiRes, 4);

    // Screen Probe Placement
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_ScreenProbeUniformPlaceTileWidth, 16);
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_ScreenProbeProbeHemiRes, 8);   // 64 rays per probe
    IFSHADER_DEFINE_CONST_UINT32(kAyanami_ScreenProbeTracePerProbe, 64); // 64 rays per probe

    // Kernel Sizes
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiGlobalDFCompositeTileSize, 8);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiGlobalDFRayMarchTileSize, 16);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiShadowVisibilityObjectsPerBlock, 8);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiShadowVisibilityCardSizePerBlock, 8);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiObjectGridTileSize, 4);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiRadiosityTraceKernelSize, kAyanami_RadiosityTracesPerCardTile);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiSphericalHarmonicsCvtKernelSize, 64);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiRadiosityIntegrateKernelSizeY, kAyanami_CardTileWidth);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiReconFromSCTileSize, 8);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiReconFromSCDepthTileSize, 8);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiSCDirectLightObjectsPerBlock, 8);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiSCDirectLightCardSizePerBlock, 8);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiDbgObjGridTileSize, 8);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiScrProbeAdaptivePlaceKernelSize, 8);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiScrProbeAdaptiveGroupKernelSize, 64);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiDbgScrProbeUniformVisKernelSize, 8);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiScrProbeScreenTraceKernelSize, kAyanami_ScreenProbeProbeHemiRes); // 1 probe per TG
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiScrProbeMDFTraceKernelSize, 64);                                  // 64 traces per TG
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiScrProbeGDFTraceKernelSize, 64);                                  // 64 traces per TG

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiScrProbeMDFCullPrepKernelSize, 64);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiScrProbeIntegrateSHKernelSize, 64);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiScrProbePixelGatherKernelSize, 8);

    // Temporal
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiFinalTemporalReprojKernelSizeX, 16);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiFinalTemporalReprojKernelSizeY, 16);


}