
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
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiFinalTemporalReprojKernelSizeX, 16);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiFinalTemporalReprojKernelSizeY, 16);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiSCDirectLightObjectsPerBlock, 8);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiSCDirectLightCardSizePerBlock, 8);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiScrProbeAdaptiveGroupKernelSize, 64);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiReconFromSCTileSize, 8);
    IFSHADER_DEFINE_CONST_UINT32(kAyanamiReconFromSCDepthTileSize, 8);

    IFSHADER_DEFINE_CONST_UINT32(kAyanamiGlobalDFRayMarchTileSize, 16);
}