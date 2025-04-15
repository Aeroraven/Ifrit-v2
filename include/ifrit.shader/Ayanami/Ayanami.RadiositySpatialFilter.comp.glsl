
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

#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

layout(
    local_size_x = kAyanamiRadiosityTraceKernelSize, 
    local_size_y = 1, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec2 m_TraceCoordJitter;
    vec2 m_ProbeCenterJitter;
    uint m_TraceRadianceAtlasUAV;
    uint m_CardResolution;
    uint m_CardAtlasResolution;
    uint m_CardDepthAtlasSRV;
    uint m_CardNormalAtlasSRV;
    uint m_CardLightingAtlasSRV;
    uint m_AllCardObjDataId;
    uint m_AllMeshDFDataId;
    uint m_NumTotalCards;
    uint m_FilteredRadianceAtlasUAV;
}PushConst;

void main(){
    uint tID = gl_LocalInvocationID.x;
    uvec2 gID = gl_WorkGroupID.xy;

    uint TileIndex;
    uvec2 OffsetInTile;
    uvec2 TraceRayCoord;
    AyaShared_RayTraceCoordToCardInfo(tID, PushConst.m_TraceCoordJitter, OffsetInTile, TileIndex, TraceRayCoord);

    uvec2 WriteSlot = AyaShared_GetRadianceSlot(TileIndex, OffsetInTile, TraceRayCoord, PushConst.m_CardAtlasResolution);
    RadiosityRayCardSample SampledData = AyaShared_GetRadiosityRayCardSample(TileIndex, OffsetInTile, PushConst.m_CardAtlasResolution,
        PushConst.m_CardResolution, PushConst.m_NumTotalCards, PushConst.m_CardDepthAtlasSRV,
        PushConst.m_CardNormalAtlasSRV, PushConst.m_AllCardObjDataId, PushConst.m_AllMeshDFDataId);

}