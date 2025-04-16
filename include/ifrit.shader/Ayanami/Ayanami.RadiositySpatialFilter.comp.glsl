
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

vec4 GetNeighbourRadianceSample(
    RadiosityRayCardSample RefTexel,
    ivec2 NeighbourProbeCoord,
    ivec2 CurrentProbeCoord,
    uvec2 OffsetInTile,
    uvec2 TraceRayCoord
){
    int TilesPerCardWidth = PushConst.m_CardAtlasResolution / kAyanami_RadiosityCardTileWidth;
    int ProbesPerFixedSizeCardWidth = kAyanami_RadiosityProbesPerCardTileWidth * kAyanami_RadiosityCardTileWidth;
    ivec2 CurrentCardCoord = ivec2(CurrentProbeCoord / ProbesPerFixedSizeCardWidth);
    ivec2 NeighbourCardCoord = ivec2(NeighbourProbeCoord / ProbesPerFixedSizeCardWidth);
    if(CurrentCardCoord != NeighbourCardCoord){
        return vec4(0.0);
    }

    int TilesPerAtlasWidth = PushConst.m_CardAtlasResolution / kAyanami_RadiosityCardTileWidth;
    ivec2 NeighbourTileCoord = ivec2(NeighbourProbeCoord / kAyanami_RadiosityProbesPerCardTileWidth);
    uint NeighbourTileIndex = uint(NeighbourTileCoord.x + NeighbourTileCoord.y * TilesPerAtlasWidth);

    // TODO: WRONG offsetInTile
    RadiosityRayCardSample NearSample = AyaShared_GetRadiosityRayCardSample(NeighbourTileIndex, OffsetInTile, PushConst.m_CardAtlasResolution,
    PushConst.m_zCardResolution, PushConst.m_NumTotalCards, PushConst.m_CardDepthAtlasSRV,
        PushConst.m_CardNormalAtlasSRV, PushConst.m_AllCardObjDataId, PushConst.m_AllMeshDFDataId);

    if(!NearSample.m_PresentInAtlas || !NearSample.m_ValidSample){
        return vec4(0.0);
    }

    vec3 NearWorldPos = NearSample.m_WorldPos;
    vec3 CurWorldPos = RefTexel.m_WorldPos;
    vec3 CurWorldNormal = RefTexel.m_WorldNormal;
    
    float CurPlaneOffset = dot(CurWorldPos, CurWorldNormal);
    float PosDistance = length(NearWorldPos - CurWorldPos);
    float PosPlaneDistance = abs(dot(NearWorldPos, CurWorldNormal) - CurPlaneOffset);
    float PosPlaneDistanceRelSq = pow(max(PosPlaneDistance/(PosDistance + 0.001), 0.1), 2.0);
    float DepthFactor = exp2(PosPlaneDistanceRelSq * 1.0);
    if(PosDistance > 0.1){
        DepthFactor = 1.0;
    }else{
        DepthFactor = 0.0;
        return vec4(0.0);
    }

    uvec2 NeighbourSlot = AyaShared_GetRadianceSlot(NeighbourTileIndex, OffsetInTile, TraceRayCoord, PushConst.m_CardAtlasResolution);
    vec3 SampledRadiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_TraceRadianceAtlasUAV), ivec2(NeighbourSlot)).rgb;
    return vec4(SampledRadiance * DepthFactor, 1.0);
}

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

    if(!SampledData.m_PresentInAtlas){
        return;
    }

    // Then get the probe pos in the tile
    uint ProbesPerTile = kAyanami_RadiosityProbesPerCardTileWidth * kAyanami_RadiosityProbesPerCardTileWidth;
    uint ProbeIndex = GThreadId % kAyanami_RadiosityTracesPerProbe;
    uint ProbeIndexInTileLinear = ProbeIndex % ProbesPerTile;
    uvec2 ProbePosInTile = uvec2(ProbeIndexInTileLinear % kAyanami_RadiosityProbesPerCardTileWidth,
                                 ProbeIndexInTileLinear / kAyanami_RadiosityProbesPerCardTileWidth);
    uint TilesPerAtlasWidth = PushConst.m_CardAtlasResolution / kAyanami_RadiosityCardTileWidth;
    uint TileIdx_X = TileIndex % TilesPerAtlasWidth;
    uint TileIdx_Y = TileIndex / TilesPerAtlasWidth;
    uvec2 OffsetByTile = uvec2(TileIdx_X, TileIdx_Y) * kAyanami_RadiosityCardTileWidth;

    uvec2 ProbeCoordOffsetByTile = uvec2(OffsetByTile + ProbePosInTile) * kAyanami_RadiosityProbesPerCardTileWidth;
    uvec2 ProbeCoordOverall = ProbeCoordOffsetByTile + ProbePosInTile;

    vec3 SrcRadiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_TraceRadianceAtlasUAV), ivec2(WriteSlot)).rgb;
    float TotalWeights = 1.0;
    if(SampledData.m_ValidSample){
        uint NearSamplesNum= 4;
        ivec2 NearSamples[4];
        NearSamples[0] = ivec2(0, 1);
        NearSamples[1] = ivec2(1, 0);
        NearSamples[2] = ivec2(0, -1);
        NearSamples[3] = ivec2(-1, 0);

        for(uint i = 0; i < NearSamplesNum; i++){
            ivec2 NearProbeCoordOverall = ivec2(ProbeCoordOverall) + NearSamples[i];
            vec4 NearRadiance = GetNeighbourRadianceSample(SampledData, NearProbeCoordOverall, ProbeCoordOverall, OffsetInTile, TraceRayCoord);
            if(NearRadiance.a > 0.0){
                SrcRadiance += NearRadiance.rgb;
                TotalWeights += 1.0;
            }
        }
        SrcRadiance /= TotalWeights;
    }

    imageStore(GetUAVImage2DRGBA32F(PushConst.m_FilteredRadianceAtlasUAV), ivec2(WriteSlot), vec4(FinalRadiance, 1.0));

}