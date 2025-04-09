
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
#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"

layout(
    local_size_x = kAyanamiRadiosityTraceKernelSize, 
    local_size_y = 1, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec2 m_TraceCoordJitter;
    vec2 m_ProbeCenterJitter;
    uint m_TraceRadianceAtlasUAV;
    uint m_GlobalDFSRV;
    uint m_CardResolution;
    uint m_CardAtlasResolution;
    uint m_CardDepthAtlasSRV;
    uint m_CardNormalAtlasSRV;
    uint m_AllCardObjDataId;
    uint m_AllMeshDFDataId;
    uint m_NumTotalCards;
}PushConst;

RegisterStorage(BAllCardData,{
    CardData m_Mats[];
});

RegisterStorage(BMeshDFDesc,{
    MeshDFDesc m_Data[];
});

RegisterStorage(BMeshDFMeta,{
    MeshDFMeta m_Data;
});

RegisterUniform(BLocalTransform,{
    mat4 m_LocalToWorld;
    mat4 m_WorldToLocal;
    vec4 m_MaxScale;
});

struct CardSampledData{
    vec3 m_WorldPos;
    vec3 m_WorldNormal;
    bool m_ValidSample; // In atlas but hit nothing
    bool m_PresentInAtlas;
};


void RayTraceCoordToCardInfo(uint GThreadId, out uvec2 OffsetInCardTile,out uint CardTileId, out uvec2 TraceRayCoord){
    uint ProbeId = GThreadId % kAyanami_RadiosityTracesPerProbe;
    
    uint ProbesPerTile = kAyanami_RadiosityProbesPerCardTileWidth * kAyanami_RadiosityProbesPerCardTileWidth;
    uint CurTileId = ProbeId / ProbesPerTile;
    uint CurProbeIdInTile = GThreadId % ProbesPerTile;
    uvec2 ProbePosInTile = uvec2(CurProbeIdInTile % kAyanami_RadiosityProbesPerCardTileWidth,
                                 CurProbeIdInTile / kAyanami_RadiosityProbesPerCardTileWidth);

    uint ProbeSpacing = kAyanami_CardTileWidth / kAyanami_RadiosityProbesPerCardTileWidth;
    uvec2 ProbeOffset = ProbePosInTile * ProbeSpacing + uvec2(ProbeSpacing * PushConst.m_TraceCoordJitter);
    OffsetInCardTile = ProbeOffset;
    CardTileId = CurTileId;

    uint RayOffsetInProbe =  GThreadId % kAyanami_RadiosityTracesPerProbe;
    uvec2 RayCoord = uvec2(RayOffsetInProbe % kAyanami_RadiosityProbHemiRes,
                             RayOffsetInProbe / kAyanami_RadiosityProbHemiRes);
    TraceRayCoord = RayCoord;
}

mat4 GetCardMeshLocalToWorld(uint CardId){
    uint MeshDFId = CardId / 6;
    MeshDFDesc MeshDesc = GetResource(BMeshDFDesc, PushConst.m_AllMeshDFDataId).m_Data[MeshDFId];
    MeshDFMeta MeshMeta = GetResource(BMeshDFMeta, PushConst.m_AllMeshDFDataId).m_Data;
    uint TransformId = MeshDesc.m_TransformId;
    return GetResource(BLocalTransform, TransformId).m_LocalToWorld;
}

mat4 GetCardViewVPToWorld(uint CardId){
    CardData CardDesc = GetResource(BAllCardData, PushConst.m_AllCardObjDataId).m_Mats[CardId];
    return CardDesc.m_VPInv;
}

CardSampledData SampleCard(uint CardId, uvec2 InCardUV, uvec2 AtlasUV){
    vec2 InCardUVF = (vec2(InCardUV) + vec2(0.5)) / PushConst.m_CardResolution;
    vec2 AtlasUVF = (vec2(AtlasUV) + vec2(0.5)) / PushConst.m_CardAtlasResolution;
    InCardUVF = InCardUVF * 2.0 - 1.0;
    AtlasUVF = AtlasUVF * 2.0 - 1.0;

    float Depth = SampleTexture2D(PushConst.m_CardDepthAtlasSRV,sLinearClamp,AtlasUVF).r;
    vec2 LocalNormalRG = SampleTexture2D(PushConst.m_CardNormalAtlasSRV,sLinearClamp,AtlasUVF).rg * 2.0 - 1.0;
    vec3 LocalNormal = normalize(vec3(LocalNormalRG, sqrt(1.0 - dot(LocalNormalRG, LocalNormalRG))));

    if(Depth == 1.0){
        CardSampledData SampledData;
        SampledData.m_WorldPos = vec3(0.0);
        SampledData.m_WorldNormal = vec3(0.0);
        SampledData.m_ValidSample = false;
        return SampledData;
    }

    vec4 OrthoNDC = vec4(InCardUVF, Depth, 1.0);
    mat4 CardViewToLocal = GetCardViewVPToWorld(CardId);
    vec4 LocalPos = CardViewToLocal * OrthoNDC;
    LocalPos /= LocalPos.w;

    mat4 CardMeshToWorld = GetCardMeshLocalToWorld(CardId);
    vec4 WorldNormal = CardMeshToWorld * vec4(LocalNormal, 0.0);
    vec4 WorldPos = CardMeshToWorld * vec4(LocalPos.xyz, 1.0);
    WorldPos /= WorldPos.w;
    WorldNormal = normalize(WorldNormal);

    CardSampledData SampledData;
    SampledData.m_WorldPos = WorldPos.xyz;
    SampledData.m_WorldNormal = WorldNormal.xyz;
    SampledData.m_ValidSample = true;
}

CardSampledData GetCardSampledData(uint CardTileId, uvec2 OffsetInCardTile){
    uint TilesPerAtlasWidth = PushConst.m_CardAtlasResolution / kAyanami_CardTileWidth;
    uint TileX = CardTileId % TilesPerAtlasWidth;
    uint TileY = CardTileId / TilesPerAtlasWidth;

    uint CardX = TileX * kAyanami_CardTileWidth + OffsetInCardTile.x;
    uint CardY = TileY * kAyanami_CardTileWidth + OffsetInCardTile.y;
    uint CardIdX = CardX % PushConst.m_CardResolution;
    uint CardIdY = CardY % PushConst.m_CardResolution;
    uint CardId = CardIdX + CardIdY * PushConst.m_CardResolution;

    uvec2 InCardUV = uvec2(CardX, CardY) % kAyanami_CardTileWidth;
    uvec2 AtlasUV = uvec2(CardX, CardY);

    if(CardId>= PushConst.m_NumTotalCards){
        CardSampledData SampledData;
        SampledData.m_WorldPos = vec3(0.0);
        SampledData.m_WorldNormal = vec3(0.0);
        SampledData.m_ValidSample = false;
        SampledData.m_PresentInAtlas = false;
        return SampledData;
    }

    CardSampledData SampledData = SampleCard(CardId, InCardUV, AtlasUV);
    SampledData.m_PresentInAtlas = true;
    return SampledData;
}

// Atlas = 8192x8192
// Tile = 8x8
// 2x2 Probes per tile => 1 probe = 4x4 area (16traces) => 1 probe = 16 storage slots for radiance
uvec2 GetRadianceSlot(uint TileIndex, uvec2 OffsetInTile, uvec2 TraceRayCoord){
    uint TilesPerAtlasWidth = PushConst.m_CardAtlasResolution / kAyanami_CardTileWidth;
    uint TileX = TileIndex % TilesPerAtlasWidth;
    uint TileY = TileIndex / TilesPerAtlasWidth;
    uvec2 OffsetByTile = uvec2(TileX, TileY) * kAyanami_CardTileWidth;

    uvec2 InTileProbeId = OffsetInTile / kAyanami_RadiosityProbHemiRes;
    uvec2 OffsetByProbe = InTileProbeId * kAyanami_RadiosityProbHemiRes;
    return OffsetByTile + OffsetInTile + TraceRayCoord;
}

struct GlobalDFTraceResult{
    vec3 m_HitPos;
    float m_Light;
};

GlobalDFTraceResult TraceGlobalDF(vec3 RayOrigin, vec3 RayDir){
    GlobalDFTraceResult Result;
    return Result;
}

void main(){
    uint tID = gl_LocalInvocationID.x;
    uvec2 gID = gl_WorkGroupID.xy;

    uint TileIndex;
    uvec2 OffsetInTile;
    uvec2 TraceRayCoord;
    RayTraceCoordToCardInfo(tID, OffsetInTile, TileIndex, TraceRayCoord);

    uvec2 WriteSlot = GetRadianceSlot(TileIndex, OffsetInTile, TraceRayCoord);

    CardSampledData SampledData = GetCardSampledData(TileIndex, OffsetInTile);
    if(!SampledData.m_PresentInAtlas){
        return;
    }

    vec3 RadianceVal = vec3(0.0);
    float HitDistance = 1e30;

    if(SampledData.m_ValidSample){
        // Prepare for global df tracing
        vec2 ProbeUV = (vec2(TraceRayCoord) + PushConst.m_ProbeCenterJitter) / float(kAyanami_RadiosityProbHemiRes);
        vec4 RayPDF = ifrit_SampleCosineHemisphereWithPDF(ProbeUV);
        vec3 LocalRayDir = RayPDF.xyz;
        float PDF = RayPDF.w;
        mat3 TBN = ifrit_FrisvadONB(SampledData.m_WorldNormal);
        vec3 WorldRayDir = TBN * LocalRayDir;

        // Here, trace!
        vec3 RayOrigin = SampledData.m_WorldPos + WorldRayDir * 0.01;
        vec3 RayDir = WorldRayDir;
        GlobalDFTraceResult TraceResult = TraceGlobalDF(RayOrigin, RayDir);

        // Write to atlas
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_TraceRadianceAtlasUAV), ivec2(WriteSlot), vec4(RadianceVal, 1.0));
    }
}