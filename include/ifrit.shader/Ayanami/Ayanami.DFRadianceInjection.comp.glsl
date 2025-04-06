
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
#include "Random/Random.WNoise2D.glsl"

layout(
    local_size_x = kAyanamiRadianceInjectionCardSizePerBlock, 
    local_size_y = kAyanamiRadianceInjectionCardSizePerBlock, 
    local_size_z = kAyanamiRadianceInjectionObjectsPerBlock 
) in;

RegisterStorage(BAllCardData,{
    CardData m_Mats[];
});

RegisterStorage(BAllWorldData,{
    uint m_TransformId[];
});

RegisterUniform(BLocalTransform,{
    mat4 m_LocalToWorld;
    mat4 m_WorldToLocal;
    float m_MaxScale;
});

RegisterStorage(BTileScatter,{
    uint m_Data[];
});

RegisterStorage(BMeshDFDesc,{
    MeshDFDesc m_Data[];
});

RegisterStorage(BMeshDFMeta,{
    MeshDFMeta m_Data;
});

RegisterStorage(BTileAtomics,{
    uint m_Data[];
});

layout(push_constant)  uniform PushConstData{
    mat4 m_ShadowLightVP;
    vec4 m_ShadowLightDir;

    uint m_TotalCards;
    uint m_CardResolution;
    uint m_CardAtlasResolution;

    uint m_RadianceUAV;
    uint m_CardDataId;
    uint m_DepthAtlasSRVId;

    uint m_WorldObjId;

    uint m_ShadowCullTileDFAtomics;
    uint m_ShadowCullTileDFList;
    uint m_ShadowCullTileSize;
    uint m_ShadowCullTotalDFs;
    uint m_MeshDFDescListId;
    float m_ShadowCoefK;
} PushConst;

float DistanceFieldShadowInObj(vec3 rayOriginWS, uint meshDFId){

    MeshDFDesc desc = GetResource(BMeshDFDesc, PushConst.m_MeshDFDescListId).m_Data[meshDFId];
    MeshDFMeta meta = GetResource(BMeshDFMeta, desc.m_MdfMetaId).m_Data;
    mat4 worldToLocal = GetResource(BLocalTransform, desc.m_TransformId).m_WorldToLocal;

    uint sdfId = meta.sdfId;
    vec3 lb = meta.bboxMin.xyz;
    vec3 rt = meta.bboxMax.xyz;
    vec3 extent = rt - lb;
    float maxExtent = min(min(extent.x, extent.y), extent.z);
    vec3 rayDir = normalize(-PushConst.m_ShadowLightDir.xyz);

    float advance = 0.01;
    vec4 pO = vec4(rayOriginWS + rayDir * advance, 1.0);
    pO = worldToLocal * pO;
    pO = pO / pO.w;

    vec4 pD = vec4(rayOriginWS + rayDir * (1.0+advance), 1.0);
    pD = worldToLocal * pD;
    pD = pD / pD.w;

    vec3 d = pD.xyz - pO.xyz;
    vec3 o = pO.xyz;
    vec3 nD = normalize(d);

    float t;
    bool hit = ifrit_RayboxIntersection(o,nD,lb,rt,t);
    t = max(0.0,t);

    vec3 hitp = o + nD * t;
    float retShadow = 1.0;
    float selfBias = 0e-4*maxExtent;
    float volBias = 1e-3*maxExtent;
    if(hit){
        for(int i=0;i<20;i++){
            vec3 uvw= (hitp - lb) / (rt - lb);
            uvw = clamp(uvw, 0.0, 1.0);
            float sdf = texture(GetSampler3D(meta.sdfId), uvw).x-volBias;
            t+= max(1e-4*maxExtent,abs(sdf)* 0.5) ;
            hitp = o + nD * t;
            
            retShadow = min(retShadow, PushConst.m_ShadowCoefK*abs(sdf)/(abs(t)+1e-6)*100.0);
            if(abs(sdf)<1e-1){
                break;
            }
        }
    }
    return retShadow;
}

float DistanceFieldShadowInTile(vec3 rayOriginWS, uint tileId){
    float shadowAttn = 1.0;
    uint tileOffset = PushConst.m_ShadowCullTotalDFs * tileId;
    uint dfInTile = GetResource(BTileAtomics, PushConst.m_ShadowCullTileDFAtomics).m_Data[tileId];
    for(uint i=0;i<dfInTile;i++){
        uint dfId = GetResource(BTileScatter, PushConst.m_ShadowCullTileDFList).m_Data[tileOffset + i];
        shadowAttn = min(shadowAttn,DistanceFieldShadowInObj(rayOriginWS,dfId));
    }
    return shadowAttn;
}

float DistanceFieldShadow(vec3 rayOriginWS){
    vec4 lightPos = PushConst.m_ShadowLightVP * vec4(rayOriginWS, 1.0);
    vec2 uv = (lightPos.xy / lightPos.w) * 0.5 + 0.5;
    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0){
        return 1.0;
    }
    uint tileX = uint(uv.x * PushConst.m_ShadowCullTileSize);
    uint tileY = uint(uv.y * PushConst.m_ShadowCullTileSize);
    uint tileId = tileX + tileY * PushConst.m_ShadowCullTileSize;
    return DistanceFieldShadowInTile(rayOriginWS, tileId);
}


// For those card pixels not covered by camera space shadow map, DF shadow tracing is used.
// It's better to use a CS filter pass to filter out those pixels for less warp divergence.
// (Normally in tiles for better rw pattern?)
// However, the performance optimization will be considered later. I am too lazy =v=.
void main(){
    uvec3 tID = gl_GlobalInvocationID;
    uint maxCardsInLine = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
    uint cardIndex_X = tID.z % maxCardsInLine;
    uint cardIndex_Y = tID.z / maxCardsInLine;
    uvec2 cardOffset = uvec2(cardIndex_X * PushConst.m_CardResolution, cardIndex_Y * PushConst.m_CardResolution);
    uvec2 tileOffset = uvec2(tID.x, tID.y);
    uvec2 overallOffset = cardOffset + tileOffset;

    uint cardIndex = tID.z;
    uint tileIndex = tID.x + tID.y * gl_WorkGroupSize.x;

    mat4 atlasToLocal = GetResource(BAllCardData, PushConst.m_CardDataId).m_Mats[cardIndex].m_VPInv;
    uint transformId = GetResource(BAllWorldData, PushConst.m_WorldObjId).m_TransformId[cardIndex];
    mat4 localToWorld = GetResource(BLocalTransform, transformId).m_LocalToWorld;
    mat4 atlasToWorld = localToWorld * atlasToLocal;

    vec2 tileOffsetToNDCxy = (vec2(tileOffset)+0.5) / vec2(PushConst.m_CardResolution);
    tileOffsetToNDCxy = tileOffsetToNDCxy * 2.0 - 1.0;

    vec2 atlasSampleUV = (overallOffset+0.5) / vec2(PushConst.m_CardAtlasResolution);
    float tileOffsetNdcZ = texture(GetSampler2D(PushConst.m_DepthAtlasSRVId), atlasSampleUV).r;
    vec4 tileOffsetNdc = vec4(tileOffsetToNDCxy, tileOffsetNdcZ, 1.0);

    vec4 worldPos = atlasToWorld * tileOffsetNdc;
    vec4 worldPosNDC = worldPos / worldPos.w;

    if(tileOffsetNdcZ == 1.0){
        return;
    }

    vec2 shadowVisCov = imageLoad(GetUAVImage2DR32F(PushConst.m_RadianceUAV), ivec2(overallOffset)).rg;
    if(shadowVisCov.y <= 0.1){
        shadowVisCov.x = DistanceFieldShadow(worldPos.xyz);
        imageStore(GetUAVImage2DR32F(PushConst.m_RadianceUAV), ivec2(overallOffset), vec4(shadowVisCov.x, 1.0, 0.0, 1.0));
    }
}