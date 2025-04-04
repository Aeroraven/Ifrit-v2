
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

layout(push_constant)  uniform PushConstData{
    uint totalCards;
    uint cardResolution;
    uint packedShadowMarkBits;
    uint totalLights;
    uint cardAtlasResolution;

    uint lightDataId;
    uint radianceOutId;
    uint cardDataId;
    uint depthAtlasSRVId;

    uint worldObjId;
    uint m_PerFrameId;
} PushConst;

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


struct ShadowMaps{
    uvec4 m_ViewRef;
    uvec4 m_ShadowRef;
    uvec4 m_ViewMapping; //Useless for shader, should be optimized
    vec4 m_CsmStart;
    vec4 m_CsmEnd;
    uint m_CsmNumSplits;
};

RegisterStorage(BShadowMaps,{
    ShadowMaps m_Data[];
});

RegisterUniform(BPerFrameData,{
    PerFramePerViewData m_Data;
});

uint GetCSMSplitId(vec3 viewPos,uint lightId){
    ShadowMaps shadowMap = GetResource(BShadowMaps,PushConst.lightDataId).m_Data[lightId];
    float depth = viewPos.z;
    uint splitId = 0;
    uint splitIdNext = 0;
    for(int i = 0; i < shadowMap.m_CsmNumSplits; i++){
        if(depth > shadowMap.m_CsmStart[i] && depth <= shadowMap.m_CsmEnd[i]){
            splitId = i;
            // check if the next split should be used (for fading)
            if(i < shadowMap.m_CsmNumSplits - 1){
                if(depth > shadowMap.m_CsmStart[i+1]){
                    splitId |= 0x80000000;
                }
            }
            break;
        }
    }
    return splitId;
}


uint Uvec4ToUint(uvec4 v, uint idx){
    if(idx == 0){
        return v.x;
    }else if(idx == 1){
        return v.y;
    }else if(idx == 2){
        return v.z;
    }else{
        return v.w;
    }
}

vec2 ShadowMapSingle(uint lightId, vec3 worldPos,uint csmIdx){
    ShadowMaps shadowMap = GetResource(BShadowMaps,PushConst.lightDataId).m_Data[lightId];
    uint viewRef = Uvec4ToUint(shadowMap.m_ViewRef,csmIdx);
    uint shadowRef = Uvec4ToUint(shadowMap.m_ShadowRef,csmIdx);
    mat4 lightView = GetResource(BPerFrameData,viewRef).m_Data.m_worldToView;
    mat4 lightProj = GetResource(BPerFrameData,viewRef).m_Data.m_perspective;
    mat4 lightVP = lightProj * lightView;

    vec4 lightPos = lightVP * vec4(worldPos,1.0);
    vec3 lightPosNDC = lightPos.xyz / lightPos.w;
    vec2 lightPosNDCxy = lightPosNDC.xy * 0.5 + 0.5;

    float shadowMapZ = texture(GetSampler2D(shadowRef), lightPosNDCxy.xy).r;
    float refZ = lightPosNDC.z;
    
    //if ndc out of range, return 0 coverage to g
    if(lightPosNDCxy.x < 0.0 || lightPosNDCxy.x > 1.0 || lightPosNDCxy.y < 0.0 || lightPosNDCxy.y > 1.0){
        return vec2(0.0,0.0);
    }
    float shadowVis = (refZ < shadowMapZ - 1e-3) ? 1.0 : 0.0;
    return vec2(shadowVis,1.0);
}

vec2 GlobalShadowVisibility(vec3 worldPos, vec3 viewPos){
    vec2 avgShadow = vec2(0.0,0.0);

    for(int i = 0; i < PushConst.totalLights; i++){
        uint csmLevel = GetCSMSplitId(viewPos,i);
        uint reqNextLevel = csmLevel & 0x80000000;
        csmLevel = csmLevel & 0x7FFFFFFF;
        float curCSMEnd = GetResource(BShadowMaps,PushConst.lightDataId).m_Data[i].m_CsmEnd[csmLevel];
        float fade = 0.0;
        if(reqNextLevel != 0){
            float nextCSMStart = GetResource(BShadowMaps,PushConst.lightDataId).m_Data[i].m_CsmStart[csmLevel+1];
            fade = 1.0 - (viewPos.z - curCSMEnd) / (nextCSMStart - curCSMEnd);
        }
        float ditherRand = ifrit_wnoise2(vec2(worldPos.xy));
        if(ditherRand < fade){
            csmLevel++;
        }
        uvec4 viewRef4 = GetResource(BShadowMaps,PushConst.lightDataId).m_Data[i].m_ViewRef;
        uint viewRef = Uvec4ToUint(viewRef4,csmLevel);
        float lightOrthoSize = GetResource(BPerFrameData,viewRef).m_Data.m_cameraOrthoSize;
        
        avgShadow += ShadowMapSingle(i,worldPos,csmLevel);
    }
    return avgShadow;
}
    

void main(){
    uvec3 tID = gl_GlobalInvocationID;
    uint maxCardsInLine = PushConst.cardAtlasResolution / PushConst.cardResolution;
    uint cardIndex_X = tID.z % maxCardsInLine;
    uint cardIndex_Y = tID.z / maxCardsInLine;
    uvec2 cardOffset = uvec2(cardIndex_X * PushConst.cardResolution, cardIndex_Y * PushConst.cardResolution);
    uvec2 tileOffset = uvec2(tID.x, tID.y);
    uvec2 overallOffset = cardOffset + tileOffset;

    uint cardIndex = tID.z;
    uint tileIndex = tID.x + tID.y * gl_WorkGroupSize.x;


    mat4 atlasToLocal = GetResource(BAllCardData, PushConst.cardDataId).m_Mats[cardIndex].m_VPInv;
    uint transformId = GetResource(BAllWorldData, PushConst.worldObjId).m_TransformId[cardIndex];
    mat4 localToWorld = GetResource(BLocalTransform, transformId).m_LocalToWorld;
    mat4 atlasToWorld = localToWorld * atlasToLocal;

    vec2 tileOffsetToNDCxy = (vec2(tileOffset)+0.5) / vec2(PushConst.cardAtlasResolution);
    tileOffsetToNDCxy = tileOffsetToNDCxy * 2.0 - 1.0;

    vec2 atlasSampleUV = (overallOffset+0.5) / vec2(PushConst.cardAtlasResolution);
    float tileOffsetNdcZ = texture(GetSampler2D(PushConst.depthAtlasSRVId), atlasSampleUV).r;
    vec4 tileOffsetNdc = vec4(tileOffsetToNDCxy, tileOffsetNdcZ, 1.0);

    vec4 worldPos = atlasToWorld * tileOffsetNdc;
    vec4 worldPosNDC = worldPos / worldPos.w;

    // Test if the world position can be seen by the light.
    // Two components should write to the desired texture:
    // 1. light visibility
    // 2. light coverage (from camera view) , because shadow maps are 'camera-centric'.

    mat4 worldToView = GetResource(BPerFrameData, PushConst.m_PerFrameId).m_Data.m_worldToView;
    vec4 viewPos = worldToView * worldPos;

    float shadowVisibility = 0.0;
    float shadowCoverage = 0.0;
    vec2 shadowVisibilityAndCoverage = GlobalShadowVisibility(worldPos.xyz, viewPos.xyz);
    shadowVisibility = shadowVisibilityAndCoverage.x;
    shadowCoverage = shadowVisibilityAndCoverage.y;

    imageStore(GetUAVImage2DR32F(PushConst.radianceOutId), ivec2(overallOffset), vec4(shadowVisibility, shadowCoverage, 0.0, 1.0));
}