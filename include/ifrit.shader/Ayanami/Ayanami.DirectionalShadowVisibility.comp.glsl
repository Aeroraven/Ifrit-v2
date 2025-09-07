
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


#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "SamplerUtils.SharedConst.h"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"
#include "Random/Random.WNoise2D.glsl"

layout(
    local_size_x = kAyanamiShadowVisibilityCardSizePerBlock, 
    local_size_y = kAyanamiShadowVisibilityCardSizePerBlock, 
    local_size_z = kAyanamiShadowVisibilityObjectsPerBlock 
    ) in;
 
layout(push_constant)  uniform PushConstData{
    uint totalCards;
    uint CardResolution;
    uint packedShadowMarkBits;
    uint totalLights;
    uint CardAtlasResolution;

    uint LightDataId;
    uint ShadowMaskOutUAV;
    uint CardDataId;
    uint depthAtlasSRVId;

    uint WorldObjId;
    uint m_PerFrameId;
    uint m_NormalAtlasSRV;
} PushConst;

RegisterStorage(BAllCardData,{
    CardData m_Mats[];
});

RegisterStorage(BAllWorldData,{
    uint m_TransformId[];
});

RegisterStorage(BModelTransform,{
    FLocalTransformData m_Data;
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

RegisterStorage(BPerFrameData,{
    PerFramePerViewData m_Data;
});

uint GetCSMSplitId(vec3 ViewPos,uint LightId){
    ShadowMaps ShadowMap = GetResource(BShadowMaps,PushConst.LightDataId).m_Data[LightId];
    float depth = ViewPos.z;
    uint SplitId = 0;
    uint SplitIdNext = 0;
    for(int i = 0; i < ShadowMap.m_CsmNumSplits; i++){
        if(depth > ShadowMap.m_CsmStart[i] && depth <= ShadowMap.m_CsmEnd[i]){
            SplitId = i;
            // check if the next Split should be used (for fading)
            if(i < ShadowMap.m_CsmNumSplits - 1){
                if(depth > ShadowMap.m_CsmStart[i+1]){
                    SplitId |= 0x80000000;
                }
            }
            break;
        }
    }
    return SplitId;
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

vec2 ShadowMapSingle(uint LightId, vec3 WorldPos,uint csmIdx){
    ShadowMaps ShadowMap = GetResource(BShadowMaps,PushConst.LightDataId).m_Data[LightId];
    uint ViewRef = Uvec4ToUint(ShadowMap.m_ViewRef,csmIdx);
    uint ShadowRef = Uvec4ToUint(ShadowMap.m_ShadowRef,csmIdx);
    mat4 LightView = GetResource(BPerFrameData,ViewRef).m_Data.m_worldToView;
    mat4 LightProj = GetResource(BPerFrameData,ViewRef).m_Data.m_perspective;
    mat4 LightVP = LightProj * LightView;

    vec4 LightPos = LightVP * vec4(WorldPos,1.0);
    vec3 LightPosNDC = LightPos.xyz / LightPos.w;
    vec2 LightPosNDCxy = LightPosNDC.xy * 0.5 + 0.5;

    // This sampler is maintained by syaro. Don't move this.
    float ShadowMapZ = SampleTexture2D(ShadowRef,sLinearClamp, LightPosNDCxy.xy).r;
    float refZ = LightPosNDC.z;
    
    //if ndc out of range, return 0 coverage to g
    if(LightPosNDCxy.x < 0.0 || LightPosNDCxy.x > 1.0 || LightPosNDCxy.y < 0.0 || LightPosNDCxy.y > 1.0){
        return vec2(0.0,0.0);
    }
    float ShadowVis = (refZ < ShadowMapZ + 1e-5) ? 1.0 : 0.0;
    return vec2(ShadowVis,1.0);
}

vec2 GlobalShadowVisibility(vec3 WorldPos, vec3 ViewPos){
    vec2 AvgShadow = vec2(0.0,0.0);

    for(int i = 0; i < PushConst.totalLights; i++){
        uint CsmLevel = GetCSMSplitId(ViewPos,i);
        uint reqNextLevel = CsmLevel & 0x80000000;
        CsmLevel = CsmLevel & 0x7FFFFFFF;
        float curCSMEnd = GetResource(BShadowMaps,PushConst.LightDataId).m_Data[i].m_CsmEnd[CsmLevel];
        float fade = 0.0;
        if(reqNextLevel != 0){
            float nextCSMStart = GetResource(BShadowMaps,PushConst.LightDataId).m_Data[i].m_CsmStart[CsmLevel+1];
            fade = 1.0 - (ViewPos.z - curCSMEnd) / (nextCSMStart - curCSMEnd);
        }
        float ditherRand = ifrit_wnoise2(vec2(WorldPos.xy));
        if(ditherRand < fade){
            CsmLevel++;
        }
        AvgShadow += ShadowMapSingle(i,WorldPos,CsmLevel);
        //AvgShadow = vec2((CsmLevel+1.0)/5.0, 1.0);
    }
    return AvgShadow;
}
    

void main(){
    uvec3 tID = gl_GlobalInvocationID;
    uint maxCardsInLine = PushConst.CardAtlasResolution / PushConst.CardResolution;
    uint CardIndex_X = tID.z % maxCardsInLine;
    uint CardIndex_Y = tID.z / maxCardsInLine;
    uvec2 CardOffset = uvec2(CardIndex_X * PushConst.CardResolution, CardIndex_Y * PushConst.CardResolution);
    uvec2 TileOffset = uvec2(tID.x, tID.y);
    uvec2 OverallOffset = CardOffset + TileOffset;

    uint CardIndex = tID.z;
    uint TileIndex = tID.x + tID.y * gl_WorkGroupSize.x;

    mat4 AtlasToLocal = GetResource(BAllCardData, PushConst.CardDataId).m_Mats[CardIndex].m_VPInv;
    uint transformId = GetResource(BAllWorldData, PushConst.WorldObjId).m_TransformId[CardIndex];
    mat4 LocalToWorld = GetResource(BModelTransform, transformId).m_Data.m_LocalToWorld;
    mat4 AtlasToWorld = LocalToWorld * AtlasToLocal;
    
    vec2 TileOffsetToNDCxy = (vec2(TileOffset)+0.5) / vec2(PushConst.CardResolution);
    TileOffsetToNDCxy = TileOffsetToNDCxy * 2.0 - 1.0;

    vec2 AtlasSampleUV = (OverallOffset+0.5) / vec2(PushConst.CardAtlasResolution);
    float TileOffsetNdcZ = SampleTexture2D(PushConst.depthAtlasSRVId, sNearestClamp,AtlasSampleUV).r; 
#if INTERNAL_AYANAMI_NORMAL_DEBUG
    vec3 TexelNormalVS = SampleTexture2D(PushConst.m_NormalAtlasSRV, sLinearClamp, AtlasSampleUV).xyz * 2.0 - 1.0;
#else
    vec2 TexelNormalRG = SampleTexture2D(PushConst.m_NormalAtlasSRV, sLinearClamp, AtlasSampleUV).rg * 2.0 - 1.0;
    vec3 TexelNormalVS = vec3(TexelNormalRG, sqrt(1.0 - dot(TexelNormalRG, TexelNormalRG)));

#endif
    
    vec4 TileOffsetNdc = vec4(TileOffsetToNDCxy, TileOffsetNdcZ, 1.0);

    vec4 WorldPosH = AtlasToWorld * TileOffsetNdc;
    vec4 WorldPosP = WorldPosH / WorldPosH.w;
    vec3 WorldNormal = normalize(LocalToWorld * vec4(TexelNormalVS, 0.0)).xyz;

    // add a slight normal offset to avoid self shadowing
    float NormalOffset =  1e-3;
    WorldPosP += vec4(WorldNormal * NormalOffset,0.0); // normal offsetting

    // Test if the World position can be seen by the Light.
    // Two components should write to the desired texture:
    // 1. Light visibility
    // 2. Light coverage (from camera View) , because Shadow maps are 'camera-centric'.

    if(TileOffsetNdcZ == 1.0){
        imageStore(GetUAVImage2DR32F(PushConst.ShadowMaskOutUAV), ivec2(OverallOffset), vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    mat4 WorldToView = GetResource(BPerFrameData, PushConst.m_PerFrameId).m_Data.m_worldToView;
    vec4 ViewPos = WorldToView * WorldPosP;

    float ShadowVisibility = 0.0;
    float ShadowCoverage = 0.0;
    vec2 ShadowVisibilityAndCoverage = GlobalShadowVisibility(WorldPosP.xyz, ViewPos.xyz);
    ShadowVisibility = ShadowVisibilityAndCoverage.x;
    ShadowCoverage = ShadowVisibilityAndCoverage.y;

    imageStore(GetUAVImage2DR32F(PushConst.ShadowMaskOutUAV), ivec2(OverallOffset), vec4(ShadowVisibility, ShadowCoverage, 0.0, 1.0));
}