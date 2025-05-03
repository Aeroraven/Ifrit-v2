
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

layout(push_constant)  uniform PushConstData{
    uint m_PerFrameCBV;
    uint m_TotalLights;
    uint m_LightDataId;
    uint m_GBufferDepthSRV;
    uint m_GNormalSRV;
} PushConst;

layout(location = 0) in vec2 vTexCoord;

layout(location = 0) out float oShadowMask;

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

uint GetCSMSplitId(vec3 ViewPos,uint LightId){
    ShadowMaps ShadowMap = GetResource(BShadowMaps,PushConst.m_LightDataId).m_Data[LightId];
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

vec2 ShadowMapSingle(uint LightId, vec3 WorldPos,uint CsmIdx){
    ShadowMaps ShadowMap = GetResource(BShadowMaps,PushConst.m_LightDataId).m_Data[LightId];
    
    uint ViewRef = ShadowMap.m_ViewRef[CsmIdx];
    PerFramePerViewData LightView = AyaShared_GetPerFrameData(ViewRef);

    uint ShadowRef = ShadowMap.m_ShadowRef[CsmIdx];
    mat4 LightVP = LightView.m_worldToClip;

    vec4 LightPos = LightVP * vec4(WorldPos,1.0);
    vec3 LightPosNDC = LightPos.xyz / LightPos.w;
    vec2 LightPosNDCxy = LightPosNDC.xy * 0.5 + 0.5;

    // This sampler is maintained by syaro. Don't move this.
    float ShadowMapZ = SampleTexture2D(ShadowRef,sLinearClamp, LightPosNDCxy.xy).r;
    float refZ = LightPosNDC.z;

    if(LightPosNDCxy.x < 0.0 || LightPosNDCxy.x > 1.0 || LightPosNDCxy.y < 0.0 || LightPosNDCxy.y > 1.0){
        return vec2(0.0,0.0);
    }
    float TotalVis = 0.0;
    float TotalSample = 0.0;
    for(int PcfRangeX = -1; PcfRangeX <= 1; PcfRangeX++){
        for(int PcfRangeY = -1; PcfRangeY <= 1; PcfRangeY++){
            vec2 Offset = vec2(float(PcfRangeX),float(PcfRangeY)) * 0.5 / float(2048);
            vec2 SampleUV = LightPosNDCxy.xy + Offset;
            if(SampleUV.x < 0.0 || SampleUV.x > 1.0 || SampleUV.y < 0.0 || SampleUV.y > 1.0){
                continue;
            }
            float ShadowMapZSample = SampleTexture2D(ShadowRef,sLinearClamp, SampleUV).r;
            TotalVis += (refZ < ShadowMapZSample + 1e-5) ? 1.0 : 0.0;
            TotalSample += 1.0;
        }
    }
    TotalVis /= TotalSample;    
    return vec2(TotalVis,1.0);
}

vec2 GlobalShadowVisibility(vec3 WorldPos, vec3 ViewPos){
    vec2 AvgShadow = vec2(0.0,0.0);

    for(int i = 0; i < PushConst.m_TotalLights; i++){
        uint CsmLevel = GetCSMSplitId(ViewPos,i);
        uint reqNextLevel = CsmLevel & 0x80000000;
        CsmLevel = CsmLevel & 0x7FFFFFFF;
        float curCSMEnd = GetResource(BShadowMaps,PushConst.m_LightDataId).m_Data[i].m_CsmEnd[CsmLevel];
        float fade = 0.0;
        if(reqNextLevel != 0){
            float nextCSMStart = GetResource(BShadowMaps,PushConst.m_LightDataId).m_Data[i].m_CsmStart[CsmLevel+1];
            fade = 1.0 - (ViewPos.z - curCSMEnd) / (nextCSMStart - curCSMEnd);
        }
        float DitherRand = ifrit_wnoise2(vec2(WorldPos.xy));
        if(DitherRand < fade){
            CsmLevel++;
        }
        AvgShadow += ShadowMapSingle(i,WorldPos,CsmLevel);
    }
    return AvgShadow;
}

void main(){
    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    
    float ClipDepth = SampleTexture2D(PushConst.m_GBufferDepthSRV, sNearestClamp, vTexCoord).r;
    vec3 NdcPos = vec3(vTexCoord * 2.0 - 1.0, ClipDepth);

    mat4 ClipToWorld = PerFrame.m_clipToWorld;
    mat4 InvPerspective = PerFrame.m_invPerspective;
    mat4 ViewToWorld = PerFrame.m_viewToWorld;

    vec4 WorldPosH = ClipToWorld * vec4(NdcPos, 1.0);
    vec3 WorldPos = WorldPosH.xyz / WorldPosH.w;
    vec4 ViewPosH = InvPerspective * vec4(NdcPos, 1.0);
    vec3 ViewPos = ViewPosH.xyz / ViewPosH.w;

    vec3 ViewNormal = SampleTexture2D(PushConst.m_GNormalSRV, sLinearClamp, vTexCoord).xyz;
    ViewNormal = ViewNormal * 2.0 - 1.0;
    vec3 WorldNormal = (ViewToWorld * vec4(ViewNormal, 0.0)).xyz;
    WorldNormal = normalize(WorldNormal);

    float NormalOffset =  2e-2;
    WorldPos += WorldNormal * NormalOffset; // normal offsetting
    //ViewPos += ViewNormal * NormalOffset; // normal offsetting

    vec2 ShadowVis = GlobalShadowVisibility(WorldPos, ViewPos);
    oShadowMask = ShadowVis.x;
}

