
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
#include "DeferredPBR.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

#include "Random/Random.WNoise2D.glsl"

layout(push_constant)  uniform PushConstData{
    uint m_PerFrameCBV;
    uint m_ShadowOcclusionSRV;
    uint m_GAlbedoSRV;
    uint m_GNormalSRV;
    uint m_GDepthSRV;
    uint m_TotalLights;
    uint m_LightDataId;
} PushConst;

layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 oDirectLighting;

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

vec3 GetLightFront(uint LightIdx){
    ShadowMaps ShadowMap = GetResource(BShadowMaps,PushConst.m_LightDataId).m_Data[LightIdx];
    uint ViewRef = ShadowMap.m_ViewRef[0];
    PerFramePerViewData LightView = AyaShared_GetPerFrameData(ViewRef);
    vec3 LightFront = LightView.m_cameraFront.xyz;
    return LightFront;
}

vec3 ShadingFromLight(uint LightIdx, vec3 WorldPos, vec3 WorldNormal, vec3 Albedo, vec3 ShadowMask,vec3 Vx){
    vec3 LightDir = -normalize(GetLightFront(LightIdx));
    vec3 V = normalize(Vx);
    vec3 H = normalize(LightDir + V);
    float NdotH = max(0.0, dot(WorldNormal, H));

    float Roughness = 0.62;
    float D = dpbr_trowbridgeReitzGGX(NdotH,Roughness);

    float NdotV = max(0.0, dot(WorldNormal, V));
    float NdotL = max(0.0, dot(WorldNormal, LightDir));
    float G = dpbr_smithSchlickGGX(NdotV,NdotL,Roughness);

    vec3 F0 = vec3(0.04);
    float Metallic = 0.03;
    float HdotV = max(dot(H,V),0.0);
    vec3 F = dpbr_fresnelSchlickMetallic(F0,Albedo,Metallic,HdotV);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - Metallic;

    float PIx = 3.14159265359;
    vec3 Specular = dpbr_cookTorranceBRDF(F,G,D,NdotV,NdotL);

    vec3 Lo = ((kD/PIx)* Albedo+ Specular) * NdotL * 12.3;

    return Lo * ShadowMask ;
}

void main(){
    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    
    float ClipDepth = SampleTexture2D(PushConst.m_GDepthSRV, sNearestClamp, vTexCoord).r;
    vec3 NdcPos = vec3(vTexCoord * 2.0 - 1.0, ClipDepth);

    mat4 ClipToWorld = PerFrame.m_clipToWorld;
    mat4 InvPerspective = PerFrame.m_invPerspective;
    mat4 ViewToWorld = PerFrame.m_viewToWorld;

    vec4 WorldPosH = ClipToWorld * vec4(NdcPos, 1.0);
    vec3 WorldPos = WorldPosH.xyz / WorldPosH.w;
    vec4 ViewPosH = InvPerspective * vec4(NdcPos, 1.0);
    vec3 ViewPos = ViewPosH.xyz / ViewPosH.w;

    vec3 ViewNormal = SampleTexture2D(PushConst.m_GNormalSRV, sNearestClamp, vTexCoord).xyz;
    ViewNormal = ViewNormal * 2.0 - 1.0;
    vec3 Albedo = SampleTexture2D(PushConst.m_GAlbedoSRV, sNearestClamp, vTexCoord).xyz;
    vec3 ShadowMask = vec3(SampleTexture2D(PushConst.m_ShadowOcclusionSRV, sNearestClamp, vTexCoord).x);

    vec4 WorldNormal = ViewToWorld * vec4(ViewNormal, 0.0);
    WorldNormal = normalize(WorldNormal);

    vec4 ViewDirVS = vec4(normalize(ViewPos).xyz, 0.0);
    vec4 ViewDirWS = (ViewToWorld * ViewDirVS);
    ViewDirWS = normalize(ViewDirWS);

    uint TotalLights = PushConst.m_TotalLights;
    vec3 DirectLighting = vec3(0.0);
    for(uint i = 0; i < TotalLights; i++){
        vec3 Shading = ShadingFromLight(i, WorldPos, WorldNormal.xyz, Albedo, ShadowMask, -ViewDirWS.xyz);
        DirectLighting += Shading;
    }
    
    oDirectLighting = vec4(DirectLighting, 1.0);
}

