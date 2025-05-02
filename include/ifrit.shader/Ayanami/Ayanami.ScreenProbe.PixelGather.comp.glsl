
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
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"
#include "Math.SphericalHarmonics.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

layout(
    local_size_x = kAyanamiScrProbePixelGatherKernelSize, 
    local_size_y = kAyanamiScrProbePixelGatherKernelSize, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    uint m_RTWidth;
    uint m_RTHeight;
    uint m_PerFrameCBV;
    uint m_ScrNormalCombSRV;
    uint m_ScrDepthCombSRV;
    uint m_OutputSHCoefBufferUAV;
    uint m_OutTexUAV;
}PushConst;

struct ScreenSpaceSample{
    vec3 m_WorldPos;
    vec3 m_WorldNormal;
    float m_SceneDepth;
    bool m_Valid;
};

RegisterStorage(BOutputSHCoefBuffer,{
    float m_Data[];
});


ScreenSpaceSample GetScreenSample(uvec2 ScreenCoord){
    vec2 GBufferUV = (vec2(ScreenCoord) + vec2(0.5)) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);

    vec3 ViewNormal = SampleTexture2D(PushConst.m_ScrNormalCombSRV,sLinearClamp,GBufferUV).xyz;
    ViewNormal = normalize(ViewNormal * 2.0 - 1.0);
    float ViewDepth = SampleTexture2D(PushConst.m_ScrDepthCombSRV,sNearestClamp,GBufferUV).x;
    if(ViewDepth>= 1.0){
        return ScreenSpaceSample(vec3(0.0), vec3(0.0) , 0.0, false);
    }
    vec4 WorldPosAndViewDepth = AyaShared_GetWorldPosFromDepthPersp(PerFrame,ViewDepth,GBufferUV);
    vec3 WorldPos = WorldPosAndViewDepth.xyz;
    float ViewDepthZ = WorldPosAndViewDepth.w;
    mat4 ViewToWorld = PerFrame.m_viewToWorld;
    vec3 WorldNormal = normalize((ViewToWorld * vec4(ViewNormal, 0.0)).xyz);
    return ScreenSpaceSample(WorldPos, WorldNormal, ViewDepthZ, true);
}

ScreenSpaceSample GetProbePos(uvec2 ProbeId){
    // We temporarily drop the jitter
    uvec2 ScreenUV = ProbeId * kAyanami_ScreenProbeUniformPlaceTileWidth;
    ScreenSpaceSample Sample = GetScreenSample(ScreenUV);
    if(!Sample.m_Valid){
        return Sample;
    }
    Sample.m_WorldPos = vec3(1e30, 1e30, 1e30);
}


vec4 GetNeighbourProbeWeights(uvec2 ScreenCoord, ScreenSpaceSample CoordSample){
    // Get the lt probe id first
    uvec2 ProbeId = ScreenCoord / kAyanami_ScreenProbeUniformPlaceTileWidth;

    float BilinearWeightsX = 1.0 - abs(float(ScreenCoord.x % kAyanami_ScreenProbeUniformPlaceTileWidth) / float(kAyanami_ScreenProbeUniformPlaceTileWidth));
    float BilinearWeightsY = 1.0 - abs(float(ScreenCoord.y % kAyanami_ScreenProbeUniformPlaceTileWidth) / float(kAyanami_ScreenProbeUniformPlaceTileWidth));

    float GatherBaseingBias = 0.05;
    BilinearWeightsX =  (BilinearWeightsX + GatherBaseingBias) / (1.0 + 2.0*GatherBaseingBias);
    BilinearWeightsY =  (BilinearWeightsY + GatherBaseingBias) / (1.0 + 2.0*GatherBaseingBias);

    uvec2 NearProbes[4];
    NearProbes[0] = ProbeId + uvec2(0, 0);
    NearProbes[1] = ProbeId + uvec2(1, 0);
    NearProbes[2] = ProbeId + uvec2(0, 1);
    NearProbes[3] = ProbeId + uvec2(1, 1);

    vec3 ProbePosWS[4];
    ProbePosWS[0] = GetProbePos(NearProbes[0]).m_WorldPos;
    ProbePosWS[1] = GetProbePos(NearProbes[1]).m_WorldPos;  
    ProbePosWS[2] = GetProbePos(NearProbes[2]).m_WorldPos;
    ProbePosWS[3] = GetProbePos(NearProbes[3]).m_WorldPos;

    vec4 PlaneCoef = vec4(CoordSample.m_WorldNormal, -dot(CoordSample.m_WorldNormal, CoordSample.m_WorldPos));
    vec4 PlaneDist;
    PlaneDist.x = abs(dot(PlaneCoef, vec4(ProbePosWS[0], 1.0)));
    PlaneDist.y = abs(dot(PlaneCoef, vec4(ProbePosWS[1], 1.0)));
    PlaneDist.z = abs(dot(PlaneCoef, vec4(ProbePosWS[2], 1.0)));
    PlaneDist.w = abs(dot(PlaneCoef, vec4(ProbePosWS[3], 1.0)));

    vec4 RelativeDepth = PlaneDist/CoordSample.m_SceneDepth;
    vec4 ProbeDepthValid;
    ProbeDepthValid.x = (ProbePosWS[0].z > 1e20) ? 0.0 : 1.0;
    ProbeDepthValid.y = (ProbePosWS[1].z > 1e20) ? 0.0 : 1.0;
    ProbeDepthValid.z = (ProbePosWS[2].z > 1e20) ? 0.0 : 1.0;
    ProbeDepthValid.w = (ProbePosWS[3].z > 1e20) ? 0.0 : 1.0;

    vec4 DepthWeights;
    DepthWeights = exp2(-80.0 * (RelativeDepth * RelativeDepth));

    vec4 BilinearWeights;
    BilinearWeights.x = BilinearWeightsX * BilinearWeightsY;
    BilinearWeights.y = (1.0 - BilinearWeightsX) * BilinearWeightsY;
    BilinearWeights.z = BilinearWeightsX * (1.0 - BilinearWeightsY);
    BilinearWeights.w = (1.0 - BilinearWeightsX) * (1.0 - BilinearWeightsY);

    vec4 FinalWeights = vec4(1.0);
    FinalWeights *= DepthWeights * ProbeDepthValid * BilinearWeights;

    return FinalWeights;
}

MThreeBandSH GetProbeSHSingleChannel(uint Offset){
    MThreeBandSH SHCoeffs;
    SHCoeffs.m_Coef1.x = GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset];
    SHCoeffs.m_Coef1.y = GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset+1];
    SHCoeffs.m_Coef1.z = GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset+2];
    SHCoeffs.m_Coef1.w = GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset+3];
    SHCoeffs.m_Coef2.x = GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset+4];
    SHCoeffs.m_Coef2.y = GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset+5];
    SHCoeffs.m_Coef2.z = GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset+6];
    SHCoeffs.m_Coef2.w = GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset+7];
    SHCoeffs.m_Coef3 = GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset+8];
    return SHCoeffs;
}

MThreeBandSH_RGB GetProbeSH(uint ProbeId){
    uint Offset = ProbeId *27;
    MThreeBandSH_RGB SHCoeffs;
    SHCoeffs.m_R = GetProbeSHSingleChannel(Offset);
    SHCoeffs.m_G = GetProbeSHSingleChannel(Offset+9);
    SHCoeffs.m_B = GetProbeSHSingleChannel(Offset+18);
    return SHCoeffs;
}

vec3 GatherProbesUniformOnly(uvec2 ScreenCoord, ScreenSpaceSample CoordSample, vec4 Weights){
    uint ProbeCntPerX = ifrit_DivRoundUp(PushConst.m_RTWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint ProbeCntPerY = ifrit_DivRoundUp(PushConst.m_RTHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);

    uvec2 ProbeId00 = ScreenCoord / kAyanami_ScreenProbeUniformPlaceTileWidth;
    uvec2 ProbeId01 = ProbeId00 + uvec2(1, 0);
    uvec2 ProbeId10 = ProbeId00 + uvec2(0, 1);
    uvec2 ProbeId11 = ProbeId00 + uvec2(1, 1);

    uint ProbeId00I = ProbeId00.x + ProbeId00.y * ProbeCntPerX;
    uint ProbeId01I = ProbeId01.x + ProbeId01.y * ProbeCntPerX;
    uint ProbeId10I = ProbeId10.x + ProbeId10.y * ProbeCntPerX;
    uint ProbeId11I = ProbeId11.x + ProbeId11.y * ProbeCntPerX;

    vec3 WorldNormal = normalize(CoordSample.m_WorldNormal);
    MThreeBandSH CosineLobeNormal = ifrit_SHCosineLobe3Encode(WorldNormal);
    float ValR[4];
    float ValG[4];
    float ValB[4];

    ValR[0] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId00I).m_R);
    ValR[1] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId01I).m_R);
    ValR[2] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId10I).m_R);
    ValR[3] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId11I).m_R);
    ValG[0] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId00I).m_G);
    ValG[1] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId01I).m_G);
    ValG[2] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId10I).m_G);
    ValG[3] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId11I).m_G);
    ValB[0] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId00I).m_B);
    ValB[1] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId01I).m_B);
    ValB[2] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId10I).m_B);
    ValB[3] = ifrit_DotSH3(CosineLobeNormal, GetProbeSH(ProbeId11I).m_B);
    
    vec3 SHVal[4];
    SHVal[0] = vec3(ValR[0], ValG[0], ValB[0]);
    SHVal[1] = vec3(ValR[1], ValG[1], ValB[1]);
    SHVal[2] = vec3(ValR[2], ValG[2], ValB[2]);
    SHVal[3] = vec3(ValR[3], ValG[3], ValB[3]);

    float TotalWeights = Weights.x + Weights.y + Weights.z + Weights.w + 1e-6;

    SHVal[0] *= Weights.x / TotalWeights;
    SHVal[1] *= Weights.y / TotalWeights;
    SHVal[2] *= Weights.z / TotalWeights;
    SHVal[3] *= Weights.w / TotalWeights;

    vec3 FinalSH = vec3(0.0);
    FinalSH += SHVal[0];
    FinalSH += SHVal[1];
    FinalSH += SHVal[2];
    FinalSH += SHVal[3];
    
    return FinalSH;
}

void main(){
    uvec2 ScreenCoord = gl_GlobalInvocationID.xy;
    if(ScreenCoord.x >= PushConst.m_RTWidth || ScreenCoord.y >= PushConst.m_RTHeight) return;

    ScreenSpaceSample CoordSample = GetScreenSample(ScreenCoord);
    vec4 ProbeWeights = GetNeighbourProbeWeights(ScreenCoord, CoordSample);

    if(!CoordSample.m_Valid){
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_OutTexUAV), ivec2(ScreenCoord), vec4(0.0));
        return;
    }

    vec3 GatheredSH = GatherProbesUniformOnly(ScreenCoord, CoordSample, ProbeWeights);
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_OutTexUAV), ivec2(ScreenCoord), vec4(GatheredSH, 1.0));
}