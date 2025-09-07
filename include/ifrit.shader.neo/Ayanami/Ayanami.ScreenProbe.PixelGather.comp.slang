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


#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedConst.hlsli"
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedUtils.hlsli"
#include "ifrit.shader.neo/Math.SphericalHarmonics.hlsli"

namespace IfritShader{
namespace Ayanami{

    struct PushConstant_T
    {
        uint m_RTWidth;
        uint m_RTHeight;
        TConstantBufferHandle<PerFramePerViewData> m_PerFrame;
        TTexture2DHandle<float4> m_GBufferNormal;
        TTexture2DHandle<float> m_GBufferDepth;
        TRWStructuredBufferHandle<float> m_OutputSHCoefBuffer;
        TRWTexture2DHandle<float4> m_OutTex;
    };

    struct ScreenSpaceSample{
        float3 m_WorldPos;
        float3 m_WorldNormal;
        float m_SceneDepth;
        bool m_Valid;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    ScreenSpaceSample GetScreenSample(uint2 ScreenCoord)
    {
        float2 GBufferUV = (float2(ScreenCoord) + float2(0.5)) / float2(PushConst.m_RTWidth, PushConst.m_RTHeight);
        PerFramePerViewData PerFrame = PushConst.m_PerFrame.Load();

        float3 ViewNormal = PushConst.m_GBufferNormal.SampleLevel(ESamplerType::SLinearClamp, GBufferUV, 0).xyz;
        ViewNormal = normalize(ViewNormal * 2.0f - 1.0f);
        float ViewDepth = PushConst.m_GBufferDepth.SampleLevel(ESamplerType::SLinearClamp, GBufferUV, 0);
        if(ViewDepth >= 1.0f)
        {
            ScreenSpaceSample InvalidSample;
            InvalidSample.m_WorldPos = float3(0.0f, 0.0f, 0.0f);
            InvalidSample.m_WorldNormal = float3(0.0f, 0.0f, 0.0f);
            InvalidSample.m_SceneDepth = 0.0f;
            InvalidSample.m_Valid = false;
            return InvalidSample;
        }
        float4 WorldPosAndViewDepth = GetWorldPosFromDepthPersp(PerFrame,ViewDepth,GBufferUV);
        float3 WorldPos = WorldPosAndViewDepth.xyz;
        float ViewDepthZ = WorldPosAndViewDepth.w;
        float4x4 ViewToWorld = PerFrame.m_ViewToWorld;
        
        float3 WorldNormal = normalize((mul(ViewToWorld , float4(ViewNormal, 0.0))).xyz);
        return ScreenSpaceSample(WorldPos, WorldNormal, ViewDepthZ, true);
    }

    ScreenSpaceSample GetProbePos(uint2 ProbeId){
        // TODO: add jitter later !
        uint2 ScreenUV = ProbeId * kAyanami_ScreenProbeUniformPlaceTileWidth;
        ScreenSpaceSample Sample = GetScreenSample(ScreenUV);
        if(Sample.m_Valid){
            return Sample;
        }
        Sample.m_WorldPos = float3(1e30, 1e30, 1e30);
        return Sample;
    }
    

    float4 GetNeighbourProbeWeights(uint2 ScreenCoord, ScreenSpaceSample CoordSample)
    {
        uint2 ProbeId = ScreenCoord / kAyanami_ScreenProbeUniformPlaceTileWidth;

        float BilinearWeightsX = 1.0f - abs(float(ScreenCoord.x % kAyanami_ScreenProbeUniformPlaceTileWidth) / float(kAyanami_ScreenProbeUniformPlaceTileWidth));
        float BilinearWeightsY = 1.0f - abs(float(ScreenCoord.y % kAyanami_ScreenProbeUniformPlaceTileWidth) / float(kAyanami_ScreenProbeUniformPlaceTileWidth));
        
        float GatherBaseingBias = 0.05;
        BilinearWeightsX =  (BilinearWeightsX + GatherBaseingBias) / (1.0 + 2.0*GatherBaseingBias);
        BilinearWeightsY =  (BilinearWeightsY + GatherBaseingBias) / (1.0 + 2.0*GatherBaseingBias);

        uint2 NearProbes[4];
        NearProbes[0] = ProbeId + uint2(0, 0);
        NearProbes[1] = ProbeId + uint2(1, 0);
        NearProbes[2] = ProbeId + uint2(0, 1);
        NearProbes[3] = ProbeId + uint2(1, 1);

        float3 ProbePosWS[4];
        ProbePosWS[0] = GetProbePos(NearProbes[0]).m_WorldPos;
        ProbePosWS[1] = GetProbePos(NearProbes[1]).m_WorldPos;  
        ProbePosWS[2] = GetProbePos(NearProbes[2]).m_WorldPos;
        ProbePosWS[3] = GetProbePos(NearProbes[3]).m_WorldPos;

        float4 PlaneCoef = float4(CoordSample.m_WorldNormal, -dot(CoordSample.m_WorldNormal, CoordSample.m_WorldPos));
        float4 PlaneDist;
        PlaneDist.x = abs(dot(PlaneCoef, float4(ProbePosWS[0], 1.0)));
        PlaneDist.y = abs(dot(PlaneCoef, float4(ProbePosWS[1], 1.0)));
        PlaneDist.z = abs(dot(PlaneCoef, float4(ProbePosWS[2], 1.0)));
        PlaneDist.w = abs(dot(PlaneCoef, float4(ProbePosWS[3], 1.0)));

        float4 RelativeDepth = PlaneDist/CoordSample.m_SceneDepth;
        float4 ProbeDepthValid;
        ProbeDepthValid.x = (ProbePosWS[0].z > 1e20) ? 0.0 : 1.0;
        ProbeDepthValid.y = (ProbePosWS[1].z > 1e20) ? 0.0 : 1.0;
        ProbeDepthValid.z = (ProbePosWS[2].z > 1e20) ? 0.0 : 1.0;
        ProbeDepthValid.w = (ProbePosWS[3].z > 1e20) ? 0.0 : 1.0;

        float4 DepthWeights;
        DepthWeights = exp2(-80.0f * (RelativeDepth * RelativeDepth));
        
        float4 BilinearWeights;
        BilinearWeights.x = BilinearWeightsX * BilinearWeightsY;
        BilinearWeights.y = (1.0 - BilinearWeightsX) * BilinearWeightsY;
        BilinearWeights.z = BilinearWeightsX * (1.0 - BilinearWeightsY);
        BilinearWeights.w = (1.0 - BilinearWeightsX) * (1.0 - BilinearWeightsY);

        float4 FinalWeights = float4(1.0);
        FinalWeights *= DepthWeights * ProbeDepthValid * BilinearWeights;

        return FinalWeights;
    }

    Math::FThreeBandSH GetProbeSHSingleChannel(uint Offset){
        Math::FThreeBandSH SHCoeffs;
        SHCoeffs.m_Coef1.x = PushConst.m_OutputSHCoefBuffer.Load(Offset + 0);
        SHCoeffs.m_Coef1.y = PushConst.m_OutputSHCoefBuffer.Load(Offset + 1);
        SHCoeffs.m_Coef1.z = PushConst.m_OutputSHCoefBuffer.Load(Offset + 2);
        SHCoeffs.m_Coef1.w = PushConst.m_OutputSHCoefBuffer.Load(Offset + 3);
        SHCoeffs.m_Coef2.x = PushConst.m_OutputSHCoefBuffer.Load(Offset + 4);
        SHCoeffs.m_Coef2.y = PushConst.m_OutputSHCoefBuffer.Load(Offset + 5);
        SHCoeffs.m_Coef2.z = PushConst.m_OutputSHCoefBuffer.Load(Offset + 6);
        SHCoeffs.m_Coef2.w = PushConst.m_OutputSHCoefBuffer.Load(Offset + 7);
        SHCoeffs.m_Coef3 = PushConst.m_OutputSHCoefBuffer.Load(Offset + 8);
        return SHCoeffs;
    }

    Math::FThreeBandSHRgb GetProbeSH(uint ProbeId){
        uint Offset = ProbeId *27;
        Math::FThreeBandSHRgb SHCoeffs;
        SHCoeffs.m_Red = GetProbeSHSingleChannel(Offset);
        SHCoeffs.m_Green = GetProbeSHSingleChannel(Offset+9);
        SHCoeffs.m_Blue = GetProbeSHSingleChannel(Offset+18);
        return SHCoeffs;
    }

    float3 GatherProbesUniformOnly(uint2 ScreenCoord, ScreenSpaceSample CoordSample, float4 Weights){
        uint ProbeCntPerX = DivRoundUp(PushConst.m_RTWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
        uint ProbeCntPerY = DivRoundUp(PushConst.m_RTHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
    
        uint2 ProbeId00 = ScreenCoord / kAyanami_ScreenProbeUniformPlaceTileWidth;
        uint2 ProbeId01 = ProbeId00 + uint2(1, 0);
        uint2 ProbeId10 = ProbeId00 + uint2(0, 1);
        uint2 ProbeId11 = ProbeId00 + uint2(1, 1);
    
        uint ProbeId00I = ProbeId00.x + ProbeId00.y * ProbeCntPerX;
        uint ProbeId01I = ProbeId01.x + ProbeId01.y * ProbeCntPerX;
        uint ProbeId10I = ProbeId10.x + ProbeId10.y * ProbeCntPerX;
        uint ProbeId11I = ProbeId11.x + ProbeId11.y * ProbeCntPerX;
    
        float3 WorldNormal = normalize(CoordSample.m_WorldNormal);
        Math::FThreeBandSH CosineLobeNormal = Math::SHCosineLobe3Encode(WorldNormal);
        float ValR[4];
        float ValG[4];
        float ValB[4];
    
        ValR[0] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId00I).m_Red);
        ValR[1] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId01I).m_Red);
        ValR[2] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId10I).m_Red);
        ValR[3] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId11I).m_Red);
        ValG[0] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId00I).m_Green);
        ValG[1] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId01I).m_Green);
        ValG[2] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId10I).m_Green);
        ValG[3] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId11I).m_Green);
        ValB[0] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId00I).m_Blue);
        ValB[1] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId01I).m_Blue);
        ValB[2] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId10I).m_Blue);
        ValB[3] = Math::DotSH3(CosineLobeNormal, GetProbeSH(ProbeId11I).m_Blue);
        
        float3 SHVal[4];
        SHVal[0] = float3(ValR[0], ValG[0], ValB[0]);
        SHVal[1] = float3(ValR[1], ValG[1], ValB[1]);
        SHVal[2] = float3(ValR[2], ValG[2], ValB[2]);
        SHVal[3] = float3(ValR[3], ValG[3], ValB[3]);
    
        float TotalWeights = Weights.x + Weights.y + Weights.z + Weights.w + 1e-6;
    
        SHVal[0] *= Weights.x / TotalWeights;
        SHVal[1] *= Weights.y / TotalWeights;
        SHVal[2] *= Weights.z / TotalWeights;
        SHVal[3] *= Weights.w / TotalWeights;
    
        float3 FinalSH = float3(0.0);
        FinalSH += SHVal[0];
        FinalSH += SHVal[1];
        FinalSH += SHVal[2];
        FinalSH += SHVal[3];
        
        return FinalSH;
    }
    

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiScrProbePixelGatherKernelSize,
        kAyanamiScrProbePixelGatherKernelSize,
        1
    )
    void ScreenProbePixelGatherCS(
        uint3 ThreadID : SV_DispatchThreadID,
        uint3 GroupID : SV_GroupID,
        uint3 GroupThreadID : SV_GroupThreadID
    )
    {
        if(ThreadID.x >= PushConst.m_RTWidth ||ThreadID.y >= PushConst.m_RTHeight)
        {
            return; 
        }
        ScreenSpaceSample CoordSample = GetScreenSample(ThreadID.xy);
        float4 ProbeWeights = GetNeighbourProbeWeights(ThreadID.xy, CoordSample);

        if(!CoordSample.m_Valid){
            PushConst.m_OutTex.Store(ThreadID.xy, float4(0.0f, 0.0f, 0.0f, 1.0f));
            return;
        }
        float3 GatheredSH = GatherProbesUniformOnly(ThreadID.xy, CoordSample, ProbeWeights);
        PushConst.m_OutTex.Store(ThreadID.xy, float4(GatheredSH, 1.0f));
    }

}
}