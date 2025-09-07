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
        float2 m_RayJitter;
        FAdaptiveProbesCounterHandle m_AdaptiveProbeCounter;
        TRWStructuredBufferHandle<uint> m_AdaptiveProbeList;
        uint m_RtWidth;
        uint m_RtHeight;
        TRWTexture2DHandle<float4> m_ScreenProbeLightingAtlas;
        TRWStructuredBufferHandle<float> m_OutputSHCoefBuffer;
    };

    void WriteSH(uint Offset, float Val)
    {
        PushConst.m_OutputSHCoefBuffer.Store(Val,Offset);
    }

    void WriteSH_Channel(uint Offset,Math::FThreeBandSH SHCoeffs){
        WriteSH(Offset, SHCoeffs.m_Coef1.x);
        WriteSH(Offset+1, SHCoeffs.m_Coef1.y);
        WriteSH(Offset+2, SHCoeffs.m_Coef1.z);
        WriteSH(Offset+3, SHCoeffs.m_Coef1.w);
        WriteSH(Offset+4, SHCoeffs.m_Coef2.x);
        WriteSH(Offset+5, SHCoeffs.m_Coef2.y);
        WriteSH(Offset+6, SHCoeffs.m_Coef2.z);
        WriteSH(Offset+7, SHCoeffs.m_Coef2.w);
        WriteSH(Offset+8, SHCoeffs.m_Coef3);
    }

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiScrProbeIntegrateSHKernelSize,
        1,
        1
    )
    void ScreenProbeIntegrateSHCS(uint3 ThreadID: SV_DispatchThreadID,
                                  uint3 GroupID: SV_GroupID,
                                  uint3 GroupThreadID: SV_GroupThreadID)
    {
        uint ProbeCntPerX = DivRoundUp(PushConst.m_RtWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
        uint ProbeCntPerY = DivRoundUp(PushConst.m_RtHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
        uint TotalUniformProbes = ProbeCntPerX * ProbeCntPerY;
        uint TotalProbes = PushConst.m_AdaptiveProbeCounter.GetCounter() + TotalUniformProbes;

        uint ProbeId = ThreadID.x;
        if(ProbeId >= TotalProbes) return;    

        float2 ProbeUV;
        if(ProbeId < TotalUniformProbes){
            uint ProbeIdX = ProbeId % ProbeCntPerX;
            uint ProbeIdY = ProbeId / ProbeCntPerX;
            uint ProbeLocX = ProbeIdX * kAyanami_ScreenProbeUniformPlaceTileWidth;
            uint ProbeLocY = ProbeIdY * kAyanami_ScreenProbeUniformPlaceTileWidth;
            ProbeUV = float2(float(ProbeLocX),float(ProbeLocY)) / float2(PushConst.m_RtWidth, PushConst.m_RtHeight);
        }else{
            uint AdaptiveProbeId = ProbeId - TotalUniformProbes;
            uint2 AdaptiveProbeCoord = UnpackAdaptiveProbeLocation(
                PushConst.m_AdaptiveProbeList.Load(AdaptiveProbeId)
            );
            ProbeUV = float2(AdaptiveProbeCoord) / float2(PushConst.m_RtWidth, PushConst.m_RtHeight);
        }

        Math::FThreeBandSHRgb SHCoeffs = Math::ZeroSH3Rgb();

        for(uint i=0;i<kAyanami_ScreenProbeProbeHemiRes;i++){
            for(uint j=0;j<kAyanami_ScreenProbeProbeHemiRes;j++){
                uint2 TraceRayCoord = uint2(i,j);
                uint2 WritingSlot;

                WritingSlot = GetScreenSpaceProbeWritingSlotAfterFix(ProbeId, ProbeCntPerX, TraceRayCoord+uint2(1,1));
                
                float3 SampledRay =  GetScreenProbeTraceCoord(TraceRayCoord,PushConst.m_RayJitter);
                float3 Irradiance = PushConst.m_ScreenProbeLightingAtlas.Load(WritingSlot).xyz;
                
                Math::FThreeBandSHRgb RayBasisRGB = Math::SHBasis3EncodeRgb(SampledRay);
                Math::FThreeBandSHRgb RayBasisRGBScaled = Math::MulSH3Color(RayBasisRGB, Irradiance * 4.0 * kPI);
                SHCoeffs = Math::AddSH3(SHCoeffs, RayBasisRGBScaled);
            }
        }

        SHCoeffs = Math::MulSH3(SHCoeffs, 1.0 / float(kAyanami_ScreenProbeProbeHemiRes * kAyanami_ScreenProbeProbeHemiRes));
        uint OutSHCoefOffset = 27*ProbeId;

        WriteSH_Channel(OutSHCoefOffset, SHCoeffs.m_Red);
        WriteSH_Channel(OutSHCoefOffset+9, SHCoeffs.m_Green);
        WriteSH_Channel(OutSHCoefOffset+18, SHCoeffs.m_Blue);
    }
}
}