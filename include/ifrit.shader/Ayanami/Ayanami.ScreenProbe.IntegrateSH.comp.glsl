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
#include "Math.Sampling.glsl"
#include "Math.RayUtils.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"
#include "Ayanami/Ayanami.ScreenProbe.Shared.glsl"

// TODO: it's expected to make probe process per TG. here the simplified solution is used.
// That is, a probe per thread

layout(
    local_size_x = kAyanamiScrProbeIntegrateSHKernelSize, 
    local_size_y = 1, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec2 m_RayJitter;
    uint m_AdaptiveProbesCounterUAV;
    uint m_AdaptiveProbesListUAV;
    uint m_RTWidth;
    uint m_RTHeight;
    uint m_ScreenProbeLightingAtlasUAV;
    uint m_OutputSHCoefBufferUAV;
} PushConst;

RegisterStorage(BAdaptiveProbesCounter,{
    uint m_Counter;
    uint m_InvoX; // below two are used for debugging only
    uint m_InvoY; 
    uint m_InvoZ;
    uint m_InvoTraceX; // used for screen space tracing
    uint m_InvoTraceY;
    uint m_InvoTraceZ;
});

RegisterStorage(BAdaptiveProbesList,{
    uint m_PackedCoords[];
});

RegisterStorage(BOutputSHCoefBuffer,{
    float m_Data[];
});

uvec2 UnpackLocation(uint PackedCoords){
    uint y = PackedCoords & 0xFFFF;
    uint x = (PackedCoords >> 16) & 0xFFFF;
    return uvec2(x,y);
}

uvec2 GetAdaptiveProbeCoord(uint AdaptiveProbeId){
    uint PackedCoord = GetResource(BAdaptiveProbesList,PushConst.m_AdaptiveProbesListUAV).m_PackedCoords[AdaptiveProbeId];
    return UnpackLocation(PackedCoord);
}

uvec2 GetProbeWritingSlot(uint ProbeId, uint ProbeCntPerX, uvec2 TraceRayCoord){
    uint ProbeX = ProbeId % ProbeCntPerX;
    uint ProbeY = ProbeId / ProbeCntPerX;
    uint ProbeLocX; //= ProbeX * (kAyanami_ScreenProbeProbeHemiRes+2);
    uint ProbeLocY; //= ProbeY * (kAyanami_ScreenProbeProbeHemiRes+2);
    if(kEnableOctMapBorderFix){
        ProbeLocX = ProbeX * (kAyanami_ScreenProbeProbeHemiRes+2);
        ProbeLocY = ProbeY * (kAyanami_ScreenProbeProbeHemiRes+2);
    }else{
        ProbeLocX = ProbeX * kAyanami_ScreenProbeProbeHemiRes;
        ProbeLocY = ProbeY * kAyanami_ScreenProbeProbeHemiRes;
    }
    uvec2 ProbeLoc = uvec2(ProbeLocX, ProbeLocY);
    uvec2 WritingSlot = ProbeLoc + TraceRayCoord;
    return WritingSlot; 
}

void WriteSH(uint Offset,float Val){
    GetResource(BOutputSHCoefBuffer,PushConst.m_OutputSHCoefBufferUAV).m_Data[Offset] = Val;
}

void WriteSH_Channel(uint Offset,MThreeBandSH SHCoeffs){
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

void main(){
    uint ProbeCntPerX = ifrit_DivRoundUp(PushConst.m_RTWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint ProbeCntPerY = ifrit_DivRoundUp(PushConst.m_RTHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint TotalUniformProbes = ProbeCntPerX * ProbeCntPerY;
    uint TotalProbes = GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_Counter+ TotalUniformProbes;

    uint ProbeId = gl_GlobalInvocationID.x;
    if(ProbeId >= TotalProbes) return;    

    vec2 ProbeUV;
    if(ProbeId < TotalUniformProbes){
        uint ProbeIdX = ProbeId % ProbeCntPerX;
        uint ProbeIdY = ProbeId / ProbeCntPerX;
        uint ProbeLocX = ProbeIdX * kAyanami_ScreenProbeUniformPlaceTileWidth;
        uint ProbeLocY = ProbeIdY * kAyanami_ScreenProbeUniformPlaceTileWidth;
        ProbeUV = vec2(float(ProbeLocX),float(ProbeLocY)) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    }else{
        uint AdaptiveProbeId = ProbeId - TotalUniformProbes;
        uvec2 AdaptiveProbeCoord = GetAdaptiveProbeCoord(AdaptiveProbeId);
        ProbeUV = vec2(AdaptiveProbeCoord) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    }

    MThreeBandSH_RGB SHCoeffs = ifrit_ZeroSH3RGB();

    for(uint i=0;i<kAyanami_ScreenProbeProbeHemiRes;i++){
        for(uint j=0;j<kAyanami_ScreenProbeProbeHemiRes;j++){
            uvec2 TraceRayCoord = uvec2(i,j);

            uvec2 WritingSlot;
            if(kEnableOctMapBorderFix){
                WritingSlot = GetProbeWritingSlot(ProbeId, ProbeCntPerX, TraceRayCoord+uvec2(1,1));
            }else{
                WritingSlot = GetProbeWritingSlot(ProbeId, ProbeCntPerX, TraceRayCoord);
            }
            
            vec3 SampledRay =  AyaShared_GetScreenProbeTraceCoord(TraceRayCoord,PushConst.m_RayJitter);
            vec3 Radiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot)).xyz;

            MThreeBandSH_RGB RayBasisRGB = ifrit_SHBasis3EncodeRGB(SampledRay);
            MThreeBandSH_RGB RayBasisRGBScaled = ifrit_MulSH3RGBColor(RayBasisRGB, Radiance);
            SHCoeffs = ifrit_AddSH3RGB(SHCoeffs, RayBasisRGBScaled);
        }
    }

    SHCoeffs = ifrit_MulSH3RGB(SHCoeffs, 1.0 / float(kAyanami_ScreenProbeProbeHemiRes * kAyanami_ScreenProbeProbeHemiRes));
    uint OutSHCoefOffset = 27*ProbeId;

    WriteSH_Channel(OutSHCoefOffset, SHCoeffs.m_R);
    WriteSH_Channel(OutSHCoefOffset+9, SHCoeffs.m_G);
    WriteSH_Channel(OutSHCoefOffset+18, SHCoeffs.m_B);
}