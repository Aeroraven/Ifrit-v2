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
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"
#include "Math.SphericalHarmonics.glsl"
#include "Math.Sampling.glsl"
#include "Math.RayUtils.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

layout(
    local_size_x = kAyanamiScrProbeGDFTraceKernelSize, 
    local_size_y = 1, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec4 m_WorldBoundMin;
    vec4 m_WorldBoundMax;
    vec2 m_RayJitter;
    uint m_GlobalDFSRV;
    uint m_GlobalDFTraceProposalCounterUAV;
    uint m_GlobalDFTraceProposalListUAV;
    uint m_AdaptiveProbesListUAV;
    uint m_PerFrameCBV;
    uint m_RTWidth;
    uint m_RTHeight;
    uint m_GBufferDepthSRV;
    uint m_ScreenProbeLightingAtlasUAV;
}PushConst;

const float kRayProceedAdvance = 1e-3;

struct TraceRayProposal{
    uvec2 m_TraceRayCoord;
    uint m_ProbeId;
};


RegisterStorage(BGlobalDFTraceProposalIndirectArgs,{
    uint m_MdfFailureRays;
    uint m_InvoX; // indirect args for mdf tracing for ss probes
    uint m_InvoY;
    uint m_InvoZ;
});

RegisterStorage(BGloablDFTraceProposalList,{
    uint m_List[];
});

RegisterStorage(BAdaptiveProbesList,{
    uint m_PackedCoords[];
});


TraceRayProposal UnpackLocationAndRay(uint packedData){
    // low->high: (3bit rayId_x, 3bit rayId_y,  26bit probeId)
    uint RayIdX = packedData & 0x7; // 3bit
    uint RayIdY = (packedData >> 3) & 0x7; // 3bit
    uint ProbeId = (packedData >> 6) & 0x3FFFFFF; // 26bit
    return TraceRayProposal(uvec2(RayIdX, RayIdY), ProbeId);
}

uvec2 GetProbeWritingSlot(uint ProbeId, uint ProbeCntPerX, uvec2 TraceRayCoord){
    uint ProbeX = ProbeId % ProbeCntPerX;
    uint ProbeY = ProbeId / ProbeCntPerX;
    uint ProbeLocX = ProbeX * kAyanami_ScreenProbeProbeHemiRes;
    uint ProbeLocY = ProbeY * kAyanami_ScreenProbeProbeHemiRes;
    uvec2 ProbeLoc = uvec2(ProbeLocX, ProbeLocY);
    uvec2 WritingSlot = ProbeLoc + TraceRayCoord;
    return WritingSlot; 
}

float TraceGlobalDF(vec3 RayOrigin, vec3 RayDir){
    float HitTime = AyaShared_RayMarchGlobalDF(RayOrigin,RayDir,PushConst.m_GlobalDFSRV,PushConst.m_WorldBoundMin.xyz,
        PushConst.m_WorldBoundMax.xyz,0.017,0.03,180);

    return HitTime;
}

uvec2 UnpackLocation(uint PackedCoords){
    uint y = PackedCoords & 0xFFFF;
    uint x = (PackedCoords >> 16) & 0xFFFF;
    return uvec2(x,y);
}

uvec2 GetAdaptiveProbeCoord(uint AdaptiveProbeId){
    uint PackedCoord = GetResource(BAdaptiveProbesList,PushConst.m_AdaptiveProbesListUAV).m_PackedCoords[AdaptiveProbeId];
    return UnpackLocation(PackedCoord);
}

void main(){
    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    float ClipNear = PerFrame.m_cameraNear;
    float ClipFar = PerFrame.m_cameraFar;
    mat4 ClipToWorld = PerFrame.m_clipToWorld;

    uint TraceRayId = gl_GlobalInvocationID.x;
    uint TotalTraceRays = GetResource(BGlobalDFTraceProposalIndirectArgs,PushConst.m_GlobalDFTraceProposalCounterUAV).m_MdfFailureRays;
    if(TraceRayId < TotalTraceRays){
        uint TraceRayPackedData = GetResource(BGloablDFTraceProposalList,PushConst.m_GlobalDFTraceProposalListUAV).m_List[TraceRayId];
        TraceRayProposal TraceRay = UnpackLocationAndRay(TraceRayPackedData);

        uint ProbeCntPerX = ifrit_DivRoundUp(PushConst.m_RTWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
        uint ProbeCntPerY = ifrit_DivRoundUp(PushConst.m_RTHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
        uint TotalUniformProbes = ProbeCntPerX * ProbeCntPerY;

        vec2 ProbeUV;
        if(TraceRay.m_ProbeId < TotalUniformProbes){
            uint ProbeIdX = TraceRay.m_ProbeId % ProbeCntPerX;
            uint ProbeIdY = TraceRay.m_ProbeId / ProbeCntPerX;
            uint ProbeLocX = ProbeIdX * kAyanami_ScreenProbeUniformPlaceTileWidth;
            uint ProbeLocY = ProbeIdY * kAyanami_ScreenProbeUniformPlaceTileWidth;
            ProbeUV = vec2(float(ProbeLocX),float(ProbeLocY)) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
        }else{
            uint AdaptiveProbeId = TraceRay.m_ProbeId - TotalUniformProbes;
            uvec2 AdaptiveProbeCoord = GetAdaptiveProbeCoord(AdaptiveProbeId);
            ProbeUV = vec2(AdaptiveProbeCoord) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
        }

        float ProbeLocDepthNDC = SampleTexture2D(PushConst.m_GBufferDepthSRV, sNearestClamp, ProbeUV).r;
        vec3 ProbeLocNDC = vec3(ProbeUV*2.0-1.0, ProbeLocDepthNDC);
        float ProbeLocDepthVS = ifrit_recoverViewSpaceDepth(ProbeLocDepthNDC, ClipNear, ClipFar);
        vec3 ProbeLocCS = ProbeLocNDC * ProbeLocDepthVS;
        vec4 ProbeLocWSH = ClipToWorld * vec4(ProbeLocCS, ProbeLocDepthVS);
        vec3 ProbeLocWS = ProbeLocWSH.xyz / ProbeLocWSH.w;

        vec2 TraceRayUV =(vec2(TraceRay.m_TraceRayCoord) + vec2(PushConst.m_RayJitter)) / vec2(kAyanami_ScreenProbeProbeHemiRes);
        vec4 SampledRayAndPDF = ifrit_SampleUniformSphereWithPDF(TraceRayUV);
        vec3 SampledRay = SampledRayAndPDF.xyz;
        float SampledRayPDF = SampledRayAndPDF.w;

        ProbeLocWS += SampledRay * kRayProceedAdvance;
        uvec2 WritingSlot = GetProbeWritingSlot(TraceRay.m_ProbeId, ProbeCntPerX, TraceRay.m_TraceRayCoord);

        float TraceResult = TraceGlobalDF(ProbeLocWS, SampledRay);
        if(TraceResult>=0.0){
            imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(0.0,0.0,1.0, 0.0));
        }
    }
}