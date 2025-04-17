
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
    local_size_x = kAyanami_ScreenProbeProbeHemiRes, 
    local_size_y = kAyanami_ScreenProbeProbeHemiRes, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec2 m_RayJitter;
    uint m_HizStorage;
    uint m_PerFrameCBV;
    uint m_RTWidth;
    uint m_RTHeight;
    uint m_ScreenProbeLightingAtlasUAV;
    uint m_MeshDFTraceProposalCounterUAV;
    uint m_MeshDFTraceProposalListUAV;
    uint m_AdaptiveProbesListUAV;
}PushConst;

const float kRayProceedMax = 1000.0;
const float kRayProceedAdvance = 1e-3;

RegisterStorage(BHiZStorage,{
    uint m_Pad;
    uint m_Mip[];
});

RegisterStorage(BAdaptiveProbesList,{
    uint m_PackedCoords[];
});

// the compacted list for the failed rays
RegisterStorage(BMeshDFTraceProposalIndirectArgs,{
    uint m_SsgiFailureRays;
    uint m_InvoX; // indirect args for mdf tracing for ss probes
    uint m_InvoY;
    uint m_InvoZ;
});

RegisterStorage(BMeshDFTraceProposalList,{
    uint m_List[];
});

shared uint sFailureRayCount;
shared uint sFailureRayGlobalStart;
shared uint sFailureRayList[kAyanami_ScreenProbeProbeHemiRes * kAyanami_ScreenProbeProbeHemiRes];

uvec2 UnpackLocation(uint PackedCoords){
    uint y = PackedCoords & 0xFFFF;
    uint x = (PackedCoords >> 16) & 0xFFFF;
    return uvec2(x,y);
}

uint PackLocationAndRay(uint ProbeId, uvec2 RayId){
    // low->high: (3bit rayId_x, 3bit rayId_y,  26bit probeId)
    uint RayIdXEnc = (RayId.x & 0x7);
    uint RayIdYEnc = (RayId.y & 0x7) << 3;
    uint ProbeXEnc = (ProbeId & 0x3FFFFFF) << 6;
    uint PackedCoord = RayIdXEnc | RayIdYEnc | ProbeXEnc;
    return PackedCoord;
}

uvec2 GetAdaptiveProbeCoord(uint AdaptiveProbeId){
    uint PackedCoord = GetResource(BAdaptiveProbesList,PushConst.m_AdaptiveProbesListUAV).m_PackedCoords[AdaptiveProbeId];
    return UnpackLocation(PackedCoord);
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

float GetHizDepth(ivec2 UV, uint Mip, bool Ranged){
    ivec2 MipUV = UV >> Mip;
    uint MipId = GetResource(BHiZStorage,PushConst.m_HizStorage).m_Mip[Mip];
    if(!Ranged){
        return imageLoad(GetUAVImage2DR32F(MipId), MipUV).r;
    }
    float t0 = imageLoad(GetUAVImage2DR32F(MipId), MipUV).r;
    float t1 = imageLoad(GetUAVImage2DR32F(MipId), MipUV+ivec2(1,0)).r;
    float t2 = imageLoad(GetUAVImage2DR32F(MipId), MipUV+ivec2(0,1)).r;
    float t3 = imageLoad(GetUAVImage2DR32F(MipId), MipUV+ivec2(1,1)).r;
    return min(min(t0,t1),min(t2,t3));
}

float GetHizDepth(vec2 UV, uint Mip){
    vec2 PixelUV = UV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    ivec2 PixelUVInt = ivec2(PixelUV);
    return GetHizDepth(PixelUVInt, Mip, false);
}

float GetHizDepthRanged(vec2 UV, uint Mip){
    vec2 PixelUV = UV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    ivec2 PixelUVInt = ivec2(PixelUV);
    return GetHizDepth(PixelUVInt, Mip, true);
}

vec3 SsgiTraceImpl(vec3 RayStartVS, vec3 RayEndVS, vec2 RayStartUV, vec2 RayEndUV){
    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    float ClipNear = PerFrame.m_cameraNear;
    float ClipFar = PerFrame.m_cameraFar;


    vec2 DiffUV = RayEndUV - RayStartUV;
    vec2 DiffPixels = DiffUV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    ivec2 DiffPixelsInt = ivec2(DiffPixels);
    ivec2 RayStartUVInt = ivec2(RayStartUV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight));

    float MaxStepsF = max(abs(DiffPixels.x), abs(DiffPixels.y));
    uint MaxSteps = uint(MaxStepsF) + 1; // total steps required on marching Hiz level 0
    float MinimalStep = 1.0 / float(MaxSteps);
    
    int CurMip = 0;
    bool FinalHit = false;
    vec2 HitUV = vec2(0.0);
    float DepthDiffVS = 0.0;

    int ProceedSignX = DiffPixels.x > 0.0 ? 1 : -1;
    int ProceedSignY = DiffPixels.y > 0.0 ? 1 : -1;
    float CurStepF  = 0.0;
    int MaxIters = 40;
    int CurIters = 0;
    while(CurStepF <= 1 && CurMip >= 0 && CurIters < MaxIters){
        CurIters += 1;
        float T = CurStepF;
        vec2 CurUV = mix(RayStartUV, RayEndUV, T);
        float ReferenceZ = GetHizDepth(CurUV, CurMip);
        ReferenceZ = ifrit_recoverViewSpaceDepth(ReferenceZ, ClipNear, ClipFar);
        float CurZ = ifrit_perspectiveLerp(RayStartVS.z, RayEndVS.z, RayStartVS.z, RayEndVS.z, T);
        ivec2 CurUVInt = ivec2(CurUV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight));
        ivec2 CurUVIntMip = CurUVInt >> CurMip;

        bool IsCollided = false;
        if(CurZ > ReferenceZ){
            IsCollided = true;
        }

        if(IsCollided && CurMip == 0){
            FinalHit = true;
            HitUV = mix(RayStartUV, RayEndUV, T);
            DepthDiffVS = CurZ - ReferenceZ;
            break;
        }

        // step if not collided
        if(!IsCollided){
            int NextTexelX = ((CurUVIntMip.x + ProceedSignX)<<CurMip) - RayStartUVInt.x;
            int NextTexelY = ((CurUVIntMip.y + ProceedSignY)<<CurMip) - RayStartUVInt.y;
            float StepX = float(NextTexelX) / float(DiffPixelsInt.x);
            float StepY = float(NextTexelY) / float(DiffPixelsInt.y);
            float NextStep = max(CurStepF+MinimalStep, min(abs(StepX), abs(StepY)));
            CurStepF = NextStep;
            CurMip = min(CurMip+1, 6);
        }else{
            CurMip-=1;
        }
    }

    // Check the hit z difference
    if(abs(DepthDiffVS) > 5e-2){
        FinalHit = false;
    }

    return vec3(HitUV, FinalHit ? 1.0 : 0.0);
}

vec3 SsgiTrace(vec3 RayDirWS, vec3 RayOriginWS){

    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    bool ValidSample = true;
    mat4 WorldToClip = PerFrame.m_worldToClip;
    mat4 ClipToWorld = PerFrame.m_clipToWorld;
    mat4 WorldToView = PerFrame.m_worldToView;
    mat4 ViewToClip = PerFrame.m_perspective;
    float ClipNear = PerFrame.m_cameraNear;

    vec4 OriginVS = WorldToView * vec4(RayOriginWS, 1.0);
    vec4 ProceedVS = WorldToView * vec4(RayOriginWS + RayDirWS * kRayProceedMax, 1.0);
    vec4 RayDirVS = ProceedVS - OriginVS;
    // Need to limit the proceed vs in the view frustum. (at least, larger than near plane)
    if(ProceedVS.z < ClipNear+1e-3){
        // find O+td intersection with near plane.
        float t = (ClipNear+1e-3 - OriginVS.z) / RayDirVS.z;
        if(abs(RayDirVS.z) < 1e-4){
           ValidSample = false;
        }
        ProceedVS = OriginVS + RayDirVS * t;
    }

    vec4 OriginCS = ViewToClip * OriginVS;
    vec4 ProceedCS = ViewToClip * ProceedVS;

    vec2 OriginNDCxy = OriginCS.xy / OriginCS.w;
    vec2 ProceedNDCxy = ProceedCS.xy / ProceedCS.w;
    vec2 OriginUV = (OriginNDCxy + 1.0) * 0.5;
    vec2 ProceedUV = (ProceedNDCxy + 1.0) * 0.5;

    vec2 ProceedDirUV = ProceedUV - OriginUV;
    vec2 RayNDCIntersection = ifrit_RayIntersectWithUnitRect2D(OriginUV, ProceedDirUV);

    // The tracing center does not present in the screen space.
    if(OriginUV.x < 0.0 || OriginUV.x > 1.0 || OriginUV.y < 0.0 || OriginUV.y > 1.0){
        ValidSample = false;
    }
    if(RayNDCIntersection.x>0.0 || RayNDCIntersection.y < 0.0){
        ValidSample = false;
    }
    vec2 ClampedDistantUV = OriginUV + RayNDCIntersection.y * ProceedDirUV;
    vec2 AbsProceedDirUV = abs(ProceedDirUV);
    if(AbsProceedDirUV.x < 1e-4 || AbsProceedDirUV.y < 1e-4){
        ValidSample = false;
    }
    // Get the ray end point in the view space
    vec3 ClampedDistantVS = ifrit_perspectiveLerp3D(OriginVS.xyz, ProceedVS.xyz, OriginVS.z, ProceedVS.z, RayNDCIntersection.y);

    vec3 RayTraceStartVS = OriginVS.xyz;
    vec3 RayTraceEndVS = ClampedDistantVS;
    vec2 RayTraceStartUV = OriginUV;
    vec2 RayTraceEndUV = ClampedDistantUV;

    vec3 RayTraceDirNew = normalize(RayTraceEndVS - RayTraceStartVS);
    vec3 RayTraceDirOld = normalize(ProceedVS.xyz - OriginVS.xyz);
    
    if(!ValidSample){
        // The ray is not valid, return the invalid color
        return vec3(0.0, 0.0, 0.0);
    }
    return SsgiTraceImpl(RayTraceStartVS, RayTraceEndVS, RayTraceStartUV, RayTraceEndUV);
}

void main(){
    if(ifrit_IsFirstLane()){
        sFailureRayCount = 0;
    }
    if(ifrit_IsGlobalFirstThread()){
        GetResource(BMeshDFTraceProposalIndirectArgs,PushConst.m_MeshDFTraceProposalListUAV).m_InvoY = 1;
        GetResource(BMeshDFTraceProposalIndirectArgs,PushConst.m_MeshDFTraceProposalListUAV).m_InvoZ = 1;
    }
    barrier();


    uint ProbeId = gl_WorkGroupID.x;
    uvec2 TraceRayCoord = gl_LocalInvocationID.xy;
    vec2 TraceRayUV = (vec2(TraceRayCoord)+PushConst.m_RayJitter) / vec2(kAyanami_ScreenProbeProbeHemiRes);
    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    float ClipNear = PerFrame.m_cameraNear;
    float ClipFar = PerFrame.m_cameraFar;
    mat4 ClipToWorld = PerFrame.m_clipToWorld;

    // Note that, importance sampling is not used here. It's scheduled in the future.
    vec4 SampledRayAndPDF = ifrit_SampleUniformSphereWithPDF(TraceRayUV);
    vec3 SampledRay = SampledRayAndPDF.xyz;
    float SampledRayPDF = SampledRayAndPDF.w;

    // Get probe center WS
    uint ProbeCntPerX = ifrit_DivRoundUp(PushConst.m_RTWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint ProbeCntPerY = ifrit_DivRoundUp(PushConst.m_RTHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint TotalUniformProbes = ProbeCntPerX * ProbeCntPerY;
    
    vec3 ProbeLocWS;
    vec2 ProbeUV;
    if(ProbeId < TotalUniformProbes){
        uint ProbeX = ProbeId % ProbeCntPerX;
        uint ProbeY = ProbeId / ProbeCntPerX;
        uint ProbeLocX = ProbeX * kAyanami_ScreenProbeUniformPlaceTileWidth;
        uint ProbeLocY = ProbeY * kAyanami_ScreenProbeUniformPlaceTileWidth;
        ProbeUV = vec2(ProbeLocX, ProbeLocY) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    }else{
        // This is an adaptive probe
        uint AdaptiveProbeId = ProbeId - TotalUniformProbes;
        uvec2 AdaptiveProbeCoord = GetAdaptiveProbeCoord(AdaptiveProbeId);
        ProbeUV = vec2(AdaptiveProbeCoord) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    }

    float ProbeLocDepthNDC = GetHizDepth(ProbeUV, 0);
    bool ValidProbe = true;
    if(ProbeLocDepthNDC >= 1.0){
        ValidProbe = false;
    }
    float ProbeLocDepthVS = ifrit_recoverViewSpaceDepth(ProbeLocDepthNDC, ClipNear, ClipFar);

    vec3 ProbeLocNDC = vec3(ProbeUV * 2.0-1.0, ProbeLocDepthNDC);
    vec3 ProbeLocCS = ProbeLocNDC * ProbeLocDepthVS;
    vec4 ProbeLocWSH = ClipToWorld * vec4(ProbeLocCS, ProbeLocDepthVS);
    ProbeLocWS = ProbeLocWSH.xyz / ProbeLocWSH.w;
    ProbeLocWS += SampledRay * kRayProceedAdvance;

    vec3 SsgiTraceResult = ValidProbe ? SsgiTrace(SampledRay, ProbeLocWS) : vec3(0.0, 0.0, 0.0);

    // TODO: sample lighting
    uvec2 WritingSlot = GetProbeWritingSlot(ProbeId, ProbeCntPerX, TraceRayCoord);
    if(!ValidProbe){
        // probe is not valid, write the invalid color
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(0.0, 0.0, 1.0, 1.0));
        
    }else if(SsgiTraceResult.z < 0.5){
        // screen hit miss
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(1.0, 0.0, 0.0, 1.0));
        uint FailureRayId = atomicAdd(sFailureRayCount, 1);
        sFailureRayList[FailureRayId] = PackLocationAndRay(ProbeId, TraceRayCoord);
    }else{
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(0.0, 1.0, 0.0, 1.0));
    }

    barrier();

    // writing into compact list
    if(ifrit_IsFirstLane()){
        uint FailureRayCount = sFailureRayCount;
        uint GlobalListStart = atomicAdd(GetResource(BMeshDFTraceProposalIndirectArgs,PushConst.m_MeshDFTraceProposalCounterUAV).m_SsgiFailureRays, FailureRayCount);
        uint TotalFailureRays = GlobalListStart + FailureRayCount;
        uint TotalMDFTraceTGs = ifrit_DivRoundUp(TotalFailureRays, kAyanamiScrProbeMDFTraceKernelSize);
        sFailureRayGlobalStart = GlobalListStart;
        atomicMax(GetResource(BMeshDFTraceProposalIndirectArgs,PushConst.m_MeshDFTraceProposalCounterUAV).m_InvoX, TotalMDFTraceTGs);
    }
    barrier();

    uint LocalId = gl_LocalInvocationIndex;
    if(LocalId < sFailureRayCount){
        uint FailureRayId = sFailureRayList[LocalId];
        uint GlobalRayId = sFailureRayGlobalStart + LocalId;
        GetResource(BMeshDFTraceProposalList,PushConst.m_MeshDFTraceProposalListUAV).m_List[GlobalRayId] = FailureRayId;
    }
}