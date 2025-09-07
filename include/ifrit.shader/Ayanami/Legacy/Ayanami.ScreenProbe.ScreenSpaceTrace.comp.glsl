
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
    uint m_LastFrameFinalLightingSRV;
}PushConst;

const float kRayProceedMax = 5.0;
const float kRayProceedAdvance = 6e-3;
const uint kMaxTraceIters = 60;
const bool kHizProceed = true;
const bool kUseWordSpaceSsgi = false;

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
    if(UV.x<0.0 || UV.x>1.0 || UV.y<0.0 || UV.y>1.0){
        return -1.0;
    }
    vec2 PixelUV = UV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    ivec2 PixelUVInt = ivec2(PixelUV);
    return GetHizDepth(PixelUVInt, Mip, false);
}

float GetHizDepthPx(vec2 UV, uint Mip){
    if(UV.x<0.0 || UV.x>PushConst.m_RTWidth || UV.y<0.0 || UV.y>PushConst.m_RTHeight){
        return -1.0;
    }
    vec2 PixelUV = UV;
    ivec2 PixelUVInt = ivec2(PixelUV);
    return GetHizDepth(PixelUVInt, Mip, false);
}

float GetHizDepthRanged(vec2 UV, uint Mip){
    vec2 PixelUV = UV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    ivec2 PixelUVInt = ivec2(PixelUV);
    return GetHizDepth(PixelUVInt, Mip, true);
}

vec3 SsgiTraceImpl(vec3 RayStartVS, vec3 RayEndVS, vec2 RayStartUV, vec2 RayEndUV){
    // To alleviate the rounding problem, some strategies are used:
    // References from: 
    // https://github.com/Raphael2048/FengRender/blob/master/resources/shaders/ssr.hlsl

    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    float ClipNear = PerFrame.m_cameraNear;
    float ClipFar = PerFrame.m_cameraFar;

    vec2 DiffUV = RayEndUV - RayStartUV;
    vec2 DiffPixels = DiffUV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
    ivec2 DiffPixelsInt = ivec2(DiffPixels);
    
    float MaxStepsF = max(abs(DiffPixels.x), abs(DiffPixels.y));
    uint MaxSteps = uint(MaxStepsF) + 1; // total steps required on marching Hiz level 0
    float MinimalStep = 1.0 / float(MaxSteps);
    
    int CurMip = 0;
    bool FinalHit = false;
    vec2 HitUV = vec2(0.0);
    float DepthDiffVS = 0.0;

    int ProceedSignX = DiffPixels.x > 0.0 ? 1 : -1;
    int ProceedSignY = DiffPixels.y > 0.0 ? 1 : -1;

    int ProceedStepX = DiffPixels.x >= 0.0 ? 1 : 0;
    int ProceedStepY = DiffPixels.y >= 0.0 ? 1 : 0;

    float CurStepF  = 0.0;
    int MaxIters = int(kMaxTraceIters);
    int CurIters = 0;

    float ProceedTexelX = 0.0;
    float ProceedTexelY = 0.0;
    float ProceedRefZ = 0.0;
    float ProceedCurZ = 0.0;

    vec2 RayStartPx = vec2(RayStartUV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight));
    vec2 RayEndPx = vec2(RayEndUV * vec2(PushConst.m_RTWidth, PushConst.m_RTHeight));
    vec2 CurPx = RayStartPx;
    ivec2 RayStartUVInt = ivec2(RayStartPx);

    bool MainDirectionX = abs(DiffPixels.x) > abs(DiffPixels.y);
    vec2 NormSSDirection = normalize(DiffPixels);

    float LastCurZ = 0.0;
    float LastRefZ = 0.0;
    float LastIter  = 0.0;

    while(CurMip >= 0 && CurIters < MaxIters){
        CurIters += 1;
        CurStepF = (MainDirectionX)?
            (CurPx.x - RayStartPx).x / DiffPixels.x :
            (CurPx.y - RayStartPx).y / DiffPixels.y;
        

        float T = CurStepF;
        vec2 CurUV = CurPx / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);

        float ReferenceZ =  GetHizDepthPx(CurPx, CurMip);
        bool ValidZ = ReferenceZ > 0.0 && ReferenceZ < 1.0;
        ReferenceZ = ifrit_recoverViewSpaceDepth(ReferenceZ, ClipNear, ClipFar);
        float CurZ = ifrit_PerspectiveLerpVS(RayStartVS.z, RayEndVS.z, T);
        ivec2 CurUVIntMip = ivec2(CurPx) >> CurMip;

        if(ValidZ){
            LastCurZ = CurZ;
            LastRefZ = ReferenceZ;
            LastIter = float(CurIters);
        }

        ProceedRefZ = ReferenceZ;
        ProceedCurZ = RayStartVS.z;

        bool IsCollided = false;
        if(CurZ - ReferenceZ>=-3e-4 && (CurMip!=0 || ValidZ) && T>0.0){
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
            int NextTexelX = ((CurUVIntMip.x + ProceedStepX)<<CurMip) - RayStartUVInt.x;
            int NextTexelY = ((CurUVIntMip.y + ProceedStepY)<<CurMip) - RayStartUVInt.y;

            float StepX = float(NextTexelX) / float(NormSSDirection.x);
            float StepY = float(NextTexelY) / float(NormSSDirection.y);
            
            float NextStep;
            if(abs(NormSSDirection.x)<1e-6){
                NextStep = StepY;
            }else if(abs(NormSSDirection.y)<1e-6){
                NextStep = StepX;
            }else{
                NextStep = min(StepX, StepY);
            }
            CurPx = vec2(RayStartUVInt.xy) + NextStep * NormSSDirection.xy + vec2(ProceedSignX,ProceedSignY) * vec2(0.0001);
            
            if(kHizProceed)
                CurMip = min(CurMip+1, 6);
        }else{
            if(kHizProceed)
                CurMip-=1;
        }
    }

    //return vec3(LastCurZ, LastRefZ, LastIter);

    // Check the hit z difference
    if(abs(DepthDiffVS) > 0.05){
        FinalHit = false;
        return vec3(HitUV, 0.0);
    }
    return vec3(HitUV, FinalHit ? 1.0 : 0.0);
}

vec3 SsgiTraceImplWorldSpace(vec3 RayStartWS, vec3 RayEndWS){
    // this performs ssgi in world space, or literally, ray marching in world space.
    // this is not the same as the screen space ssgi, but it is used for debugging.

    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    mat4 WorldToClip = PerFrame.m_worldToClip;
    int CurStep = 0;
    int MaxIters = int(kMaxTraceIters);
    for(CurStep = 0; CurStep < MaxIters; CurStep++){
        float T = float(CurStep) / float(MaxIters);
        vec3 CurPos = mix(RayStartWS, RayEndWS, T);
        vec4 CurPosClip = WorldToClip * vec4(CurPos, 1.0);
        vec2 CurPosNDC = CurPosClip.xy / CurPosClip.w;
        vec2 CurPosUV = (CurPosNDC + 1.0) * 0.5;
        float ReferenceZ = GetHizDepth(CurPosUV, 0);

        if(CurPos.z >= ReferenceZ && ReferenceZ > 0.0 && ReferenceZ < 1.0){
            if(abs(CurPos.z - ReferenceZ) > 0.05){
                // hit the hiz, but not the screen space
                return vec3(0.0, 0.0, 0.0);
            }
            // hit the hiz
            vec3 HitUV = vec3(CurPosUV, 1.0);
            return HitUV;
        }
    }
    // no hit
    vec3 HitUV = vec3(0.0, 0.0, 0.0);
    return HitUV;
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


    vec3 RayTraceStartVS = OriginVS.xyz;
    vec3 RayTraceEndVS = ProceedVS.xyz;
    vec2 RayTraceStartUV = OriginUV;
    vec2 RayTraceEndUV = ProceedUV;

    vec3 RayTraceDirNew = normalize(RayTraceEndVS - RayTraceStartVS);
    vec3 RayTraceDirOld = normalize(ProceedVS.xyz - OriginVS.xyz);
    
    if(!ValidSample){
        // The ray is not valid, return the invalid color
        return vec3(0.0, 0.0, 0.0);
    }

    if(kUseWordSpaceSsgi){
        // this is a world space ssgi trace, not the screen space ssgi trace.
        vec3 SsgiTraceResult = SsgiTraceImplWorldSpace(RayOriginWS, RayOriginWS + RayTraceDirNew * kRayProceedMax);
        return SsgiTraceResult;
    }
    return SsgiTraceImpl(RayTraceStartVS, RayTraceEndVS, RayTraceStartUV, RayTraceEndUV);
}

void main(){
    if(ifrit_IsFirstLane()){
        sFailureRayCount = 0;
    }
    if(ifrit_IsGlobalFirstThread()){
        GetResource(BMeshDFTraceProposalIndirectArgs,PushConst.m_MeshDFTraceProposalCounterUAV).m_InvoY = 1;
        GetResource(BMeshDFTraceProposalIndirectArgs,PushConst.m_MeshDFTraceProposalCounterUAV).m_InvoZ = 1;
    }
    barrier();


    uint ProbeId = gl_WorkGroupID.x;
    uvec2 TraceRayCoord = gl_LocalInvocationID.xy;
    
    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    float ClipNear = PerFrame.m_cameraNear;
    float ClipFar = PerFrame.m_cameraFar;
    mat4 ClipToWorld = PerFrame.m_clipToWorld;

    // Note that, importance sampling is not used here. It's scheduled in the future.
    vec3 SampledRay = AyaShared_GetScreenProbeTraceCoord(TraceRayCoord,PushConst.m_RayJitter);

    // Get probe center WS
    uint ProbeCntPerX = ifrit_DivRoundUp(PushConst.m_RTWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint ProbeCntPerY = ifrit_DivRoundUp(PushConst.m_RTHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint TotalUniformProbes = ProbeCntPerX * ProbeCntPerY;
    
    vec3 ProbeLocWS;
    vec2 ProbeUV;
    vec2 ProbeUVPx;
    if(ProbeId < TotalUniformProbes){
        uint ProbeX = ProbeId % ProbeCntPerX;
        uint ProbeY = ProbeId / ProbeCntPerX;
        uint ProbeLocX = ProbeX * kAyanami_ScreenProbeUniformPlaceTileWidth;
        uint ProbeLocY = ProbeY * kAyanami_ScreenProbeUniformPlaceTileWidth;
        ProbeUV = vec2(ProbeLocX, ProbeLocY) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
        ProbeUVPx = vec2(ProbeLocX, ProbeLocY);
    }else{
        // This is an adaptive probe
        uint AdaptiveProbeId = ProbeId - TotalUniformProbes;
        uvec2 AdaptiveProbeCoord = GetAdaptiveProbeCoord(AdaptiveProbeId);
        ProbeUV = vec2(AdaptiveProbeCoord) / vec2(PushConst.m_RTWidth, PushConst.m_RTHeight);
        ProbeUVPx = vec2(AdaptiveProbeCoord);
    }

    float ProbeLocDepthNDC = GetHizDepthPx(ProbeUVPx, 0);
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

    if(!kVisTracingHierarchy){
        if(ValidProbe){
            if(!kSkipScreenTrace&&SsgiTraceResult.z > 0.5){
                // screen hit
                vec2 HitUV = SsgiTraceResult.xy;
                vec3 HitRadiance = SampleTexture2D(PushConst.m_LastFrameFinalLightingSRV, sLinearClamp, HitUV).xyz;
                imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(HitRadiance, 1.0));
            }else{
                // screen hit miss
                imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(0.0));
                uint FailureRayId = atomicAdd(sFailureRayCount, 1);
                sFailureRayList[FailureRayId] = PackLocationAndRay(ProbeId, TraceRayCoord);
            }
        }else{
            imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(0.0));
        }

    }else{
        if(!ValidProbe){
            // probe is not valid, write the invalid color
            imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(1.0, 1.0, 1.0, 1.0));
            
        }else if(SsgiTraceResult.z < 0.5){
            // screen hit miss
            imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(1.0,1.0,1.0, 0.0));
            uint FailureRayId = atomicAdd(sFailureRayCount, 1);
            sFailureRayList[FailureRayId] = PackLocationAndRay(ProbeId, TraceRayCoord);
        }else{
            imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(1.0,0.0,0.0, 1.0));
        }
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