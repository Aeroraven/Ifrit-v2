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
    local_size_x = kAyanamiScrProbeMDFTraceKernelSize, 
    local_size_y = 1, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec4 m_WorldBoundMin;
    vec4 m_WorldBoundMax;
    vec4 m_CullGridSize;
    vec2 m_RayJitter;
    uint m_PerFrameCBV;
    uint m_RTWidth;
    uint m_RTHeight;
    uint m_ScreenProbeLightingAtlasUAV;
    uint m_MeshDFTraceProposalCounterUAV;
    uint m_MeshDFTraceProposalListUAV;
    uint m_AdaptiveProbesListUAV;
    uint m_GBufferDepthSRV;
    uint m_CullGridCounterUAV;
    uint m_CullGridListUAV;
    uint m_MeshDFDescListId;
    uint m_MaxMdfsPerGrid;
    uint m_GlobalDFTraceProposalCounterUAV;
    uint m_GlobalDFTraceProposalListUAV;
    uint m_NumMeshDFs;
}PushConst;

const float kRayProceedMax = 1000.0;
const float kRayProceedAdvance = 3e-2;
const uint kMaxTraceSteps = 60;
const float kMDFHitThreshold = 0.01;
const int kGridSearchRange = 1;
const bool kGridCulling = false;

struct TraceRayProposal{
    uvec2 m_TraceRayCoord;
    uint m_ProbeId;
};

shared uint sFailureRayCount;
shared uint sFailureRayGlobalStart;
shared uint sFailureRayList[kAyanamiScrProbeMDFTraceKernelSize];


RegisterStorage(BCullScatterOutput,{
    uint m_List[];
});

RegisterStorage(BCullScatterCounter,{
    uint m_Counter[];
}); 


RegisterStorage(BMeshDFTraceProposalIndirectArgs,{
    uint m_SsgiFailureRays;
    uint m_InvoX; // indirect args for mdf tracing for ss probes
    uint m_InvoY;
    uint m_InvoZ;
});

RegisterStorage(BGlobalDFTraceProposalIndirectArgs,{
    uint m_MdfFailureRays;
    uint m_InvoX; // indirect args for mdf tracing for ss probes
    uint m_InvoY;
    uint m_InvoZ;
});

RegisterStorage(BGloablDFTraceProposalList,{
    uint m_List[];
});

RegisterStorage(BMeshDFTraceProposalList,{
    uint m_List[];
});

RegisterStorage(BAdaptiveProbesList,{
    uint m_PackedCoords[];
});

uvec2 UnpackLocation(uint PackedCoords){
    uint y = PackedCoords & 0xFFFF;
    uint x = (PackedCoords >> 16) & 0xFFFF;
    return uvec2(x,y);
}

TraceRayProposal UnpackLocationAndRay(uint packedData){
    // low->high: (3bit rayId_x, 3bit rayId_y,  26bit probeId)
    uint RayIdX = packedData & 0x7; // 3bit
    uint RayIdY = (packedData >> 3) & 0x7; // 3bit
    uint ProbeId = (packedData >> 6) & 0x3FFFFFF; // 26bit
    return TraceRayProposal(uvec2(RayIdX, RayIdY), ProbeId);
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

vec3 MeshDFGridTraceSingleMDF(vec3 RayDirWS, vec3 RayOriginWS, uint MeshDFId, float CurHitTime){
    MeshDFMeta MDFMeta = AyaShared_GetMeshDFData(PushConst.m_MeshDFDescListId, MeshDFId);
    mat4 WorldToLocal = AyaShared_GetWorldToLocalMesh(PushConst.m_MeshDFDescListId, MeshDFId);
    vec3 MeshDFScale = AyaShared_GetMeshDFScale(PushConst.m_MeshDFDescListId, MeshDFId);
    vec2 MeshDFQuantScale = AyaShared_GetSdfQuantScale(MDFMeta);

    vec3 RayOriginLS = (WorldToLocal * vec4(RayOriginWS, 1.0)).xyz;
    vec3 RayProceedLS = (WorldToLocal * vec4((RayOriginWS + RayDirWS), 1.0)).xyz;
    vec3 RayDirLS = RayProceedLS - RayOriginLS;
    vec3 RayDirLSN = normalize(RayDirLS);

    vec3 BboxLB = MDFMeta.bboxMin.xyz;
    vec3 BboxRT = MDFMeta.bboxMax.xyz;
    uint SdfId = MDFMeta.sdfId;

    float TMaxCurHit = CurHitTime * length(RayDirLS);
    float T,TMax;
    bool IsHit = ifrit_RayboxIntersectionDual(RayOriginLS, RayDirLSN, BboxLB, BboxRT, T, TMax);

    T = max(T, 0.0);
    TMax = min(TMax, TMaxCurHit);

    vec3 HitPoint = RayOriginLS + RayDirLSN * T;
    bool IsFinalHit = false;
    if(IsHit){
        for(int i=0;i<kMaxTraceSteps;i++){
            vec3 UVW = (HitPoint - BboxLB) / (BboxRT - BboxLB);
            float Sdf = AyaShared_SampleMeshDF(SdfId, UVW, MeshDFQuantScale);
            float AbsSdf = abs(Sdf);
            if(Sdf<kMDFHitThreshold){
                IsFinalHit = true;
                break;
            }
            T += max(3e-4, AbsSdf * 0.2);
            if(T >= TMax){
                break;
            }
        }
    }
    float HitTimeWS = T / length(RayDirLS);
    if(IsFinalHit){
        return vec3(HitTimeWS, 0.0, 0.0);
    }else{
        return vec3(-1.0, -1.0, 0.0);
    }
}

void MeshDFGridTraceGrids(vec3 RayDirWS, vec3 RayOriginWS, uvec3 GridPos, inout int HitMeshDFId, inout float HitTime){

    uint GridId = ifrit_ToCellId(GridPos, uvec3(PushConst.m_CullGridSize.xyz));
    uint GridOffset = GridId * PushConst.m_MaxMdfsPerGrid;
    uint GridElements = GetResource(BCullScatterCounter, PushConst.m_CullGridCounterUAV).m_Counter[GridId];

    for(uint i = 0; i < GridElements; i++){
        uint MeshDFId = GetResource(BCullScatterOutput, PushConst.m_CullGridListUAV).m_List[GridOffset + i];
        vec3 HitResult = MeshDFGridTraceSingleMDF(RayDirWS, RayOriginWS, MeshDFId, HitTime);
        if(HitResult.x >= 0.0 && HitResult.x < HitTime){
            HitTime = HitResult.x;
            HitMeshDFId = int(MeshDFId);
        }
    }
}

vec4 MeshDFGridTrace(vec3 RayDirWS, vec3 RayOriginWS){
    int HitMeshDFId = -1;
    float HitTime = kRayProceedMax;

    // locate the tracing origin in the grid space
    vec3 RayOriginUV = (RayOriginWS - PushConst.m_WorldBoundMin.xyz) / (PushConst.m_WorldBoundMax.xyz - PushConst.m_WorldBoundMin.xyz);
    vec3 RayOriginGS = RayOriginUV * PushConst.m_CullGridSize.xyz;
    uvec3 RayOriginGSI = uvec3(RayOriginGS);

    ivec3 RayOriginGSIS = ivec3(RayOriginGSI);
    if(kGridCulling){
        for(int i = -kGridSearchRange; i <= kGridSearchRange; i++){
            for(int j = -kGridSearchRange; j <= kGridSearchRange; j++){
                for(int k = -kGridSearchRange; k <= kGridSearchRange; k++){
                    ivec3 GridPos = ivec3(RayOriginGSI) + ivec3(i, j, k);
                    if(GridPos.x >= 0 && GridPos.x < PushConst.m_CullGridSize.x &&
                    GridPos.y >= 0 && GridPos.y < PushConst.m_CullGridSize.y &&
                    GridPos.z >= 0 && GridPos.z < PushConst.m_CullGridSize.z){
                        MeshDFGridTraceGrids(RayDirWS, RayOriginWS, uvec3(GridPos), HitMeshDFId, HitTime);
                    }
                }
            }
        }
    }else{
        for(uint i = 0; i < PushConst.m_NumMeshDFs; i++){
            vec3 HitResult = MeshDFGridTraceSingleMDF(RayDirWS, RayOriginWS, i, HitTime);
            if(HitResult.x >= 0.0 && HitResult.x < HitTime){
                HitTime = HitResult.x;
                HitMeshDFId = int(i);
            }
        }
    }
    


    if(HitMeshDFId == -1){
        return vec4(RayOriginWS, 0.0);
    }else{
        return vec4(RayOriginWS + RayDirWS * HitTime, 1.0);
    }
}


void main(){
    // prepare indirect args for global df tracing (TODO: move this into a separate kernel)
    if(ifrit_IsGlobalFirstThread()){
        GetResource(BGlobalDFTraceProposalIndirectArgs,PushConst.m_GlobalDFTraceProposalCounterUAV).m_InvoY = 1;
        GetResource(BGlobalDFTraceProposalIndirectArgs,PushConst.m_GlobalDFTraceProposalCounterUAV).m_InvoZ = 1;    
    }
    if(ifrit_IsFirstLane()){
        sFailureRayCount = 0;
    }
    barrier();

    // the main process for mdf tracing
    PerFramePerViewData PerFrame = AyaShared_GetPerFrameData(PushConst.m_PerFrameCBV);
    float ClipNear = PerFrame.m_cameraNear;
    float ClipFar = PerFrame.m_cameraFar;
    mat4 ClipToWorld = PerFrame.m_clipToWorld;

    uint TraceRayId = gl_GlobalInvocationID.x;
    uint TotalTraceRays = GetResource(BMeshDFTraceProposalIndirectArgs,PushConst.m_MeshDFTraceProposalCounterUAV).m_SsgiFailureRays;

    if(TraceRayId < TotalTraceRays) {
        uint TraceRayPackedData = GetResource(BMeshDFTraceProposalList,PushConst.m_MeshDFTraceProposalListUAV).m_List[TraceRayId];
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

        vec3 SampledRay = AyaShared_GetScreenProbeTraceCoord(TraceRay.m_TraceRayCoord,PushConst.m_RayJitter);

        ProbeLocWS += SampledRay * kRayProceedAdvance;
        uvec2 WritingSlot = GetProbeWritingSlot(TraceRay.m_ProbeId, ProbeCntPerX, TraceRay.m_TraceRayCoord);

        vec4 HitResult = MeshDFGridTrace(SampledRay, ProbeLocWS);

        if(!kVisTracingHierarchy){
            if(HitResult.w < 0.5){
                // mdf hit miss
                uint LocalFailureRayId = atomicAdd(sFailureRayCount, 1);
                sFailureRayList[LocalFailureRayId] = TraceRayPackedData;
            }
        }else{
            if(HitResult.w < 0.5){
                // mdf hit miss
                uint LocalFailureRayId = atomicAdd(sFailureRayCount, 1);
                sFailureRayList[LocalFailureRayId] = TraceRayPackedData;
            }else{
                imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAV), ivec2(WritingSlot), vec4(0.0,1.0,0.0, 1.0));
            }
        }
        
    }
    barrier();

    // prepare the proposals for global df tracing
    if(ifrit_IsFirstLane()){
        uint LocalFailureRayCount = sFailureRayCount;
        uint GlobalFailureRayCount = atomicAdd(GetResource(BGlobalDFTraceProposalIndirectArgs,PushConst.m_GlobalDFTraceProposalCounterUAV).m_MdfFailureRays, LocalFailureRayCount);
        sFailureRayGlobalStart = GlobalFailureRayCount;
        uint TotalFailureRays = GlobalFailureRayCount + LocalFailureRayCount;
        uint GlobalDFProposalTGs = ifrit_DivRoundUp(TotalFailureRays, kAyanamiScrProbeGDFTraceKernelSize);
        atomicMax(GetResource(BGlobalDFTraceProposalIndirectArgs,PushConst.m_GlobalDFTraceProposalCounterUAV).m_InvoX, GlobalDFProposalTGs);
    }
    barrier();  
    uint LocalId = gl_LocalInvocationID.x;
    if(LocalId < sFailureRayCount){
        uint GlobalFailureRayId = sFailureRayGlobalStart + LocalId;
        uint GlobalDFTraceProposalPackedData = sFailureRayList[LocalId];
        GetResource(BGloablDFTraceProposalList,PushConst.m_GlobalDFTraceProposalListUAV).m_List[GlobalFailureRayId] = GlobalDFTraceProposalPackedData;
    }
}