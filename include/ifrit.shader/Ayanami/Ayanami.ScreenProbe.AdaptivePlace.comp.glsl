
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
    local_size_x = kAyanamiScrProbeAdaptivePlaceKernelSize, 
    local_size_y = kAyanamiScrProbeAdaptivePlaceKernelSize, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    vec2 m_CoordJitter;
    uint m_PerFrameCBV;
    uint m_ScrNormalCombSRV;
    uint m_ScrDepthCombSRV;
    uint m_AdaptiveProbesCounterUAV;
    uint m_AdaptiveProbesListUAV;
    uint m_DownSampleSize;
    uint m_RTWidth;
    uint m_RTHeight;
    uint m_MaxAdaptiveProbes;
}PushConst;

shared uint sAdaptiveSampleCount;
shared uint sAdaptiveSampleGlobalStart;
shared uint sPackedAdaptiveSampleCoords[kAyanamiScrProbeAdaptivePlaceKernelSize * kAyanamiScrProbeAdaptivePlaceKernelSize];

struct ScreenSpaceSample{
    vec3 m_WorldPos;
    vec3 m_WorldNormal;
    float m_SceneDepth;
    bool m_Valid;
};

RegisterStorage(BAdaptiveProbesCounter,{
    uint m_Counter;
    uint m_InvoX; // below two are used for debugging only
    uint m_InvoY; 
    uint m_InvoZ;
    uint m_InvoTraceX; // used for screen space tracing
    uint m_InvoTraceY;
    uint m_InvoTraceZ;
    uint m_InvoGatherX; // used for screen space tracing
    uint m_InvoGatherY;
    uint m_InvoGatherZ;
});

RegisterStorage(BAdaptiveProbesList,{
    uint m_PackedCoords[];
});

uint AllocateGlobalAdaptiveProbeList(uint Count){
    uint GlobalStart = atomicAdd(GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_Counter, Count);
    return GlobalStart;
}

void MaximizeIndirectArgsX(uint Ref,uint Ref2,uint Ref3){
    atomicMax(GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_InvoX, Ref);
    atomicMax(GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_InvoTraceX, Ref2);
    atomicMax(GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_InvoGatherX, Ref3);
}

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

    vec4 FinalWeights = vec4(1.0);
    FinalWeights *= DepthWeights * ProbeDepthValid;

    return FinalWeights;
}

uint PackLocation(uvec2 Location){
    uint PackedLocation = 0;
    PackedLocation |= (Location.x & 0xFFFF) << 16;
    PackedLocation |= (Location.y & 0xFFFF);
    return PackedLocation;
}

void main(){

    // Get num uniform probes
    uint ProbeCntPerX = ifrit_DivRoundUp(PushConst.m_RTWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint ProbeCntPerY = ifrit_DivRoundUp(PushConst.m_RTHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint TotalUniformProbes = ProbeCntPerX * ProbeCntPerY;

    // Start the main kernel
    uvec2 DispatchCoord =  uvec2(gl_GlobalInvocationID.xy);
    uint LocalId = gl_LocalInvocationIndex.x;
    uvec2 ScreenCoord = DispatchCoord * PushConst.m_DownSampleSize + uvec2(PushConst.m_CoordJitter * PushConst.m_DownSampleSize);
    bool ValidSample = true;
    if(ScreenCoord.x >= PushConst.m_RTWidth || ScreenCoord.y >= PushConst.m_RTHeight){
        ValidSample = false;
    }

    if(ifrit_IsFirstLane()){
        sAdaptiveSampleCount = 0;
    }
    if(ifrit_IsGlobalFirstThread()){
        GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_InvoY = 1;
        GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_InvoZ = 1;
        GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_InvoTraceY = 1;
        GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_InvoTraceZ = 1;
        GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_InvoGatherY = 1;
        GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_InvoGatherZ = 1;
    }
    groupMemoryBarrier();
    barrier();
    

    ScreenSpaceSample Sample = GetScreenSample(ScreenCoord);
    if(!Sample.m_Valid){
        ValidSample = false;
    }
    if(ValidSample){
        vec4 Weights = GetNeighbourProbeWeights(ScreenCoord, Sample);
        for(uint i=0;i<4;i++){
            if(Weights[i] < 0.01){
                // here, this location is not covered by near probes.
                uint PackedLocation = PackLocation(ScreenCoord);
                uint LocalAdaptiveListIdx = atomicAdd(sAdaptiveSampleCount, 1);
                sPackedAdaptiveSampleCoords[LocalAdaptiveListIdx] = PackedLocation;
                break;
            }
        }
    }
    groupMemoryBarrier();
    barrier();

    if(ifrit_IsFirstLane()){
        sAdaptiveSampleGlobalStart = AllocateGlobalAdaptiveProbeList(sAdaptiveSampleCount);
        uint TotalProbes = sAdaptiveSampleGlobalStart + sAdaptiveSampleCount;
        uint RequiredIndirectX = ifrit_DivRoundUp(TotalProbes, kAyanamiScrProbeAdaptiveGroupKernelSize);
        uint RequiredGatherTGX = ifrit_DivRoundUp(TotalProbes+TotalUniformProbes, kAyanamiScrProbeIntegrateSHKernelSize);
        MaximizeIndirectArgsX(RequiredIndirectX,TotalProbes+TotalUniformProbes,RequiredGatherTGX);
    }
    groupMemoryBarrier();
    barrier();

    uint GlobalStart = sAdaptiveSampleGlobalStart;
    uint LocalAddedProbes = sAdaptiveSampleCount;

    if(LocalId < LocalAddedProbes){
        uint GlobalIdx = GlobalStart + LocalId;
        uint PackedLocation = sPackedAdaptiveSampleCoords[LocalId];
        if(GlobalIdx < PushConst.m_MaxAdaptiveProbes){
            // Write to the global adaptive probe list
            GetResource(BAdaptiveProbesList,PushConst.m_AdaptiveProbesListUAV).m_PackedCoords[GlobalIdx] = PackedLocation;
        }
    }


}


