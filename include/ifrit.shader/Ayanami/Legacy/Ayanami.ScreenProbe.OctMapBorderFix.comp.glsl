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
    uint m_AdaptiveProbesCounterUAV;
    uint m_RTWidth;
    uint m_RTHeight;
    uint m_ScreenProbeLightingAtlasUAVIn;
    uint m_ScreenProbeLightingAtlasUAVOut;
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


uvec2 GetProbeWritingSlotReadIn(uint ProbeId, uint ProbeCntPerX, uvec2 TraceRayCoord){
    uint ProbeX = ProbeId % ProbeCntPerX;
    uint ProbeY = ProbeId / ProbeCntPerX;
    uint ProbeLocX = ProbeX * kAyanami_ScreenProbeProbeHemiRes;
    uint ProbeLocY = ProbeY * kAyanami_ScreenProbeProbeHemiRes;
    uvec2 ProbeLoc = uvec2(ProbeLocX, ProbeLocY);
    uvec2 WritingSlot = ProbeLoc + TraceRayCoord;
    return WritingSlot; 
}

uvec2 GetProbeWritingSlotWriteTo(uint ProbeId, uint ProbeCntPerX, uvec2 TraceRayCoord){
    uint ProbeX = ProbeId % ProbeCntPerX;
    uint ProbeY = ProbeId / ProbeCntPerX;
    uint ProbeLocX = ProbeX * (kAyanami_ScreenProbeProbeHemiRes+2);
    uint ProbeLocY = ProbeY * (kAyanami_ScreenProbeProbeHemiRes+2);
    uvec2 ProbeLoc = uvec2(ProbeLocX, ProbeLocY);
    uvec2 WritingSlot = ProbeLoc + TraceRayCoord;
    return WritingSlot; 
}

// The memory access pattern should be optimized

void main(){
    uint ProbeCntPerX = ifrit_DivRoundUp(PushConst.m_RTWidth, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint ProbeCntPerY = ifrit_DivRoundUp(PushConst.m_RTHeight, kAyanami_ScreenProbeUniformPlaceTileWidth);
    uint TotalUniformProbes = ProbeCntPerX * ProbeCntPerY;
    uint TotalProbes = GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_Counter+ TotalUniformProbes;

    uint ProbeId = gl_GlobalInvocationID.x;
    if(ProbeId >= TotalProbes) return;

    for(uint i=0;i<kAyanami_ScreenProbeProbeHemiRes;i++){
        for(uint j=0;j<kAyanami_ScreenProbeProbeHemiRes;j++){
            uvec2 TraceRayCoord = uvec2(i,j);
            uvec2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, TraceRayCoord);
            uvec2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, TraceRayCoord+uvec2(1,1));
            vec3 Radiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVIn), ivec2(WritingSlotIn)).xyz;
            imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVOut), ivec2(WritingSlotOut), vec4(Radiance, 1.0));
        }
    }

    // Fixing Borders
    uint LocX=0;
    uint LocY=0;

    // top
    for(LocX=1;LocX<=kAyanami_ScreenProbeProbeHemiRes;LocX++){
        // Output:(LocX,0) <= Input:(N-LocX,0)
        uvec2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uvec2(kAyanami_ScreenProbeProbeHemiRes-LocX,0));
        uvec2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uvec2(LocX,0));
        vec3 Radiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVIn), ivec2(WritingSlotIn)).xyz;
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVOut), ivec2(WritingSlotOut), vec4(Radiance, 1.0));
    }
    // left
    for(LocY=1;LocY<=kAyanami_ScreenProbeProbeHemiRes;LocY++){
        // Output:(0,LocY) <= Input:(0,N-LocY)
        uvec2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uvec2(0,kAyanami_ScreenProbeProbeHemiRes-LocY));
        uvec2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uvec2(0,LocY));
        vec3 Radiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVIn), ivec2(WritingSlotIn)).xyz;
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVOut), ivec2(WritingSlotOut), vec4(Radiance, 1.0));
    }
    // bottom
    for(LocX=1;LocX<=kAyanami_ScreenProbeProbeHemiRes;LocX++){
        // Output:(LocX,N+1) <= Input:(N-LocX,N+1)
        uvec2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uvec2(kAyanami_ScreenProbeProbeHemiRes-LocX,kAyanami_ScreenProbeProbeHemiRes-1));
        uvec2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uvec2(LocX,kAyanami_ScreenProbeProbeHemiRes+1));
        vec3 Radiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVIn), ivec2(WritingSlotIn)).xyz;
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVOut), ivec2(WritingSlotOut), vec4(Radiance, 1.0));
    }
    // right
    for(LocY=1;LocY<=kAyanami_ScreenProbeProbeHemiRes;LocY++){
        // Output:(N+1,LocY) <= Input:(N+1,N-LocY)
        uvec2 WritingSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uvec2(kAyanami_ScreenProbeProbeHemiRes-1,kAyanami_ScreenProbeProbeHemiRes-LocY));
        uvec2 WritingSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uvec2(kAyanami_ScreenProbeProbeHemiRes+1,LocY));
        vec3 Radiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVIn), ivec2(WritingSlotIn)).xyz;
        imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVOut), ivec2(WritingSlotOut), vec4(Radiance, 1.0));
    }
    // top-left
    uvec2 CornerSlotIn;
    uvec2 CornerSlotOut;
    vec3 CornerRadiance;

    CornerSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uvec2(0,0));
    CornerSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uvec2(kAyanami_ScreenProbeProbeHemiRes+1,kAyanami_ScreenProbeProbeHemiRes+1));
    CornerRadiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVIn), ivec2(CornerSlotIn)).xyz;
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVOut), ivec2(CornerSlotOut), vec4(CornerRadiance, 1.0));

    // top-right
    CornerSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uvec2(kAyanami_ScreenProbeProbeHemiRes-1,0));
    CornerSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uvec2(0,kAyanami_ScreenProbeProbeHemiRes+1));
    CornerRadiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVIn), ivec2(CornerSlotIn)).xyz;
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVOut), ivec2(CornerSlotOut), vec4(CornerRadiance, 1.0));

    // bottom-left
    CornerSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uvec2(0,kAyanami_ScreenProbeProbeHemiRes-1));
    CornerSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uvec2(kAyanami_ScreenProbeProbeHemiRes+1,0));
    CornerRadiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVIn), ivec2(CornerSlotIn)).xyz;
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVOut), ivec2(CornerSlotOut), vec4(CornerRadiance, 1.0));

    // bottom-right
    CornerSlotIn = GetProbeWritingSlotReadIn(ProbeId, ProbeCntPerX, uvec2(kAyanami_ScreenProbeProbeHemiRes-1,kAyanami_ScreenProbeProbeHemiRes-1));
    CornerSlotOut = GetProbeWritingSlotWriteTo(ProbeId, ProbeCntPerX, uvec2(0,0));
    CornerRadiance = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVIn), ivec2(CornerSlotIn)).xyz;
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_ScreenProbeLightingAtlasUAVOut), ivec2(CornerSlotOut), vec4(CornerRadiance, 1.0));
}