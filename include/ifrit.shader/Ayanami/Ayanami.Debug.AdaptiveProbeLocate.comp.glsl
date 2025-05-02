
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
    local_size_x = kAyanamiScrProbeAdaptiveGroupKernelSize, 
    local_size_y = 1, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst{
    uint m_AdaptiveProbesCounterUAV;
    uint m_AdaptiveProbesListUAV;
    uint m_OutputTextureUAV;
} PushConst;

RegisterStorage(BAdaptiveProbesCounter,{
    uint m_Counter;
    uint m_InvoX; // below two are used for debugging only
    uint m_InvoY; 
    uint m_InvoZ;
});

RegisterStorage(BAdaptiveProbesList,{
    uint m_PackedCoords[];
});

uvec2 UnpackLocation(uint PackedCoords){
    uint y = PackedCoords & 0xFFFF;
    uint x = (PackedCoords >> 16) & 0xFFFF;
    return uvec2(x,y);
}

void main(){
    uint ProbeId = gl_GlobalInvocationID.x;
    uint TotalProbes = GetResource(BAdaptiveProbesCounter,PushConst.m_AdaptiveProbesCounterUAV).m_Counter;
    if(ProbeId >= TotalProbes) return;
    uint PackedCoords = GetResource(BAdaptiveProbesList,PushConst.m_AdaptiveProbesListUAV).m_PackedCoords[ProbeId];
    uvec2 Coords = UnpackLocation(PackedCoords);
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_OutputTextureUAV), ivec2(Coords), vec4(0.0,1.0,0.0,1.0));
}