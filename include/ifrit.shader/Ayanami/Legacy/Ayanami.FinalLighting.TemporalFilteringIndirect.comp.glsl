
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
#include "Random/Random.WNoise2D.glsl"
#include "SamplerUtils.SharedConst.h"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

layout(
    local_size_x = kAyanamiFinalTemporalReprojKernelSizeX, 
    local_size_y = kAyanamiFinalTemporalReprojKernelSizeY, 
    local_size_z = 1
) in;

layout(push_constant) uniform UPushConst{
    uint m_FrameIdx;
    uint m_CurrentFrameIndirectLightingUAV;
    uint m_HistoryIndirectLightingUAV;
    uint m_RtWidth;
    uint m_RtHeight;
}PushConst;

void main(){
    uvec2 tID = uvec2(gl_GlobalInvocationID.xy);
    if(tID.x >= PushConst.m_RtWidth || tID.y >= PushConst.m_RtHeight){
        return;
    }
    vec4 HistoryColor = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_HistoryIndirectLightingUAV), ivec2(tID));
    vec4 CurrentColor = imageLoad(GetUAVImage2DRGBA32F(PushConst.m_CurrentFrameIndirectLightingUAV), ivec2(tID));

    // if nan set to zero
    uint NumHistoryFrames = PushConst.m_FrameIdx;
    if(NumHistoryFrames > 0){
        NumHistoryFrames = NumHistoryFrames - 1;
    }
    vec4 MixedColor = mix(HistoryColor, CurrentColor, 1.0 / float(NumHistoryFrames + 1));
    imageStore(GetUAVImage2DRGBA32F(PushConst.m_CurrentFrameIndirectLightingUAV), ivec2(tID), MixedColor);
}
