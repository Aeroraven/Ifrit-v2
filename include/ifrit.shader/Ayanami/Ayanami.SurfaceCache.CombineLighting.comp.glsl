
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
    local_size_x = kAyanamiSCDirectLightCardSizePerBlock, 
    local_size_y = kAyanamiSCDirectLightCardSizePerBlock, 
    local_size_z = kAyanamiSCDirectLightObjectsPerBlock 
) in;
 
layout(push_constant)  uniform PushConstData{
    uint m_FrameIdx; // clamped to max history !!!
    uint m_DirectLightingAtlasSRV;
    uint m_IndirectLightingAtlasSRV;
    uint m_AlbedoAtlasSRV;
    uint m_FinalLightingAtlasUAV;
    uint m_CardResolution;
    uint m_CardAtlasResolution;
} PushConst;

void main(){
    uvec3 tID = gl_GlobalInvocationID;
    uint MaxCardsInLine = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
    uint CardIndex_X = tID.z % MaxCardsInLine;
    uint CardIndex_Y = tID.z / MaxCardsInLine;
    uvec2 CardOffset = uvec2(CardIndex_X * PushConst.m_CardResolution, CardIndex_Y * PushConst.m_CardResolution);
    uvec2 TileOffset = uvec2(tID.x, tID.y);
    uvec2 OverallOffset = CardOffset + TileOffset;

    vec2 AtlasUV = (vec2(OverallOffset) + vec2(0.5)) / vec2(PushConst.m_CardAtlasResolution);

    vec4 DirectLighting = SampleTexture2D(PushConst.m_DirectLightingAtlasSRV, sNearestClamp,AtlasUV);
    vec4 IndirectLighting = SampleTexture2D(PushConst.m_IndirectLightingAtlasSRV, sNearestClamp,AtlasUV);
    vec4 Albedo = SampleTexture2D(PushConst.m_AlbedoAtlasSRV, sNearestClamp,AtlasUV);

    vec4 DiffuseLambertBRDF = Albedo / kPI;
    vec4 FinalLighting = (DirectLighting + IndirectLighting) * DiffuseLambertBRDF;

    // temporal accumulation
    uint NumHistoryFrames = (PushConst.m_FrameIdx == 0) ? 0 : PushConst.m_FrameIdx - 1;
    vec4 PreviousLighting = imageLoad(GetUAVImage2DR32F(PushConst.m_FinalLightingAtlasUAV), ivec2(OverallOffset));
    vec4 MixedLighting = mix(PreviousLighting, FinalLighting, 1.0 / float(NumHistoryFrames + 1));

    imageStore(GetUAVImage2DR32F(PushConst.m_FinalLightingAtlasUAV), ivec2(OverallOffset), vec4(FinalLighting.rgb, 1.0));
}