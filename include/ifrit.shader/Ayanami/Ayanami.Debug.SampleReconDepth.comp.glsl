
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
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_atomic_float : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_shader_image_int64 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#define IFRIT_USE_INT64_IMAGE 1

#include "Base.glsl"
#include "Bindless.glsl"
#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"

layout(
    local_size_x = kAyanamiReconFromSCTileSize, 
    local_size_y = kAyanamiReconFromSCTileSize, 
    local_size_z = 1 
) in;

layout(push_constant) uniform UPushConst
{
    uint m_NumTotalCards;
    uint m_CardResolution;
    uint m_CardAtlasResolution;
    uint m_PerFrameDataId;
    uint m_CardDataBufferId;
    uint m_OutputUAV;
    uint m_CardAlbedoAtlasSRV;
    uint m_CardNormalAtlasSRV;
    uint m_CardRadianceAtlasSRV;
    uint m_CardDepthAtlasSRV;
    uint m_AtomicDepthUAV;
    uint m_OutputWidth;
    uint m_OutputHeight;
} PushConst;   

RegisterStorage(BAllCardData,{
    CardData m_Mats[];
});

RegisterUniform(BPerFrame,{
    PerFramePerViewData m_Data;
});

void main(){
    uvec2 ScreenUV = gl_GlobalInvocationID.xy;
    if(ScreenUV.x >= PushConst.m_OutputWidth || ScreenUV.y >= PushConst.m_OutputHeight) return;

    uint64_t DepthValue;
    DepthValue = imageLoad(GetUAVImage2DR64UI(PushConst.m_AtomicDepthUAV), ivec2(ScreenUV)).r;

    uint64_t DepthBits = DepthValue >> 32;
    uint U32Max = uint(1) << 32;
    float Depth = float(DepthBits) / float(U32Max);
    if(Depth>= 0.99) {
        imageStore(GetUAVImage2DR32F(PushConst.m_OutputUAV), ivec2(ScreenUV), vec4(0.0,0.0,0.0, 1.0));
        return;
    }

    uvec2 AtlasUV = uvec2(DepthValue & 0xFFFF, (DepthValue >> 16) & 0xFFFF).yx;
    vec2 AltasUVf = (vec2(AtlasUV)+0.5) / float(PushConst.m_CardAtlasResolution);
    if(AltasUVf.x >= 1.0 || AltasUVf.y >= 1.0) return;
    vec3 Albedo = SampleTexture2D(PushConst.m_CardAlbedoAtlasSRV, sLinearClamp, AltasUVf).xyz;
    imageStore(GetUAVImage2DR32F(PushConst.m_OutputUAV), ivec2(ScreenUV), vec4(Albedo, 1.0));
}