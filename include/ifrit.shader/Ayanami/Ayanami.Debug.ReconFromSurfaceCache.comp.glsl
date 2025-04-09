
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

#define IFRIT_USE_INT64_IMAGE

#include "Base.glsl"
#include "Bindless.glsl"
#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"
#include "ComputeUtils.glsl"
#include "SamplerUtils.SharedConst.h"

layout(
    local_size_x = kAyanamiReconFromSCDepthTileSize, 
    local_size_y = kAyanamiReconFromSCDepthTileSize, 
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
    uint m_MeshDFDescId;
} PushConst;   

RegisterStorage(BAllCardData,{
    CardData m_Mats[];
});

RegisterUniform(BPerFrame,{
    PerFramePerViewData m_Data;
});

RegisterStorage(BMeshDFDesc,{
    MeshDFDesc data[];
});

RegisterStorage(BMeshDFMeta,{
    MeshDFMeta data;
});

RegisterUniform(BLocalTransform,{
    mat4 m_LocalToWorld;
    mat4 m_WorldToLocal;
    vec4 m_MaxScale;
});


void WritePixel(uvec2 AtlasUV, float Depth, uvec2 ScreenUV){
    // 32Bit depth | 16Bit AtlasX | 16Bit AtlasY 
    uint64_t DepthData = uint(Depth * 65535.0) << 16 | (AtlasUV.x << 16) | AtlasUV.y;
    imageAtomicMin(GetUAVImage2DR64UI(PushConst.m_AtomicDepthUAV), ivec2(ScreenUV), uint64_t(DepthData));
}

void main(){
    uint CardId = gl_GlobalInvocationID.z;
    uint TileX = gl_GlobalInvocationID.x;
    uint TileY = gl_GlobalInvocationID.y;

    uint CardsPerRow = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
    uint CardX = CardId % CardsPerRow;
    uint CardY = CardId / CardsPerRow;
    uvec2 TileCardOffset = uvec2(CardX, CardY) * PushConst.m_CardResolution + uvec2(TileX, TileY);

    vec2 AtlasUV = (vec2(TileCardOffset) + vec2(0.5)) / vec2(PushConst.m_CardAtlasResolution);
    float Depth = SampleTexture2D(PushConst.m_CardDepthAtlasSRV, sLinearClamp ,AtlasUV).r;
    if(Depth == 1.0) return;

    vec2 CardUV = (vec2(TileX, TileY) + vec2(0.5)) / vec2(PushConst.m_CardResolution);
    vec4 CardNDC  = vec4(CardUV * 2.0 - 1.0, Depth, 1.0);

    mat4 CardToLocal = GetResource(BAllCardData,PushConst.m_CardDataBufferId).m_Mats[CardId].m_VPInv;  //PushConst.m_Mats[CardId].m_VPInv;
    vec4 LocalPos = CardToLocal * CardNDC;
    if(LocalPos.w <= 0.0) return;
    LocalPos /= LocalPos.w;

    MeshDFDesc MDFDesc = GetResource(BMeshDFDesc, PushConst.m_MeshDFDescId).data[CardId/6];
    uint TransformID = MDFDesc.m_TransformId;
    mat4 LocalToWorld = GetResource(BLocalTransform, TransformID).m_LocalToWorld;
    mat4 WorldToScreen = GetResource(BPerFrame, PushConst.m_PerFrameDataId).m_Data.m_worldToClip;
    mat4 LocalToScreen  = WorldToScreen * LocalToWorld;

    vec4 ClipPos = LocalToScreen * LocalPos;
    if(ClipPos.w <= 0.0) return;
    ClipPos /= ClipPos.w;

    vec2 ScreenUV = ClipPos.xy * 0.5 + 0.5;
    if(ScreenUV.x < 0.0 || ScreenUV.x > 1.0 || ScreenUV.y < 0.0 || ScreenUV.y > 1.0) return;

    uvec2 ScreenUVi = uvec2(ScreenUV * vec2(PushConst.m_OutputWidth, PushConst.m_OutputHeight));

    WritePixel(uvec2(TileCardOffset.x, TileCardOffset.y), Depth, ScreenUVi);
}