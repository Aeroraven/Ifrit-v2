
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
#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"
#include "Random/Random.WNoise2D.glsl"
#include "SamplerUtils.SharedConst.h"

layout(
    local_size_x = kAyanamiSCDirectLightCardSizePerBlock, 
    local_size_y = kAyanamiSCDirectLightCardSizePerBlock, 
    local_size_z = kAyanamiSCDirectLightObjectsPerBlock 
    ) in;
 
layout(push_constant)  uniform PushConstData{
    vec4 m_LightDir;
    uint m_DirectLightingUAV;
    uint m_NormalAtlasSRV;
    uint m_ShadowMaskSRV;
    uint m_CardResolution;
    uint m_CardAtlasResolution;
    uint m_MeshDFDescListId;
    
} PushConst;

RegisterStorage(BAllCardData,{
    CardData m_Mats[];
});

RegisterStorage(BAllWorldData,{
    uint m_TransformId[];
});

RegisterUniform(BLocalTransform,{
    mat4 m_LocalToWorld;
    mat4 m_WorldToLocal;
    float m_MaxScale;
});

RegisterUniform(BPerFrameData,{
    PerFramePerViewData m_Data;
});

RegisterStorage(BMeshDFDesc,{
    MeshDFDesc m_Data[];
});

RegisterStorage(BMeshDFMeta,{
    MeshDFMeta m_Data;
});

void main(){
    uvec3 tID = gl_GlobalInvocationID;
    uint MaxCardsInLine = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
    uint CardIndex_X = tID.z % MaxCardsInLine;
    uint CardIndex_Y = tID.z / MaxCardsInLine;
    uvec2 CardOffset = uvec2(CardIndex_X * PushConst.m_CardResolution, CardIndex_Y * PushConst.m_CardResolution);
    uvec2 TileOffset = uvec2(tID.x, tID.y);
    uvec2 OverallOffset = CardOffset + TileOffset;


    vec2 AtlasUV = (vec2(OverallOffset) + vec2(0.5)) / vec2(PushConst.m_CardAtlasResolution);
    vec2 NormalRG = SampleTexture2D(PushConst.m_NormalAtlasSRV, sLinearClamp, AtlasUV).rg * 2.0 - 1.0;
    vec3 NormalLocal = vec3(NormalRG, sqrt(1.0 - dot(NormalRG, NormalRG)));

    uint MeshId = tID.z / 6;
    MeshDFDesc MdfDesc = GetResource(BMeshDFDesc, PushConst.m_MeshDFDescListId).m_Data[MeshId];
    uint TransformId = MdfDesc.m_TransformId;
    mat4 LocalToWorld = GetResource(BLocalTransform, TransformId).m_LocalToWorld;

    vec3 NormalWorld = normalize(LocalToWorld * vec4(NormalLocal, 0.0)).xyz; // Warn: this is not a normal transform, but here we use equal scales
    vec3 LightDir = normalize(PushConst.m_LightDir.xyz);
    
    float Irradiance = max(dot(NormalWorld, -LightDir), 0.0);

    float ShadowVisibility = SampleTexture2D(PushConst.m_ShadowMaskSRV, sLinearClamp, AtlasUV).r;
    Irradiance *= ShadowVisibility;

    vec3 Color = vec3(1.0, 1.0, 1.0) * Irradiance;

    imageStore(GetUAVImage2DR32F(PushConst.m_DirectLightingUAV), ivec2(OverallOffset), vec4(Color, 1.0));
}