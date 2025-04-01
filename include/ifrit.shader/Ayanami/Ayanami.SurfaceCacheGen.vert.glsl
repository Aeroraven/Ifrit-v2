
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
#include "Ayanami/Ayanami.Shared.glsl"

RegisterStorage(BAllCardData,{
    CardData m_Mats[];
});

layout(location = 0) out vec2 TexCoord;
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec4 Tangent;

layout(push_constant) uniform PushConstants {
    uint albedoId;
    uint NormalTexId;
    uint objectId;
    uint cardId;
    uint vertexId;
    uint uvId;
    uint AllCardDataId;
    uint TangentId;
    uint NormalBufId;
} PushConst;

void main(){
    uint inIndex = gl_VertexIndex;
    vec4 pos = ReadVertexPosition(PushConst.vertexId, inIndex);
    vec2 uv = ReadVertexUV(PushConst.uvId, inIndex);
    vec3 normal = ReadVertexNormal(PushConst.NormalBufId, inIndex).xyz;
    vec4 tangent = ReadVertexTangent(PushConst.TangentId, inIndex);
    mat4 m_VP = GetResource(BAllCardData, PushConst.AllCardDataId).m_Mats[PushConst.cardId].m_VP;

    vec4 worldPos = m_VP * pos;
    Normal = normalize(normal);
    Tangent = tangent;
    TexCoord = vec2(uv.x, uv.y);
    gl_Position = worldPos;
}