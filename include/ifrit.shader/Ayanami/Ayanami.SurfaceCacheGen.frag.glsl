
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

layout(location = 0) in vec2 TexCoord;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec4 Tangent;

layout(location = 0) out vec4 OutColor;
layout(location = 1) out vec4 OutNormal;

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
    vec4 albedo = texture(GetSampler2D(PushConst.albedoId), TexCoord);
    vec4 normal = texture(GetSampler2D(PushConst.NormalTexId), TexCoord);
    vec3 normalMap = normal.rgb * 2.0 - 1.0;

    vec3 tangent = normalize(Tangent.xyz);
    tangent = normalize(tangent - dot(tangent, Normal) * Normal);
    vec3 bitangent = cross(Normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, Normal);

    vec3 normalLocal = normalize(TBN * normalMap);
    normalLocal = normalLocal * 0.5 + 0.5;

    OutColor = albedo;
    OutNormal = vec4(normalLocal, normal.a);
}