
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
#include "SamplerUtils.SharedConst.h"
#include "Math.SphericalHarmonics.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

layout(push_constant) uniform UPushConstant{
    uint m_MeshDFDescListId;
    uint m_PerFrameId;
    uint m_TotalMdfCount;
    uint m_GridVpUAV;
    uint m_MaxMdfsPerGrid;
    uint m_NumGridsPerSlice;
    uint m_ScatterCounterUAV;
    uint m_ScatterOutputUAV;
    uint m_NumTilesWidth;
} PushConst;

RegisterStorage(BGridMat,{
    mat4 m_VP[];
});

layout(location = 0) out flat uint vInstanceId;
layout(location = 1) out flat uint vZSlice;

void main(){
    uint VertexId = gl_VertexIndex;
    uint InstanceId = gl_InstanceIndex;

    uint ZSlice = InstanceId / PushConst.m_TotalMdfCount;
    InstanceId = InstanceId % PushConst.m_TotalMdfCount;

    MeshDFMeta MDFData = AyaShared_GetMeshDFData(PushConst.m_MeshDFDescListId, InstanceId);
    vec3 BboxMin = MDFData.bboxMin.xyz - vec3(1.0);
    vec3 BboxMax = MDFData.bboxMax.xyz + vec3(1.0);

    vec3 LerpValue = vec3(0.0, 0.0, 0.0);
    LerpValue.x = mix(BboxMin.x, BboxMax.x, float(VertexId & 1u));
    LerpValue.y = mix(BboxMin.y, BboxMax.y, float((VertexId >> 1u) & 1u));
    LerpValue.z = mix(BboxMin.z, BboxMax.z, float((VertexId >> 2u) & 1u));

    mat4 LocalToWorld = AyaShared_GetLocalToWorld(PushConst.m_MeshDFDescListId, InstanceId);
    mat4 WorldToClip = GetResource(BGridMat, PushConst.m_GridVpUAV).m_VP[ZSlice];
    mat4 LocalToClip = WorldToClip * LocalToWorld;

    vec4 ClipPos = LocalToClip * vec4(LerpValue, 1.0);
    gl_Position = ClipPos;
    vInstanceId = InstanceId;
    vZSlice = ZSlice;
}
