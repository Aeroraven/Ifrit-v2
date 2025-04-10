
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
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "Base.glsl"
#include "Bindless.glsl"
#include "Ayanami/Ayanami.Shared.glsl"
#include "Ayanami/Ayanami.SharedConst.h"


// TODO: It's better to make them group in a warp (32 for most NV devices). Now some threads are idle.
layout(local_size_x = 12, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 8, max_primitives = 12) out;

layout(location = 0) out  vec3 meshColor[];
RegisterUniform(BPerframe,{
    PerFramePerViewData m_Data;
});

RegisterStorage(BObjectCell,{
    uvec4 m_Cell[];
});

layout(push_constant) uniform UPushConstant{
    vec4 m_WorldBoundMin;
    vec4 m_WorldBoundMax;
    uint m_PerFrame;
    uint m_VoxelsPerWidth;
    uint m_ObjectGridId;
} PushConst;


uvec3 kTriangleLUT[12] = {
    uvec3(0, 1, 2), 
    uvec3(1, 3, 2),
    uvec3(5, 4, 7),
    uvec3(4, 6, 7),
    uvec3(4, 5, 0),
    uvec3(5, 1, 0),
    uvec3(4, 0, 6),
    uvec3(0, 2, 6),
    uvec3(1, 5, 3),
    uvec3(5, 7, 3),
    uvec3(2, 3, 6),
    uvec3(3, 7, 6)
};

vec3 colorLUT[6] = {
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 0.5, 0.5),
    vec3(1.0, 1.0, 1.0),
};
void main(){
    uint localThreadId = gl_LocalInvocationID.x;
    uvec3 CardIdx = uvec3(gl_WorkGroupID);
    vec3 CardUV = (vec3(CardIdx)) / vec3(PushConst.m_VoxelsPerWidth);
    vec3 CardUVNext = (vec3(CardIdx) + vec3(1.0)) / vec3(PushConst.m_VoxelsPerWidth);

    uint CellId = ifrit_ToCellId(CardIdx, uvec3(PushConst.m_VoxelsPerWidth));
    uint InvalidId = 0xFFFFFFFF;
    uvec4 CellData = GetResource(BObjectCell, PushConst.m_ObjectGridId).m_Cell[CellId];
    if(CellData.x == InvalidId && CellData.y == InvalidId && CellData.z == InvalidId && CellData.w == InvalidId){
        SetMeshOutputsEXT(0,0);
        return;
    }

    vec3 PosMin = mix(PushConst.m_WorldBoundMin.xyz, PushConst.m_WorldBoundMax.xyz, CardUV);
    vec3 PosMax = mix(PushConst.m_WorldBoundMin.xyz, PushConst.m_WorldBoundMax.xyz, CardUVNext);

    SetMeshOutputsEXT(8,12);

    mat4 WorldToScreen = GetResource(BPerframe, PushConst.m_PerFrame).m_Data.m_worldToClip;

    // Vertices
    if(localThreadId<8){
        uint lz = localThreadId & 1;
        uint ly = (localThreadId >> 1) & 1;
        uint lx = (localThreadId >> 2) & 1;

        vec3 interp = vec3(float(lx), float(ly), float(lz));
        vec3 pos = mix(PosMin, PosMax, interp);

        vec4 screenPos = WorldToScreen * vec4(pos, 1.0);
        gl_MeshVerticesEXT[localThreadId].gl_Position = screenPos;
        meshColor[localThreadId] = colorLUT[CellId % 6];
    }

    // Triangles 
    uvec3 tri = kTriangleLUT[localThreadId];
    gl_PrimitiveTriangleIndicesEXT[localThreadId] = uvec3(tri.x, tri.y, tri.z);
}