
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

RegisterStorage(BMeshDFDesc,{
    MeshDFDesc m_Data[];
});

RegisterStorage(BMeshDFMeta,{
    MeshDFMeta m_Data;
});

RegisterUniform(BLocalTransform,{
    mat4 m_LocalToWorld;
    mat4 m_WorldToLocal;
    vec4 m_MaxScale;
});

layout(push_constant) uniform UPushConstant{
    mat4 m_VP;
    uint m_NumMeshDF;
    uint m_MeshDFDescListId;
    uint m_NumTilesWidth;
    uint m_TileAtomics;
    uint m_ScatterOutput;
} PushConst;

// TODO: It's better to make them group in a warp (32 for most NV devices). Now some threads are idle.
layout(local_size_x = 12, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 8, max_primitives = 12) out;

layout(location = 0) out flat uint meshIdX[];

uvec3 kTriangleLUT[12] = {
    uvec3(0, 1, 2), 
    uvec3(0, 2, 3),
    uvec3(4, 5, 6),
    uvec3(4, 6, 7),
    uvec3(0, 1, 5),
    uvec3(0, 5, 4),
    uvec3(1, 2, 6),
    uvec3(1, 6, 5),
    uvec3(2, 3, 7),
    uvec3(2, 7, 6),
    uvec3(3, 0, 4),
    uvec3(3, 4, 7)
};
void main(){
    uint localThreadId = gl_LocalInvocationID.x;
    uint meshId = gl_WorkGroupID.x;

    MeshDFDesc mdfDesc = GetResource(BMeshDFDesc, PushConst.m_MeshDFDescListId).m_Data[meshId];
    MeshDFMeta mdfMeta = GetResource(BMeshDFMeta, mdfDesc.m_MdfMetaId).m_Data;
    vec4 bboxMin = mdfMeta.bboxMin;
    vec4 bboxMax = mdfMeta.bboxMax;
    mat4 localToWorld = GetResource(BLocalTransform, mdfDesc.m_TransformId).m_LocalToWorld;

    mat4 localToScr = PushConst.m_VP * localToWorld;

    SetMeshOutputsEXT(8,12);

    // Vertices
    if(localThreadId<8){
        uint lz = localThreadId & 1;
        uint ly = (localThreadId >> 1) & 1;
        uint lx = (localThreadId >> 2) & 1;

        vec3 interp = vec3(float(lx), float(ly), float(lz));
        vec3 pos = mix(bboxMin.xyz, bboxMax.xyz, interp);

        vec4 screenPos = localToScr * vec4(pos, 1.0);
        gl_MeshVerticesEXT[localThreadId].gl_Position = screenPos;
        meshIdX[localThreadId] = meshId;
    }

    // Triangles 
    uvec3 tri = kTriangleLUT[localThreadId];
    gl_PrimitiveTriangleIndicesEXT[localThreadId] = uvec3(tri.x, tri.y, tri.z);
}