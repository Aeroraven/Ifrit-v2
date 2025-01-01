
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

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
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

layout(local_size_x = cMeshRasterizeThreadGroupSizeX, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = cMeshRasterizeMaxVertexSize, max_primitives = cMeshRasterizeMaxPrimitiveSize) out;

#if SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL
// According to NVIDIA's documentation, AS's payload should be small
// to avoid perf penalty.
// https://developer.nvidia.com/blog/advanced-api-performance-mesh-shaders/
struct TaskSharedData{
    uint base;
    uint subIds[cMeshRasterizeTaskPayloadSize];
};

taskPayloadSharedEXT  TaskSharedData taskSharedData;
#endif

RegisterStorage(bMeshlet,{
    Meshlet data[];
});

RegisterStorage(bMeshletVertices,{
    uint data[];
});

RegisterStorage(bMeshletTriangles,{
    uint data[];
});

RegisterStorage(bVertices,{
    vec4 data[];
});

RegisterStorage(bFilteredMeshlets2,{
    uvec2 data[];
});


layout(binding = 0, set = 1) uniform PerframeViewData{
    uint refCurFrame;
    uint refPrevFrame;
}uPerframeView;

layout(binding = 0, set = 2) uniform InstanceData{
    uvec4 ref;
}uInstanceData;
layout(binding = 0, set = 3) uniform IndirectDrawData2{
    uint allMeshletsRef;
    uint allMeshletsRefSW;
    uint indDrawCmdRef;
}uIndirectDrawData2;

layout(push_constant) uniform CullingPass{
    uint passNo;
} pConst;

layout(location = 0) out flat uint ids[];

// color maps, 24 colors
vec4 colorMap[8] = vec4[8](
  vec4(1.0, 0.0, 0.0, 1.0),
  vec4(0.0, 1.0, 0.0, 1.0),
  vec4(0.0, 0.0, 1.0, 1.0),
  vec4(1.0, 1.0, 0.0, 1.0),
  vec4(1.0, 0.0, 1.0, 1.0),
  vec4(0.0, 1.0, 1.0, 1.0),
  vec4(1.0, 1.0, 1.0, 1.0),
  vec4(0.5, 0.0, 0.0, 1.0)
);

#if SYARO_SHADER_SHARED_VISBUFFER_ENABLE_PRIMCULL
shared vec2 sPositionsXY[cMeshRasterizeMaxVertexSize];
shared float sPositionsW[cMeshRasterizeMaxVertexSize];
#endif

bool isSecondCullingPass(){
    return pConst.passNo == 1;
}

uint getAcutalWorkGroup(){
#if SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL
    return gl_WorkGroupID.x;
#else
    uint inTaskWorkGroup = gl_WorkGroupID.x;
    uint slotId = inTaskWorkGroup / 4;
    uint slotOffset = (inTaskWorkGroup % 4) * 8;
    uint localInvo = (taskSharedData.subIds[slotId] >> slotOffset) & 0xFF;
    return localInvo + taskSharedData.base;
#endif
}
uint getClusterID(){
    uint actualWorkGroup = getAcutalWorkGroup();
    if(!isSecondCullingPass()){
        return actualWorkGroup;
    }else{
        return GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1 + actualWorkGroup;
    }
}
uint getObjId(){
    uint actualWorkGroup = getAcutalWorkGroup();
    if(!isSecondCullingPass()){
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[actualWorkGroup].x;
    }else{
        uint baseOffset = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1;
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[baseOffset + actualWorkGroup].x;
    }
}

uint getMeshletId(){
    uint actualWorkGroup = getAcutalWorkGroup();
    if(!isSecondCullingPass()){
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[actualWorkGroup].y;
    }else{
        uint baseOffset = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1;
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[baseOffset + actualWorkGroup].y;
    }
}

uvec2 readTriangleIndexRefOffset(uint meshletid, uint meshletRef, uint obj){
    uint meshletTriOffset = GetResource(bMeshlet,meshletRef).data[meshletid].triangle_offset;
    uint indexRef = GetResource(bMeshDataRef,obj).meshletIndexBuffer;
    return uvec2(indexRef,meshletTriOffset);
}

uint readTriangleIndex2(uvec2 refOffset, uint offset){
    uint offsetInUint8Local = offset;
    uint totalUint8Offset = refOffset.y + offsetInUint8Local;
    return GetResource(bMeshletTriangles,refOffset.x).data[totalUint8Offset];
}

uint readVertexIndex(uint meshletid, uint meshletRef, uint obj, uint offset){
    uint offsetInUint8Local = offset;
    uint vertexRef = GetResource(bMeshDataRef,obj).meshletVertexBuffer;
    uint meshletVertOffset = GetResource(bMeshlet,meshletRef).data[meshletid].vertex_offset;
    uint totalUint8Offset = meshletVertOffset + offsetInUint8Local;
    return GetResource(bMeshletVertices,vertexRef).data[totalUint8Offset];
}

void main(){
    uint objId = getObjId();
    uint mi = getMeshletId();

    uint trans = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].transformRef;
    mat4 model = GetResource(bLocalTransform,trans).m_localToWorld;
    mat4 worldToClip = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_worldToClip;
    mat4 mvp = worldToClip * model;

    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint inst = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].instanceDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint vertexRef = GetResource(bMeshDataRef,obj).vertexBuffer;

    uint totalTris = GetResource(bMeshlet,meshletRef).data[mi].triangle_count;
    uint totalVerts = GetResource(bMeshlet,meshletRef).data[mi].vertex_count;
    SetMeshOutputsEXT(totalVerts, totalTris);
    uint gtid = gl_LocalInvocationID.x;
    uvec2 triRefOffset = readTriangleIndexRefOffset(mi,meshletRef,obj);

    if(gtid < totalVerts){
        uint i = gtid;
        uint vi = readVertexIndex(mi,meshletRef,obj,i);
        vec3 v0 = GetResource(bVertices,vertexRef).data[vi].xyz;
        vec4 final= mvp * vec4(v0,1.0);
        gl_MeshVerticesEXT[i].gl_Position = final;
#if SYARO_SHADER_SHARED_VISBUFFER_ENABLE_PRIMCULL
        sPositionsXY[i] = final.xy / final.w;
        sPositionsW[i] = final.w;
#endif
        ids[i] = getClusterID();
    }

#if SYARO_SHADER_SHARED_VISBUFFER_ENABLE_PRIMCULL
    barrier();

    if(gtid<totalTris){
        bool bCulled = false;

        // Some one found that culling here introduces performance gain
        // However, the majority of the triangles are culled in the instance/meshlet
        // level. Implementing this culling here requires extra shared memory,
        // which might degrade the performance for GPU runtime resource allocation.
        //
        // In fact, the performance is worse than the original implementation, about 0.1ms/frame in
        // 1800 Standford Bunnies scene. Even I cull all the triangles here, the performance is still
        // worse than the original implementation (0.1ms/frame).
        //
        // Refs:
        // https://zeux.io/2023/04/28/triangle-backface-culling/
        // https://github.com/qiutang98/chord/tree/master
        

        uint triIndices = readTriangleIndex2(triRefOffset,gtid);
        uint triIndexA = triIndices & 0x000000FF;
        uint triIndexB = (triIndices & 0x0000FF00) >> 8;
        uint triIndexC = (triIndices & 0x00FF0000) >> 16;

        vec3 vA = vec3(sPositionsXY[triIndexA],sPositionsW[triIndexA]);
        vec3 vB = vec3(sPositionsXY[triIndexB],sPositionsW[triIndexB]);
        vec3 vC = vec3(sPositionsXY[triIndexC],sPositionsW[triIndexC]);

        float det = determinant(mat3(vA,vB,vC));
        float maxZ = max(vA.z,max(vB.z,vC.z));
        float minZ = min(vA.z,min(vB.z,vC.z));

        vec2 uvA = vA.xy * 0.5 + 0.5;
        vec2 uvB = vB.xy * 0.5 + 0.5;
        vec2 uvC = vC.xy * 0.5 + 0.5;

        vec2 maxUV = max(uvA,max(uvB,uvC)); 
        vec2 minUV = min(uvA,min(uvB,uvC));

        float renderHeight = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_renderHeight;
        float renderWidth = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_renderWidth;

        maxUV = round(maxUV * vec2(renderWidth,renderHeight));
        minUV = round(minUV * vec2(renderWidth,renderHeight));

        bCulled = (maxUV.x==minUV.x) || (maxUV.y == minUV.y);
        bCulled = bCulled || (det > 1e-3);

        if(minZ < 0.0 || maxZ > 0.0){
            bCulled = false;
        }

        gl_PrimitiveTriangleIndicesEXT[gtid] = uvec3(triIndexA,triIndexB,triIndexC);
        gl_MeshPrimitivesEXT[gtid].gl_PrimitiveID = int(gtid);
        gl_MeshPrimitivesEXT[gtid].gl_CullPrimitiveEXT = bCulled;
    }
#else
    if(gtid<totalTris){
        uint triIndices = readTriangleIndex2(triRefOffset,gtid);
        uint triIndexA = triIndices & 0x000000FF;
        uint triIndexB = (triIndices & 0x0000FF00) >> 8;
        uint triIndexC = (triIndices & 0x00FF0000) >> 16;

        gl_PrimitiveTriangleIndicesEXT[gtid] = uvec3(triIndexA,triIndexB,triIndexC);
        gl_MeshPrimitivesEXT[gtid].gl_PrimitiveID = int(gtid);
        gl_MeshPrimitivesEXT[gtid].gl_CullPrimitiveEXT = false;
    }
#endif
}