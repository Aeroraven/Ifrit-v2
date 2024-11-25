#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro.Shared.glsl"

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 128, max_primitives = 128) out;

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

RegisterStorage(bDrawCallSize,{
    uint x2;
    uint y2;
    uint z2; 
    uint x1;
    uint y1;
    uint z1;
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

bool isSecondCullingPass(){
    return pConst.passNo == 1;
}
uint getClusterID(){
    if(!isSecondCullingPass()){
        return gl_WorkGroupID.x;
    }else{
        return GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).x1 + gl_WorkGroupID.x;
    }
}
uint getObjId(){
    if(!isSecondCullingPass()){
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[gl_WorkGroupID.x].x;
    }else{
        uint baseOffset = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).x1;
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[baseOffset + gl_WorkGroupID.x].x;
    }
}
uint getMeshletId(){
    if(!isSecondCullingPass()){
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[gl_WorkGroupID.x].y;
    }else{
        uint baseOffset = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).x1;
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[baseOffset + gl_WorkGroupID.x].y;
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
    uint filterRef = GetResource(bInstanceDataRef,inst).filteredMeshletsBuffer;
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
        gl_MeshVerticesEXT[i].gl_Position = mvp * vec4(v0,1.0);
        ids[i] = getClusterID();
    }
    if(gtid < totalTris){
        uint i = gtid;
        uint triIndices = readTriangleIndex2(triRefOffset,i);
        uint triIndexA = triIndices & 0x000000FF;
        uint triIndexB = (triIndices & 0x0000FF00) >> 8;
        uint triIndexC = (triIndices & 0x00FF0000) >> 16;
        gl_PrimitiveTriangleIndicesEXT[i] = uvec3(triIndexA,triIndexB,triIndexC);
        gl_MeshPrimitivesEXT[i].gl_PrimitiveID = int(i);
    }
}