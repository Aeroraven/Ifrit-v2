#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "Base.glsl"
#include "Bindless.glsl"


layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

struct Meshlet {
  uint vertex_offset;
  uint triangle_offset;
  uint vertex_count;
  uint triangle_count;
};

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

RegisterStorage(bFilteredMeshlets,{
    uint data[];
});

RegisterStorage(bPerObjectRef,{
    PerObjectData data[];
});
RegisterUniform(bLocalTransform,{
    mat4 m_localToWorld;
});
RegisterStorage(bMeshDataRef,{
    uint vertexBuffer;
    uint meshletBuffer;
    uint meshletVertexBuffer;
    uint meshletIndexBuffer;
    uint meshletCullBuffer;
    uint bvhNodeBuffer;
    uint clusterGroupBuffer;
    uint meshletInClusterBuffer;
    uint cpQueueBuffer;
    uint cpCounterBuffer;
    uint filteredMeshletsBuffer;
    uint pad;
});
RegisterUniform(bPerframeView,{
    PerFramePerViewData data;
});
layout(binding = 0, set = 1) uniform PerframeViewData{
    uvec4 ref;
}uPerframeView;

layout(binding = 0, set = 2) uniform InstanceData{
    uvec4 ref;
}uInstanceData;

layout(location = 0) out vec3 fragColor[];

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

uint readTriangleIndex(uint meshletid, uint offset){
    uint objId = gl_WorkGroupID.y;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint offsetInUint8Local = offset;
    uint meshletTriOffset = GetResource(bMeshlet,meshletRef).data[meshletid].triangle_offset;
    uint totalUint8Offset = meshletTriOffset + offsetInUint8Local;
    uint indexRef = GetResource(bMeshDataRef,obj).meshletIndexBuffer;
    uint indexDataU32 = GetResource(bMeshletTriangles,indexRef).data[totalUint8Offset];
    return indexDataU32;
}

uint readVertexIndex(uint meshletid, uint offset){
    uint offsetInUint8Local = offset;
    uint objId = gl_WorkGroupID.y;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint vertexRef = GetResource(bMeshDataRef,obj).meshletVertexBuffer;
    uint meshletVertOffset = GetResource(bMeshlet,meshletRef).data[meshletid].vertex_offset;
    uint totalUint8Offset = meshletVertOffset + offsetInUint8Local;
    return GetResource(bMeshletVertices,vertexRef).data[totalUint8Offset];
}

void main(){
    uint mio = gl_WorkGroupID.x;
    // This should be in task shader, but there's only one mesh now.
    uint objId = gl_WorkGroupID.y;

    uint trans = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].transformRef;
    mat4 model = GetResource(bLocalTransform,trans).m_localToWorld;
    mat4 view = GetResource(bPerframeView,uPerframeView.ref.x).data.m_worldToView;
    mat4 proj = GetResource(bPerframeView,uPerframeView.ref.x).data.m_perspective;
    mat4 mvp = proj * view * model;


    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint filterRef = GetResource(bMeshDataRef,obj).filteredMeshletsBuffer;
    uint vertexRef = GetResource(bMeshDataRef,obj).vertexBuffer;
    uint mi = GetResource(bFilteredMeshlets,filterRef).data[mio];

    uint totalTris = GetResource(bMeshlet,meshletRef).data[mi].triangle_count;
    uint totalVerts = GetResource(bMeshlet,meshletRef).data[mi].vertex_count;
    SetMeshOutputsEXT(totalVerts, totalTris);
    for(uint i = 0; i < totalVerts ; i++){
        uint vi = readVertexIndex(mi,i);
        vec3 v0 = GetResource(bVertices,vertexRef).data[vi].xyz;
        gl_MeshVerticesEXT[i].gl_Position = mvp * vec4(v0,1.0);
        fragColor[i] = colorMap[mi % 8].rgb;
    }
    for(uint i = 0; i < totalTris ; i++){
        uint triIndexA = readTriangleIndex(mi,i*3 + 0);
        uint triIndexB = readTriangleIndex(mi,i*3 + 1);
        uint triIndexC = readTriangleIndex(mi,i*3 + 2);
        gl_PrimitiveTriangleIndicesEXT[i] = uvec3(triIndexA,triIndexB,triIndexC);
    }
}