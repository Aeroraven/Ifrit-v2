#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro.Shared.glsl"


layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

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

layout(binding = 0, set = 3) uniform IndirectDrawData{
    uint allMeshletsRef;
    uint indDrawCmdRef;
}uIndirectDrawData;

layout(binding = 0, set = 4) uniform EmitDepthTargetData{
    uint velocityMaterialRef; //Seems rgb32f specifiers are not provided
    uint visBufferRef;
}uEmitDepthTargetData;

layout(push_constant) uniform EmitDepthTargetPushConstant{
    uint renderWidth;
    uint renderHeight;
}uEmitDepthTargetPushConstant;

uvec2 unpackVisBuffer(uvec2 pos){
    uint sampledVal = texelFetch(GetSampler2DU(uEmitDepthTargetData.visBufferRef), ivec2(pos), 0).r;
    uint clusterId = (sampledVal >> 7);
    uint triangleId = sampledVal & 0x0000007Fu;
    return uvec2(clusterId, triangleId);
}

// Code translated from: https://jcgt.org/published/0002/02/04/
vec3 rayTriangleIntersect(vec3 p0, vec3 p1, vec3 p2, vec3 o, vec3 d){
   vec3 eo =  o - p0;
   vec3 e2 = p2 - p0;
   vec3 e1 = p1 - p0;
   vec3 r  = cross(d, e2);
   vec3 s  = cross(eo, e1);
   float iV  = 1.0f / dot(r, e1);
   float V1  = dot(r, eo);
   float V2  = dot(s,  d);
   float b   = V1 * iV;
   float c   = V2 * iV;
   float a   = 1.0f - b - c;
   return vec3(a, b, c);
}

// Requires v0,v1,v2 to be in view space
vec3 getBarycentric(vec3 v0, vec3 v1, vec3 v2, uvec2 texPos){
    vec2 uvNdc = vec2(texPos) / vec2(uEmitDepthTargetPushConstant.renderWidth, uEmitDepthTargetPushConstant.renderHeight);
    uvNdc = uvNdc * 2.0 - 1.0;
    float zNear = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_cameraNear;
    vec4 ndcPos = vec4(uvNdc, 0.0, 1.0) * zNear;
    mat4 invP = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_invPerspective;
    vec3 viewDir = normalize((invP * ndcPos).xyz);
    vec3 bary = rayTriangleIntersect(v0, v1, v2, vec3(0.0), viewDir);
    return bary;
}

uvec2 getObjMeshletId(uint clusterId){
    return GetResource(bFilteredMeshlets2,uIndirectDrawData.allMeshletsRef).data[clusterId];
}

uint readTriangleIndex(uvec2 objMeshletId, uint triangleId){
    uint meshletId = objMeshletId.y;
    uint objId = objMeshletId.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint indexRef = GetResource(bMeshDataRef,obj).meshletIndexBuffer;
    uint meshletOffset = GetResource(bMeshlet, meshletRef).data[meshletId].triangle_offset;
    uint triangleIndex = GetResource(bMeshletTriangles, indexRef).data[meshletOffset + triangleId];
    return triangleIndex;
}

uint readVertexIndex(uvec2 objMeshletId, uint vertexId){
    uint meshletId = objMeshletId.y;
    uint objId = objMeshletId.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint vertexRef = GetResource(bMeshDataRef,obj).meshletVertexBuffer;

    uint meshletOffset = GetResource(bMeshlet, meshletRef).data[meshletId].vertex_offset;
    uint vertexIndex = GetResource(bMeshletVertices, vertexRef).data[meshletOffset + vertexId];
    return vertexIndex;
}

vec3 interpolate3(vec3 v0, vec3 v1, vec3 v2, vec3 bary){
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

void main(){
    uint px = gl_GlobalInvocationID.x;
    uint py = gl_GlobalInvocationID.y;
    if(px >= uEmitDepthTargetPushConstant.renderWidth || py >= uEmitDepthTargetPushConstant.renderHeight){
        return;
    }
    uvec2 pos = uvec2(px, py);
    uvec2 clusterTriangleId = unpackVisBuffer(pos);
    if(clusterTriangleId.x == 0){
        imageStore(GetUAVImage2DRGBA32F(uEmitDepthTargetData.velocityMaterialRef), ivec2(pos), vec4(0.0));
        return;
    }
    clusterTriangleId.x -= 1;
    uvec2 objMeshletId = getObjMeshletId(clusterTriangleId.x);

    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objMeshletId.x].objectDataRef;
    uint vertexRef = GetResource(bMeshDataRef,obj).vertexBuffer;

    uint vTx = readTriangleIndex(objMeshletId, clusterTriangleId.y);
    uint v0Tx = vTx & 0x000000FFu;
    uint v1Tx = (vTx & 0x0000FF00u) >> 8;
    uint v2Tx = (vTx & 0x00FF0000u) >> 16;

    uint v0Idx = readVertexIndex(objMeshletId, v0Tx);
    uint v1Idx = readVertexIndex(objMeshletId, v1Tx);
    uint v2Idx = readVertexIndex(objMeshletId, v2Tx);

    vec4 v0 = vec4(GetResource(bVertices, vertexRef).data[v0Idx].xyz, 1.0);
    vec4 v1 = vec4(GetResource(bVertices, vertexRef).data[v1Idx].xyz, 1.0);
    vec4 v2 = vec4(GetResource(bVertices, vertexRef).data[v2Idx].xyz, 1.0);

    uint transRef = GetResource(bPerObjectRef, uInstanceData.ref.x).data[objMeshletId.x].transformRef;
    mat4 localToWorld = GetResource(bLocalTransform, transRef).m_localToWorld;
    vec4 v0ws = localToWorld * v0;
    vec4 v1ws = localToWorld * v1;
    vec4 v2ws = localToWorld * v2;

    mat4 worldToView = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_worldToView;
    vec4 v0vs = worldToView * v0ws;
    vec4 v1vs = worldToView * v1ws;
    vec4 v2vs = worldToView * v2ws;

    vec3 bary = getBarycentric(v0vs.xyz, v1vs.xyz, v2vs.xyz, pos);

    // Motion vector
    uint transLastRef = GetResource(bPerObjectRef, uInstanceData.ref.x).data[objMeshletId.x].transformRefLast;
    mat4 localToWorldLast = GetResource(bLocalTransform, transLastRef).m_localToWorld;
    mat4 worldToViewLast = GetResource(bPerframeView, uPerframeView.refPrevFrame).data.m_worldToView;
    mat4 projectionLast = GetResource(bPerframeView, uPerframeView.refPrevFrame).data.m_perspective;
    mat4 projNow = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_perspective;

    // Eliminate jitter in the motion vector
    projectionLast[0][2] = 0.0;
    projectionLast[1][2] = 0.0;
    projectionLast[2][0] = 0.0;
    projectionLast[2][1] = 0.0;

    projNow[0][2] = 0.0;
    projNow[1][2] = 0.0;
    projNow[2][0] = 0.0;
    projNow[2][1] = 0.0;

    vec4 vp = vec4(interpolate3(v0.xyz, v1.xyz, v2.xyz, bary),1.0);
    vec4 vpLast = projectionLast * worldToViewLast * localToWorldLast * vp;
    vec2 ndcVpLast = vpLast.xy / vpLast.w;
    ndcVpLast = ndcVpLast * 0.5 + 0.5;

    vec4 vpNowX = projNow * worldToView * localToWorld * vp;
    vec2 ndcVpNow = vpNowX.xy / vpNowX.w;
    ndcVpNow = ndcVpNow * 0.5 + 0.5;


    vec2 motionVector = ndcVpNow - ndcVpLast;

    // Depth, this is just for debugging.
    vec4 vpScr = projectionLast * worldToViewLast * localToWorldLast * vp;
    float msDepth = vpScr.z / vpScr.w;

    // Material ID
    uint instanceRef = GetResource(bPerObjectRef, uInstanceData.ref.x).data[objMeshletId.x].instanceDataRef;
    uint materialId = GetResource(bPerObjectRef, uInstanceData.ref.x).data[objMeshletId.x].materialId;

    // Then write to the buffer
    vec4 velMatData = vec4((motionVector), msDepth, float(materialId)+1.0);
    imageStore(GetUAVImage2DRGBA32F(uEmitDepthTargetData.velocityMaterialRef), ivec2(pos), velMatData);

}

