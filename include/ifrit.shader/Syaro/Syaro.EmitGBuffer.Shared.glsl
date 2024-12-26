
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



struct MaterialCounter{
    uint counter;
    uint offset;
};

struct MaterialPassIndirectCommand{
    uint x;
    uint y;
    uint z;
};

RegisterStorage(bMaterialCounter,{
    uint totalCounter;
    uint pad;
    MaterialCounter data[];
});

RegisterStorage(bMaterialPixelList,{
    uint data[];
});
RegisterStorage(bPerPixelCounterOffset,{
    uint data[];
});
RegisterStorage(bMaterialPassIndirectCommand,{
    MaterialPassIndirectCommand data[];
});
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
RegisterStorage(bNormals,{
    vec4 data[];
});
RegisterStorage(bUVs,{
    vec2 data[];
});
RegisterStorage(bTangents,{
    vec4 data[];
});

RegisterStorage(bFilteredMeshlets2,{
    uvec2 data[];
});

RegisterStorage(bMaterialData,{
    uint albedoTexId;
    uint normalTexId;
});

layout(binding = 0, set = 1) uniform MaterialPassData{
    uint materialDepthRef;
    uint materialCounterRef;
    uint materialPixelListRef;
    uint perPixelCounterOffsetRef;
    uint indirectCommandRef;
    uint debugImageRef;
} uMaterialPassData;

layout(binding = 0, set = 2) uniform EmitGBufferRefs{
    uint gBufferRef;
} uGBufferRefs;

layout(binding = 0, set = 3) uniform PerframeViewData{
    uint refCurFrame;
    uint refPrevFrame;
}uPerframeView;

layout(binding = 0, set = 4) uniform InstanceData{
    uvec4 ref;
}uInstanceData;

layout(binding = 0, set = 5) uniform IndirectDrawData{
    uint allMeshletsRef;
    uint indDrawCmdRef;
}uIndirectDrawData;

layout(binding = 0, set = 6) uniform EmitDepthTargetData{
    uint velocityMaterialRef; //Seems rgb32f specifiers are not provided
    uint visBufferRef;
}uEmitDepthTargetData;


layout(push_constant) uniform EmitGBufferPushConstant{
    uint materialIndex;
    uint renderWidth;
    uint renderHeight;
} uEmitGBufferPushConstant;

uint gbcomp_GetGBufferId(){
    return uGBufferRefs.gBufferRef;
}
uint gbcomp_GetTotalPixels(){
    uint materialIndex = uEmitGBufferPushConstant.materialIndex;
    uint thisMaterialCounter = GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[materialIndex].counter;
    return thisMaterialCounter;
}

uvec2 gbcomp_GetPixel(){
    uint materialIdx = uEmitGBufferPushConstant.materialIndex;
    uint tX = gl_GlobalInvocationID.x;
    uint totalPx = gbcomp_GetTotalPixels();
    if(tX>=totalPx){
        return uvec2(~0,~0);
    }
    uint offset = GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[materialIdx].offset;
    uint pixelIdx = GetResource(bMaterialPixelList, uMaterialPassData.materialPixelListRef).data[offset + tX];
    uint pX = pixelIdx % uEmitGBufferPushConstant.renderWidth;
    uint pY = pixelIdx / uEmitGBufferPushConstant.renderWidth;
    return uvec2(pX, pY);
}

uvec2 gbcomp_GetVisBufferData(uvec2 pos){
    uint sampledVal = texelFetch(GetSampler2DU(uEmitDepthTargetData.visBufferRef), ivec2(pos), 0).r;
    uint clusterId = (sampledVal >> 7)-1;
    uint triangleId = sampledVal & 0x0000007Fu;
    return uvec2(clusterId, triangleId);
}

uint _gbcomp_readTriangleIndex(uvec2 objMeshletId, uint triangleId){
    uint meshletId = objMeshletId.y;
    uint objId = objMeshletId.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint indexRef = GetResource(bMeshDataRef,obj).meshletIndexBuffer;
    uint meshletOffset = GetResource(bMeshlet, meshletRef).data[meshletId].triangle_offset;
    uint triangleIndex = GetResource(bMeshletTriangles, indexRef).data[meshletOffset + triangleId];
    return triangleIndex;
}

uint _gbcomp_readVertexIndex(uvec2 objMeshletId, uint vertexId){
    uint meshletId = objMeshletId.y;
    uint objId = objMeshletId.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint vertexRef = GetResource(bMeshDataRef,obj).meshletVertexBuffer;

    uint meshletOffset = GetResource(bMeshlet, meshletRef).data[meshletId].vertex_offset;
    uint vertexIndex = GetResource(bMeshletVertices, vertexRef).data[meshletOffset + vertexId];
    return vertexIndex;
}

uint _gbcomp_readAlbedoTexId(uvec2 objMeshletId){
    uint objId = objMeshletId.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint materialRef = GetResource(bMeshDataRef,obj).materialDataBufferId;
    uint albedoTexId = GetResource(bMaterialData, materialRef).albedoTexId;
    return albedoTexId;
}

uint _gbcomp_readNormalTexId(uvec2 objMeshletId){
    uint objId = objMeshletId.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint materialRef = GetResource(bMeshDataRef,obj).materialDataBufferId;
    uint normalTexId = GetResource(bMaterialData, materialRef).normalTexId;
    return normalTexId;
}

uvec2 _gbcomp_getObjMeshletId(uint clusterId){
    return GetResource(bFilteredMeshlets2,uIndirectDrawData.allMeshletsRef).data[clusterId];
}

// Code translated from: https://jcgt.org/published/0002/02/04/
vec3 _gbcomp_rayTriangleIntersect(vec3 p0, vec3 p1, vec3 p2, vec3 o, vec3 d){
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
vec3 _gbcomp_getBarycentric(vec3 v0, vec3 v1, vec3 v2, uvec2 texPos){
    vec2 uvNdc = vec2(texPos) / vec2(uEmitGBufferPushConstant.renderWidth, uEmitGBufferPushConstant.renderHeight);
    uvNdc = uvNdc * 2.0 - 1.0;
    float zNear = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_cameraNear;
    vec4 ndcPos = vec4(uvNdc, 0.0, 1.0) * zNear;
    mat4 invP = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_invPerspective;
    vec3 viewDir = normalize((invP * ndcPos).xyz);
    vec3 bary = _gbcomp_rayTriangleIntersect(v0, v1, v2, vec3(0.0), viewDir);
    return bary;
}

vec4 _gbcomp_interpolate4(vec4 v0, vec4 v1, vec4 v2, vec3 bary){
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

vec3 _gbcomp_interpolate3(vec3 v0, vec3 v1, vec3 v2, vec3 bary){
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

vec2 _gbcomp_interpolate2(vec2 v0, vec2 v1, vec2 v2, vec3 bary){
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

struct gbcomp_TriangleData{
    uint v0Idx;
    uint v1Idx;
    uint v2Idx;
    vec4 v0Pos;
    vec4 v1Pos;
    vec4 v2Pos;
    vec4 barycentric;
    vec4 vpPosVS;
    vec4 vpNormalVS;
    vec4 vpUV;

    vec4 vAlbedo;
    vec4 vTangent;
};

gbcomp_TriangleData gbcomp_GetTriangleData(uvec2 clusterTriangleId, uvec2 pxPos){
    gbcomp_TriangleData data;
    uvec2 objMeshletId = _gbcomp_getObjMeshletId(clusterTriangleId.x);
    uint triangleId = clusterTriangleId.y;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objMeshletId.x].objectDataRef;
    uint vertexRef = GetResource(bMeshDataRef,obj).vertexBuffer;
    
    uint vTx = _gbcomp_readTriangleIndex(objMeshletId, triangleId);
    uint v0Tx = vTx & 0x000000FFu;
    uint v1Tx = (vTx & 0x0000FF00u) >> 8;
    uint v2Tx = (vTx & 0x00FF0000u) >> 16;
    
    data.v0Idx = _gbcomp_readVertexIndex(objMeshletId, v0Tx);
    data.v1Idx = _gbcomp_readVertexIndex(objMeshletId, v1Tx);
    data.v2Idx = _gbcomp_readVertexIndex(objMeshletId, v2Tx);
    
    vec4 v0 = vec4(GetResource(bVertices, vertexRef).data[data.v0Idx].xyz, 1.0);
    vec4 v1 = vec4(GetResource(bVertices, vertexRef).data[data.v1Idx].xyz, 1.0);
    vec4 v2 = vec4(GetResource(bVertices, vertexRef).data[data.v2Idx].xyz, 1.0);

    data.v0Pos = v0;
    data.v1Pos = v1;
    data.v2Pos = v2;

    uint transRef = GetResource(bPerObjectRef, uInstanceData.ref.x).data[objMeshletId.x].transformRef;
    mat4 localToWorld = GetResource(bLocalTransform, transRef).m_localToWorld;
    vec4 v0ws = localToWorld * v0;
    vec4 v1ws = localToWorld * v1;
    vec4 v2ws = localToWorld * v2;

    mat4 worldToView = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_worldToView;
    vec4 v0vs = worldToView * v0ws;
    vec4 v1vs = worldToView * v1ws;
    vec4 v2vs = worldToView * v2ws;

    data.barycentric = vec4(_gbcomp_getBarycentric(v0vs.xyz, v1vs.xyz, v2vs.xyz, pxPos),1.0);
    data.vpPosVS = vec4(_gbcomp_interpolate3(v0vs.xyz, v1vs.xyz, v2vs.xyz, data.barycentric.xyz),1.0);

    uint normalRef = GetResource(bMeshDataRef,obj).normalBufferId;
    vec4 n0 = vec4(GetResource(bNormals, normalRef).data[data.v0Idx].xyz, 0.0);
    vec4 n1 = vec4(GetResource(bNormals, normalRef).data[data.v1Idx].xyz, 0.0);
    vec4 n2 = vec4(GetResource(bNormals, normalRef).data[data.v2Idx].xyz, 0.0);

    vec4 n0ws = localToWorld * n0;
    vec4 n1ws = localToWorld * n1;
    vec4 n2ws = localToWorld * n2;

    vec4 n0vs = worldToView * n0ws;
    vec4 n1vs = worldToView * n1ws;
    vec4 n2vs = worldToView * n2ws;

    data.vpNormalVS = vec4(normalize(_gbcomp_interpolate3(n0vs.xyz, n1vs.xyz, n2vs.xyz, data.barycentric.xyz)),1.0);
    vec4 normalLocal = vec4(normalize(_gbcomp_interpolate3(n0.xyz, n1.xyz, n2.xyz, data.barycentric.xyz)),1.0);

    uint uvRef = GetResource(bMeshDataRef,obj).uvBufferId;
    vec2 uv0 = GetResource(bUVs, uvRef).data[data.v0Idx];
    vec2 uv1 = GetResource(bUVs, uvRef).data[data.v1Idx];
    vec2 uv2 = GetResource(bUVs, uvRef).data[data.v2Idx];
    data.vpUV = vec4(_gbcomp_interpolate2(uv0, uv1, uv2, data.barycentric.xyz), 0.0, 1.0);

    uint albedoTexId = _gbcomp_readAlbedoTexId(objMeshletId);
    uint normalTexId = _gbcomp_readNormalTexId(objMeshletId);

    uint tangentRef = GetResource(bMeshDataRef,obj).tangentBufferId;
    vec4 tangent0 = GetResource(bTangents,tangentRef).data[data.v0Idx];
    vec4 tangent1 = GetResource(bTangents,tangentRef).data[data.v1Idx];
    vec4 tangent2 = GetResource(bTangents,tangentRef).data[data.v2Idx];
    data.vTangent = vec4(_gbcomp_interpolate4(tangent0,tangent1,tangent2,data.barycentric.xyz));

    if(albedoTexId != ~0u){
        data.vAlbedo = texture(GetSampler2D(albedoTexId), data.vpUV.xy);
    }

    if(normalTexId!=~0u){
        vec4 tangentX = data.vTangent;
        tangentX.xyz = normalize(tangentX.xyz);
        vec3 normal = normalLocal.xyz;
        vec3 tangent = normalize(tangentX.xyz - dot(normal,tangentX.xyz)*normal);
        vec3 bitangent = cross(tangent,normal) * tangentX.w;
        mat3 tbn = mat3(tangent,bitangent,normal);

        vec4 vNormal = texture(GetSampler2D(normalTexId), data.vpUV.xy);
        vec2 vNormalRG = vNormal.rg * 2.0 - 1.0;
        float vNormalZ = sqrt(1.0-vNormalRG.r*vNormalRG.r-vNormalRG.g*vNormalRG.g);

        vec3 vNormalRGB = vec3(vNormalRG,vNormalZ);

        vec3 rNormal = tbn * normalize(vNormalRGB);
        data.vpNormalVS = worldToView * localToWorld * vec4(rNormal,0.0);
    }
    
    return data;
}
