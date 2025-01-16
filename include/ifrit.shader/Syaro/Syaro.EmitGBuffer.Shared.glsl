
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
    uint allMeshletsRefSW;
    uint indDrawCmdRef;
}uIndirectDrawData;

layout(binding = 0, set = 6) uniform EmitDepthTargetData{
    uint velocityMaterialRef; //Seems rgb32f specifiers are not provided
    uint visBufferRef;
    uint motionVectorRef;
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

#if SYARO_SHADER_SHARED_EMIT_GBUFFER_TRIANGLE_REUSE
uvec2 gbcomp_GetTotalPixelsOffset(){
    uint materialIndex = uEmitGBufferPushConstant.materialIndex;
    uint totalPx = gbcomp_GetTotalPixels();
    uint thisMaterialOffset = GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[materialIndex].offset;
    return uvec2(totalPx, thisMaterialOffset);
}
uvec2 gbcomp_GetPixelReused(uint ofx,uvec2 totalPxMaterOffset){
    uint materialIdx = uEmitGBufferPushConstant.materialIndex;
    uint tX = gl_GlobalInvocationID.x * 4 + ofx;
    uint totalPx = totalPxMaterOffset.x;
    if(tX>=totalPx){
        return uvec2(~0,~0);
    }
    uint offset = totalPxMaterOffset.y;
    uint pixelIdx = GetResource(bMaterialPixelList, uMaterialPassData.materialPixelListRef).data[offset + tX];
    uint pX = pixelIdx % uEmitGBufferPushConstant.renderWidth;
    uint pY = pixelIdx / uEmitGBufferPushConstant.renderWidth;
    return uvec2(pX, pY);
}
#endif

uvec2 gbcomp_GetVisBufferData(uvec2 pos){
    uint sampledVal = texelFetch(GetSampler2DU(uEmitDepthTargetData.visBufferRef), ivec2(pos), 0).r;
    uint clusterId = (sampledVal >> 7)-1;
    uint triangleId = sampledVal & 0x0000007Fu;
    return uvec2(clusterId, triangleId);
}

uvec4 _gbcomp_readMeshletVertexIndicesRef(uvec2 objMeshletId){
    uint meshletId = objMeshletId.y;
    uint objId = objMeshletId.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint vertexRef = GetResource(bMeshDataRef,obj).meshletVertexBuffer;
    uint indexRef = GetResource(bMeshDataRef,obj).meshletIndexBuffer;
    uint vbuf = GetResource(bMeshDataRef,obj).vertexBuffer;
    
    return uvec4(meshletRef, vertexRef, indexRef,vbuf);
}

uint _gbcomp_readTriangleIndex(uvec2 objMeshletId, uint triangleId,uvec4 mviRef){
    uint meshletId = objMeshletId.y;
    uint objId = objMeshletId.x;
    uint meshletRef = mviRef.x;
    uint indexRef = mviRef.z;
    uint meshletOffset = GetResource(bMeshlet, meshletRef).data[meshletId].triangle_offset;
    uint triangleIndex = GetResource(bMeshletTriangles, indexRef).data[meshletOffset + triangleId];
    return triangleIndex;
}

uint _gbcomp_readVertexIndex(uvec2 objMeshletId, uint vertexId,uvec4 mviRef){
    uint meshletId = objMeshletId.y;
    uint objId = objMeshletId.x;
    uint meshletRef = mviRef.x;
    uint vertexRef = mviRef.y;
    uint meshletOffset = GetResource(bMeshlet, meshletRef).data[meshletId].vertex_offset;
    uint vertexIndex = GetResource(bMeshletVertices, vertexRef).data[meshletOffset + vertexId];
    return vertexIndex;
}

uint _gbcomp_readVertexIndex_getMeshletOffset(uvec2 objMeshletId,uvec4 mviRef){
    uint meshletId = objMeshletId.y;
    uint objId = objMeshletId.x;
    uint meshletRef = mviRef.x;
    uint vertexRef = mviRef.y;
    uint meshletOffset = GetResource(bMeshlet, meshletRef).data[meshletId].vertex_offset;
    return meshletOffset;
}
uint _gbcomp_readVertexIndex_2(uvec2 objMeshletId, uint vertexId,uvec4 mviRef,uint meshletOffset){
    uint meshletId = objMeshletId.y;
    uint objId = objMeshletId.x;
    uint meshletRef = mviRef.x;
    uint vertexRef = mviRef.y;
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

uvec2 _gbcomp_readAlbedoNormalTexId(uvec2 objMeshletId){
    uint objId = objMeshletId.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint materialRef = GetResource(bMeshDataRef,obj).materialDataBufferId;
    uint albedoTexId = GetResource(bMaterialData, materialRef).albedoTexId;
    uint normalTexId = GetResource(bMaterialData, materialRef).normalTexId;
    return uvec2(albedoTexId, normalTexId);
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

// Requires v0,v1,v2 to be in clip space
vec3 _gbcomp_getBarycentric_v2(vec4 v0, vec4 v1, vec4 v2, uvec2 texPos){
    vec2 uvNdc = vec2(texPos) / vec2(uEmitGBufferPushConstant.renderWidth, uEmitGBufferPushConstant.renderHeight);
    uvNdc = uvNdc * 2.0 - 1.0;
    
    vec3 rcpW = 1.0 / vec3(v0.w, v1.w, v2.w);
    vec3 p0 = v0.xyz * rcpW.x;
    vec3 p1 = v1.xyz * rcpW.y;
    vec3 p2 = v2.xyz * rcpW.z;

    vec3 p120x = vec3(p1.x, p2.x, p0.x);
    vec3 p120y = vec3(p1.y, p2.y, p0.y);
    vec3 p201x = vec3(p2.x, p0.x, p1.x);
    vec3 p201y = vec3(p2.y, p0.y, p1.y);

    vec3 cdx = p201y - p120y;
    vec3 cdy = p120x - p201x;

    vec3 a = cdx*(uvNdc.x-p120x) + cdy*(uvNdc.y-p120y);
    vec3 b = a*rcpW;

    float h = dot(a,rcpW);
    float rcpH = 1.0/h;
    return b*rcpH;
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
    vec4 barycentric;
    vec4 vpPosVS;
    vec4 vAlbedo;
    vec4 vTangent;
    vec3 vpNormalVS;
    vec2 vpUV;
};

#if SYARO_SHADER_SHARED_EMIT_GBUFFER_TRIANGLE_REUSE
struct gbcomp_TriangleDataShared{
    vec4 v0PosVS;
    vec4 v1PosVS;
    vec4 v2PosVS;

    vec3 v0NormalMS;
    vec3 v1NormalMS;
    vec3 v2NormalMS;
    vec4 v0Tangent;
    vec4 v1Tangent;
    vec4 v2Tangent;
    vec2 v0UV;
    vec2 v1UV;
    vec2 v2UV;
};
#endif

gbcomp_TriangleData gbcomp_GetTriangleData(uvec2 clusterTriangleId, uvec2 pxPos){
    gbcomp_TriangleData data;
    uvec2 objMeshletId = _gbcomp_getObjMeshletId(clusterTriangleId.x);
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objMeshletId.x].objectDataRef;
    uint triangleId = clusterTriangleId.y;
    
    uvec4 mviRef = _gbcomp_readMeshletVertexIndicesRef(objMeshletId);
    uint vertexRef = mviRef.w;

    uint vTx = _gbcomp_readTriangleIndex(objMeshletId, triangleId,mviRef);
    uint v0Tx = vTx & 0x000000FFu;
    uint v1Tx = (vTx & 0x0000FF00u) >> 8;
    uint v2Tx = (vTx & 0x00FF0000u) >> 16;
    
    uint v0Idx = _gbcomp_readVertexIndex(objMeshletId, v0Tx,mviRef);
    uint v1Idx = _gbcomp_readVertexIndex(objMeshletId, v1Tx,mviRef);
    uint v2Idx = _gbcomp_readVertexIndex(objMeshletId, v2Tx,mviRef);
    
    vec4 v0 = vec4(GetResource(bVertices, vertexRef).data[v0Idx].xyz, 1.0);
    vec4 v1 = vec4(GetResource(bVertices, vertexRef).data[v1Idx].xyz, 1.0);
    vec4 v2 = vec4(GetResource(bVertices, vertexRef).data[v2Idx].xyz, 1.0);

    uint transRef = GetResource(bPerObjectRef, uInstanceData.ref.x).data[objMeshletId.x].transformRef;
    mat4 localToWorld = GetResource(bLocalTransform, transRef).m_localToWorld;
    mat4 worldToView = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_worldToView;
    mat4 localToView = worldToView * localToWorld;

    vec4 v0vs = localToView * v0;
    vec4 v1vs = localToView * v1;
    vec4 v2vs = localToView * v2;

    data.barycentric = vec4(_gbcomp_getBarycentric(v0vs.xyz, v1vs.xyz, v2vs.xyz, pxPos),1.0);
    data.vpPosVS = vec4(_gbcomp_interpolate3(v0vs.xyz, v1vs.xyz, v2vs.xyz, data.barycentric.xyz),1.0);

    uint normalRef = GetResource(bMeshDataRef,obj).normalBufferId;
    vec3 n0 = vec3(GetResource(bNormals, normalRef).data[v0Idx].xyz);
    vec3 n1 = vec3(GetResource(bNormals, normalRef).data[v1Idx].xyz);
    vec3 n2 = vec3(GetResource(bNormals, normalRef).data[v2Idx].xyz);

    mat3 localToView3 = mat3(localToView);

    vec3 n0vs = localToView3 * n0;
    vec3 n1vs = localToView3 * n1;
    vec3 n2vs = localToView3 * n2;

    vec4 normalLocal = vec4(normalize(_gbcomp_interpolate3(n0.xyz, n1.xyz, n2.xyz, data.barycentric.xyz)),1.0);

    uint uvRef = GetResource(bMeshDataRef,obj).uvBufferId;
    vec2 uv0 = GetResource(bUVs, uvRef).data[v0Idx];
    vec2 uv1 = GetResource(bUVs, uvRef).data[v1Idx];
    vec2 uv2 = GetResource(bUVs, uvRef).data[v2Idx];
    data.vpUV = _gbcomp_interpolate2(uv0, uv1, uv2, data.barycentric.xyz);

    uint albedoTexId = _gbcomp_readAlbedoTexId(objMeshletId);
    uint normalTexId = _gbcomp_readNormalTexId(objMeshletId);

    uint tangentRef = GetResource(bMeshDataRef,obj).tangentBufferId;
    vec4 tangent0 = GetResource(bTangents,tangentRef).data[v0Idx];
    vec4 tangent1 = GetResource(bTangents,tangentRef).data[v1Idx];
    vec4 tangent2 = GetResource(bTangents,tangentRef).data[v2Idx];
    data.vTangent = _gbcomp_interpolate4(tangent0,tangent1,tangent2,data.barycentric.xyz);

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

        vec3 rNormal = tbn * vNormalRGB;
        data.vpNormalVS = rNormal;
    }
    
    return data;
}

#if SYARO_SHADER_SHARED_EMIT_GBUFFER_TRIANGLE_REUSE

gbcomp_TriangleDataShared gbcomp_GetTriangleDataImp(uvec2 clusterTriangleId, uvec2 pxPos){
    gbcomp_TriangleDataShared data;
    uvec2 objMeshletId = _gbcomp_getObjMeshletId(clusterTriangleId.x);
    uint triangleId = clusterTriangleId.y;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objMeshletId.x].objectDataRef;

    uvec4 mviRef = _gbcomp_readMeshletVertexIndicesRef(objMeshletId);
    uint vertexRef = mviRef.w;

    uint vTx = _gbcomp_readTriangleIndex(objMeshletId, triangleId,mviRef);
    uint v0Tx = vTx & 0x000000FFu;
    uint v1Tx = (vTx & 0x0000FF00u) >> 8;
    uint v2Tx = (vTx & 0x00FF0000u) >> 16;
    
    uint meshletOffset = _gbcomp_readVertexIndex_getMeshletOffset(objMeshletId,mviRef);
    uint v0Idx = _gbcomp_readVertexIndex_2(objMeshletId, v0Tx,mviRef,meshletOffset);
    uint v1Idx = _gbcomp_readVertexIndex_2(objMeshletId, v1Tx,mviRef,meshletOffset);
    uint v2Idx = _gbcomp_readVertexIndex_2(objMeshletId, v2Tx,mviRef,meshletOffset);

    uint transRef = GetResource(bPerObjectRef, uInstanceData.ref.x).data[objMeshletId.x].transformRef;
    mat4 localToWorld = GetResource(bLocalTransform, transRef).m_localToWorld;
    mat4 worldToClip = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_worldToClip;
    mat4 localToClip = worldToClip * localToWorld;

    vec4 v0vs = localToClip * GetResource(bVertices, vertexRef).data[v0Idx];
    vec4 v1vs = localToClip * GetResource(bVertices, vertexRef).data[v1Idx];
    vec4 v2vs = localToClip * GetResource(bVertices, vertexRef).data[v2Idx];

#if SYARO_SHADER_SHARED_EMIT_GBUFFER_TRIANGLE_REUSE
    data.v0PosVS = v0vs;
    data.v1PosVS = v1vs;
    data.v2PosVS = v2vs;
#endif

    uint normalRef = GetResource(bMeshDataRef,obj).normalBufferId;
    vec4 n0 = GetResource(bNormals, normalRef).data[v0Idx];
    vec4 n1 = GetResource(bNormals, normalRef).data[v1Idx];
    vec4 n2 = GetResource(bNormals, normalRef).data[v2Idx];

#if SYARO_SHADER_SHARED_EMIT_GBUFFER_TRIANGLE_REUSE
    data.v0NormalMS = n0.xyz;
    data.v1NormalMS = n1.xyz;
    data.v2NormalMS = n2.xyz;
#endif

    uint uvRef = GetResource(bMeshDataRef,obj).uvBufferId;
    vec2 uv0 = GetResource(bUVs, uvRef).data[v0Idx];
    vec2 uv1 = GetResource(bUVs, uvRef).data[v1Idx];
    vec2 uv2 = GetResource(bUVs, uvRef).data[v2Idx];

#if SYARO_SHADER_SHARED_EMIT_GBUFFER_TRIANGLE_REUSE
    data.v0UV = uv0;
    data.v1UV = uv1;
    data.v2UV = uv2;
#endif

    uint tangentRef = GetResource(bMeshDataRef,obj).tangentBufferId;
    vec4 tangent0 = GetResource(bTangents,tangentRef).data[v0Idx];
    vec4 tangent1 = GetResource(bTangents,tangentRef).data[v1Idx];
    vec4 tangent2 = GetResource(bTangents,tangentRef).data[v2Idx];

#if SYARO_SHADER_SHARED_EMIT_GBUFFER_TRIANGLE_REUSE
    data.v0Tangent = tangent0;
    data.v1Tangent = tangent1;
    data.v2Tangent = tangent2;
#endif

    return data;
}

gbcomp_TriangleData gbcomp_GetTriangleDataReused(gbcomp_TriangleDataShared lastData,uvec2 clusterTriangleId, uvec2 pxPos){
    gbcomp_TriangleData data;

    data.barycentric = vec4(_gbcomp_getBarycentric_v2(lastData.v0PosVS, lastData.v1PosVS, lastData.v2PosVS, pxPos),1.0);
   
    vec3 normalLocal = normalize(_gbcomp_interpolate3(lastData.v0NormalMS, lastData.v1NormalMS, lastData.v2NormalMS, data.barycentric.xyz));
    data.vpUV = _gbcomp_interpolate2(lastData.v0UV, lastData.v1UV, lastData.v2UV, data.barycentric.xyz);
    data.vTangent = _gbcomp_interpolate4(lastData.v0Tangent,lastData.v1Tangent,lastData.v2Tangent,data.barycentric.xyz);

    uvec2 objMeshletId = _gbcomp_getObjMeshletId(clusterTriangleId.x);
    uvec2 albedoNormalTexId = _gbcomp_readAlbedoNormalTexId(objMeshletId);
    uint albedoTexId = albedoNormalTexId.x;
    uint normalTexId = albedoNormalTexId.y;

    if(albedoTexId != ~0u){
        data.vAlbedo = texture(GetSampler2D(albedoTexId), data.vpUV.xy);
    }

    if(normalTexId!=~0u){
        vec4 tangentX = data.vTangent;
        tangentX.xyz = normalize(tangentX.xyz);
        vec3 normal = normalLocal;
        vec3 tangent = normalize(tangentX.xyz - dot(normal,tangentX.xyz)*normal);
        vec3 bitangent = cross(tangent,normal) * tangentX.w;
        mat3 tbn = mat3(tangent,bitangent,normal);

        vec4 vNormal = texture(GetSampler2D(normalTexId), data.vpUV.xy);
        vec2 vNormalRG = vNormal.rg * 2.0 - 1.0;
        float vNormalZ = sqrt(1.0-vNormalRG.r*vNormalRG.r-vNormalRG.g*vNormalRG.g);

        vec3 vNormalRGB = vec3(vNormalRG,vNormalZ);

        vec3 rNormal = tbn * vNormalRGB;
        data.vpNormalVS = rNormal;

        // make this to be in view space. TODO: inverse transform
        mat4 worldToView = GetResource(bPerframeView, uPerframeView.refCurFrame).data.m_worldToView;
        mat4 localToWorld = GetResource(bLocalTransform, GetResource(bPerObjectRef,uInstanceData.ref.x).data[objMeshletId.x].transformRef).m_localToWorld;
        mat4 localToView = worldToView * localToWorld;

        data.vpNormalVS = normalize(data.vpNormalVS);
        data.vpNormalVS = vec3(localToView * vec4(data.vpNormalVS,0.0));
        data.vpNormalVS = normalize(data.vpNormalVS);
    }

    return data;
}

#endif