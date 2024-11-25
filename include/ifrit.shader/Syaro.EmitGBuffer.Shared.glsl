
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


RegisterStorage(bFilteredMeshlets2,{
    uvec2 data[];
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

    uint uvRef = GetResource(bMeshDataRef,obj).uvBufferId;
    vec2 uv0 = GetResource(bUVs, uvRef).data[data.v0Idx];
    vec2 uv1 = GetResource(bUVs, uvRef).data[data.v1Idx];
    vec2 uv2 = GetResource(bUVs, uvRef).data[data.v2Idx];

    data.vpUV = vec4(_gbcomp_interpolate2(uv0, uv1, uv2, data.barycentric.xyz), 0.0, 1.0);

    return data;
}
