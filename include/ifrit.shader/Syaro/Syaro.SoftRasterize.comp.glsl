
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


// Software rasterizer 

#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_atomic_float : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

layout(local_size_x = cMeshRasterizeThreadGroupSizeX, local_size_y = 1, local_size_z = 1) in;

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

RegisterStorage(bDepth,{
    uint64_t data[];
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

layout(push_constant) uniform SWPushConstant{
    uint passNo;
    uint depthBufferId;
    uint visBufferId;
    uint renderHeight;
    uint renderWidth;
    uint swOffset;
    uint casBufferId;
} pConst;

shared vec4 sharedVertices[cMeshRasterizeMaxVertexSize];

bool isSecondCullingPass(){
    return pConst.passNo == 1;
}

uint getAcutalWorkGroup(){
    return gl_WorkGroupID.x;
}

uint getClusterID(){
    uint actualWorkGroup = getAcutalWorkGroup();
    if(!isSecondCullingPass()){
        return actualWorkGroup + pConst.swOffset;
    }else{
        return GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1sw + actualWorkGroup + pConst.swOffset;
    }
}

uint getObjId(){
    uint actualWorkGroup = getAcutalWorkGroup() + pConst.swOffset;
    if(!isSecondCullingPass()){
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRefSW).data[actualWorkGroup].x;
    }else{
        uint baseOffset = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1sw;
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRefSW).data[baseOffset + actualWorkGroup].x;
    }
}

uint getMeshletId(){
    uint actualWorkGroup = getAcutalWorkGroup() + pConst.swOffset;
    if(!isSecondCullingPass()){
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRefSW).data[actualWorkGroup].y;
    }else{
        uint baseOffset = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1sw;
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRefSW).data[baseOffset + actualWorkGroup].y;
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

void writePixel(uvec2 pos, uint triangleId, float depth){
    uint offset = pos.y * pConst.renderWidth + pos.x;
    uint lowPart = triangleId & 0x0000007Fu;
    uint ids = getClusterID();
    uint highPart = ((ids+1)<<7) & 0xFFFFFF80u;
    uint outColor = highPart | lowPart;
    double scaled = double(depth) * double(4294967295.0);
    uint depthBits = uint(scaled);
    uint64_t zbufValue = (uint64_t(depthBits) << 32) | uint64_t(outColor);
    atomicMin(GetResource(bDepth,pConst.depthBufferId).data[offset],zbufValue);
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
    uint gtid = gl_LocalInvocationID.x;
    uvec2 triRefOffset = readTriangleIndexRefOffset(mi,meshletRef,obj);

    uint clusterId = getClusterID();

    if(gtid < totalVerts){
        uint i = gtid;
        uint vi = readVertexIndex(mi,meshletRef,obj,i);
        vec3 v0 = GetResource(bVertices,vertexRef).data[vi].xyz;
        vec4 final= mvp * vec4(v0,1.0);
        sharedVertices[i] = final;
    }
    barrier();
    if(gtid<totalTris){
        uint triIndices = readTriangleIndex2(triRefOffset,gtid);
        uint triIndexA = triIndices & 0x000000FF;
        uint triIndexB = (triIndices & 0x0000FF00) >> 8;
        uint triIndexC = (triIndices & 0x00FF0000) >> 16;

        vec4 v0 = sharedVertices[triIndexA];
        vec4 v1 = sharedVertices[triIndexB];
        vec4 v2 = sharedVertices[triIndexC];

        // Start of sw rasterization
        vec2 v0xy = v0.xy / v0.w;
        vec2 v1xy = v1.xy / v1.w;
        vec2 v2xy = v2.xy / v2.w;

        vec2 uv0 = v0xy * 0.5 + 0.5;
        vec2 uv1 = v1xy * 0.5 + 0.5;
        vec2 uv2 = v2xy * 0.5 + 0.5;

        vec2 coord0 = uv0 * vec2(pConst.renderWidth, pConst.renderHeight)+0.5;
        vec2 coord1 = uv1 * vec2(pConst.renderWidth, pConst.renderHeight)+0.5;
        vec2 coord2 = uv2 * vec2(pConst.renderWidth, pConst.renderHeight)+0.5;

        uint minX = uint(min(min(coord0.x, coord1.x), coord2.x))-1;
        uint maxX = uint(max(max(coord0.x, coord1.x), coord2.x))+1;
        uint minY = uint(min(min(coord0.y, coord1.y), coord2.y))-1;
        uint maxY = uint(max(max(coord0.y, coord1.y), coord2.y))+1;

        minX = clamp(minX, 0, pConst.renderWidth);
        maxX = clamp(maxX, 0, pConst.renderWidth);
        minY = clamp(minY, 0, pConst.renderHeight);
        maxY = clamp(maxY, 0, pConst.renderHeight);

        vec3 rcpW = 1.0 / vec3(v0.w, v1.w, v2.w);
        vec3 p0 = v0.xyz * rcpW.x;
        vec3 p1 = v1.xyz * rcpW.y;
        vec3 p2 = v2.xyz * rcpW.z;

        vec3 p120x = vec3(p1.x, p2.x, p0.x);
        vec3 p120y = vec3(p1.y, p2.y, p0.y);
        vec3 p201x = vec3(p2.x, p0.x, p1.x);
        vec3 p201y = vec3(p2.y, p0.y, p1.y);

        vec2 uvNdcMin = vec2(float(minX) / float(pConst.renderWidth), float(minY) / float(pConst.renderHeight));
        uvNdcMin = uvNdcMin * 2.0 - 1.0;

        vec3 cdx = p201y - p120y;
        vec3 cdy = p120x - p201x;
        
        float v0az = v0.z / v0.w;
        float v1az = v1.z / v1.w;
        float v2az = v2.z / v2.w;

        float invRw = 1.0 / float(pConst.renderWidth);
        float invRh = 1.0 / float(pConst.renderHeight);
        vec3 base = -cdx*p120x - cdy*p120y;
        for(uint y = minY; y <= maxY; y++){
            for(uint x = minX; x <= maxX; x++){
                vec2 uvNdc = vec2((float(x)+0.5) * invRw, (float(y)+0.5) * invRh) * 2.0 - 1.0;
                vec3 a = cdx * uvNdc.x + cdy*uvNdc.y + base;
                vec3 b = a*rcpW;
                float h = dot(a,rcpW);
                float rcpH = 1.0/h;
                vec3 bary = b*rcpH;
                if(bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0){
                    float depth = dot(bary,vec3(v0az,v1az,v2az));
                    writePixel(uvec2(x, y), gtid, depth);
                }
            }
        }
    }
}