
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
#extension GL_GOOGLE_include_directive : enable

#include "Base.glsl"
#include "Bindless.glsl"
#include "Deferred.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

layout(local_size_x = cEmitGbufThreadGroupSizeX, local_size_y = 1, local_size_z = 1) in;

#include "Syaro/Syaro.EmitGBuffer.Shared.glsl"

void main(){
#if SYARO_SHADER_SHARED_EMIT_GBUFFER_TRIANGLE_REUSE
    gbcomp_TriangleDataShared triDataShared;
    uvec2 lastClusterTri = uvec2(~0);
    uint gBuffer = gbcomp_GetGBufferId();
    GBufferPixel pixelData;
    gbcomp_TriangleData triData;
    uvec2 clusterTri;

    uvec2 totalPxMatOffset = gbcomp_GetTotalPixelsOffset();

    // Loop Iter 0
    uvec2 px0 = gbcomp_GetPixelReused(0,totalPxMatOffset);
    clusterTri = gbcomp_GetVisBufferData(px0);
    triDataShared = gbcomp_GetTriangleDataImp(clusterTri,px0);
    lastClusterTri = clusterTri;
    triData = gbcomp_GetTriangleDataReused(triDataShared,clusterTri,px0);
    pixelData.albedo = triData.vAlbedo.xyz;
    pixelData.normal = triData.vpNormalVS * 0.5 + 0.5;
    gbcomp_WriteGBufferAlbedoNormal(gBuffer, px0, pixelData);

    // Loop Iter 1
    uvec2 px1 = gbcomp_GetPixelReused(1,totalPxMatOffset);
    if(px1.x==~0) return;
    clusterTri = gbcomp_GetVisBufferData(px1);
    if(lastClusterTri!=clusterTri){
        triDataShared = gbcomp_GetTriangleDataImp(clusterTri,px1);
        lastClusterTri = clusterTri;
    }
    triData = gbcomp_GetTriangleDataReused(triDataShared,clusterTri,px1);
    pixelData.albedo = triData.vAlbedo.xyz;
    pixelData.normal = triData.vpNormalVS * 0.5 + 0.5;
    gbcomp_WriteGBufferAlbedoNormal(gBuffer, px1, pixelData);

    // Loop Iter 2
    uvec2 px2 = gbcomp_GetPixelReused(2,totalPxMatOffset);
    if(px2.x==~0) return;
    clusterTri = gbcomp_GetVisBufferData(px2);
    if(lastClusterTri!=clusterTri){
        triDataShared = gbcomp_GetTriangleDataImp(clusterTri,px2);
        lastClusterTri = clusterTri;
    }
    triData = gbcomp_GetTriangleDataReused(triDataShared,clusterTri,px2);
    pixelData.albedo = triData.vAlbedo.xyz;
    pixelData.normal = triData.vpNormalVS * 0.5 + 0.5;
    gbcomp_WriteGBufferAlbedoNormal(gBuffer, px2, pixelData);

    // Loop Iter 3
    uvec2 px3 = gbcomp_GetPixelReused(3,totalPxMatOffset);
    if(px3.x==~0) return;
    clusterTri = gbcomp_GetVisBufferData(px3);
    if(lastClusterTri!=clusterTri){
        triDataShared = gbcomp_GetTriangleDataImp(clusterTri,px3);
        lastClusterTri = clusterTri;
    }
    triData = gbcomp_GetTriangleDataReused(triDataShared,clusterTri,px3);
    pixelData.albedo = triData.vAlbedo.xyz;
    pixelData.normal = triData.vpNormalVS * 0.5 + 0.5;
    gbcomp_WriteGBufferAlbedoNormal(gBuffer, px3, pixelData);

#else
    uvec2 px = gbcomp_GetPixel();
    if(px.x==~0){
        return;
    }
    uint gBuffer = gbcomp_GetGBufferId();
    uvec2 clusterTri = gbcomp_GetVisBufferData(px);

    gbcomp_TriangleData triData = gbcomp_GetTriangleData(clusterTri,px);

    GBufferPixel pixelData;
    pixelData.albedo = triData.vAlbedo.xyz;
    pixelData.normal = vec3(triData.vpNormalVS.xyz) * 0.5 + 0.5;

    gbcomp_WriteGBufferAlbedoNormal(gBuffer, px, pixelData);
#endif
}