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
    uvec2 px = gbcomp_GetPixel();
    if(px.x==~0){
        return;
    }
    uint gBuffer = gbcomp_GetGBufferId();
    uvec2 clusterTri = gbcomp_GetVisBufferData(px);

    gbcomp_TriangleData triData = gbcomp_GetTriangleData(clusterTri,px);

    GBufferPixel pixelData;
    pixelData.albedo = vec3(1.0,1.0,1.0) * 5.0;
    pixelData.normal = vec3(triData.vpNormalVS.xyz) * 0.5 + 0.5;

    gbcomp_WriteGBuffer(gBuffer, px, pixelData);
}