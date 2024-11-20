#version 450
#extension GL_GOOGLE_include_directive : enable

#include "Base.glsl"
#include "Bindless.glsl"
#include "Deferred.glsl"
#include "Syaro.Shared.glsl"


layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "Syaro.EmitGBuffer.Shared.glsl"

void main(){
    uvec2 px = gbcomp_GetPixel();
    uint gBuffer = gbcomp_GetGBufferId();
    uvec2 clusterTri = gbcomp_GetVisBufferData(px);

    gbcomp_TriangleData triData = gbcomp_GetTriangleData(clusterTri,px);

    GBufferPixel pixelData;
    pixelData.albedo = vec3(triData.vpNormalVS.xyz);

    gbcomp_WriteGBuffer(gBuffer, px, pixelData);
}