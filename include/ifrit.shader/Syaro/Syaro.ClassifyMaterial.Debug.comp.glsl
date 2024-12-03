#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "Syaro/Syaro.ClassifyMaterial.Shared.glsl"

void main(){
    uint offset = GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[0].offset;
    uint tX = gl_GlobalInvocationID.x;
    uint address = offset + tX;

    uint pixelId = GetResource(bMaterialPixelList, uMaterialPassData.materialPixelListRef).data[address];
    uint pX = pixelId % uMaterialPassPushConstant.renderWidth;
    uint pY = pixelId / uMaterialPassPushConstant.renderWidth;

    imageStore(GetUAVImage2DRGBA32F(uMaterialPassData.debugImageRef), ivec2(pX, pY), vec4(1.0, 0.0, 0.0, 1.0));
}