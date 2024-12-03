#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

// Material Pass / Scatter Pass
// This pass is unoptimized. It just do scattering.
// Optimizations mentioned in GDC 2024 slides are not implemented.

layout(local_size_x = cClassifyMaterialScatterThreadGroupSizeX, local_size_y = cClassifyMaterialScatterThreadGroupSizeY, local_size_z = 1) in;

#include "Syaro/Syaro.ClassifyMaterial.Shared.glsl"

void main(){
    uint tX = gl_GlobalInvocationID.x;
    uint tY = gl_GlobalInvocationID.y;
    
    if(2 * tX >= uMaterialPassPushConstant.renderWidth || 2 * tY >= uMaterialPassPushConstant.renderHeight){
        return;
    }
    for(uint i = 0; i < 4; i++){
        uint x = 2 * tX + i % 2;
        uint y = 2 * tY + i / 2;
        if(x >= uMaterialPassPushConstant.renderWidth || y >= uMaterialPassPushConstant.renderHeight){
            continue;
        }
        uint pixelId = x + y * uMaterialPassPushConstant.renderWidth;
        //uint materialId =  texelFetch(GetSampler2D(uMaterialPassData.materialDepthRef), ivec2(int(x),int(y)), 0).a;
        float materialIdFloat = imageLoad(GetUAVImage2DRGBA32F(uMaterialPassData.materialDepthRef), ivec2(x, y)).a;
        uint materialId = uint(materialIdFloat);
        if(materialId == 0){
            continue;
        }
        materialId--;
        uint offset = GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[materialId].offset;
        uint counter = GetResource(bPerPixelCounterOffset, uMaterialPassData.perPixelCounterOffsetRef).data[pixelId];
        uint address = offset + counter;
        GetResource(bMaterialPixelList, uMaterialPassData.materialPixelListRef).data[address] = pixelId;
    }
}