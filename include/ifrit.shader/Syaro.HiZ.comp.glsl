#version 450
#extension GL_GOOGLE_include_directive : require

// Hierrachical Z-Buffer

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro.Shared.glsl"

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

RegisterStorage(bHiZData,{
    uint width;
    uint height;
});

layout(binding = 0, set = 1) uniform HiZData{
    uint depthImg;
    uint srcImg;
    uint dstImg;
    uint hizSzRef;
}uHiZData;

void main(){
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    uint width = GetResource(bHiZData,uHiZData.hizSzRef).width;
    uint height = GetResource(bHiZData,uHiZData.hizSzRef).height;
    if(x >= width || y >= height){
        return;
    }
    if(uHiZData.srcImg == uHiZData.dstImg){
        // Copy the source image to the destination image. 
        // I think this step could be optimized. However, depth RTs cannot
        // be written in compute shaders.
        float depth = texelFetch(GetSampler2D(uHiZData.depthImg),ivec2(x,y),0).r;
        imageStore(GetUAVImage2DR32F(uHiZData.dstImg),ivec2(x,y),vec4(depth,0.0,0.0,0.0));
    }else{
        // Just find the minimum depth each 2x2 block.
        float d1 = imageLoad(GetUAVImage2DR32F(uHiZData.srcImg),ivec2(x*2,y*2)).r;
        float d2 = imageLoad(GetUAVImage2DR32F(uHiZData.srcImg),ivec2(x*2+1,y*2)).r;
        float d3 = imageLoad(GetUAVImage2DR32F(uHiZData.srcImg),ivec2(x*2,y*2+1)).r;
        float d4 = imageLoad(GetUAVImage2DR32F(uHiZData.srcImg),ivec2(x*2+1,y*2+1)).r;
        float minDepth = min(min(d1,d2),min(d3,d4));
        imageStore(GetUAVImage2DR32F(uHiZData.dstImg),ivec2(x,y),vec4(minDepth,0.0,0.0,0.0));
    }
}