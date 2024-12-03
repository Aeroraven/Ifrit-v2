
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
#extension GL_GOOGLE_include_directive : require

// Hierarchical Z-Buffer, in a single pass
// The algorithm takes following steps. For lower mip levels, a thread group
// takes a 64x64 tile, generating mips in a thread-synced way. 
// When a thread group finishes, an global atomic counter is incremented.
// The last thread groups then generate the remaining mips levels.
// It supports a texture width at most 4096.
//
// Some implementation might be incorrect, it includes extra barriers
// please refer to the original post.
//
// References:
// Single-pass Downsampler. https://gpuopen.com/fidelityfx-spd/

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

layout(local_size_x = cHiZThreadGroupSize, local_size_y = 1, local_size_z = 1) in;

RegisterStorage(bHiZStorage,{
    uint pad;
    uint mipRefs[];
});

RegisterStorage(bHizAtomics,{
    uint globalAtomics;
});

layout(binding = 0, set = 1) uniform HiZData{
    uint depthImg; // Depth image, with samplers
    uint hizRefs; // Reference to image views, UAVs 
    uint hizAtomics; // Atomic counter
}uHiZData;

layout(push_constant) uniform HiZPushConstant{
    uint width;
    uint height;
    uint depthWidth;
    uint depthHeight;
    uint mipLevels;
}uHiZPushConstant;

uint mortonDecodeX(uint code){
    // Get x from Morton code
    // https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
    uint x = code & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
}
uvec2 mortonDecode(uint code){
    // Get (x,y) from Morton code
    return uvec2(mortonDecodeX(code),mortonDecodeX(code >> 1));
}

float depthFetch(uint x,uint y){
    if(x >= uHiZPushConstant.depthWidth || y >= uHiZPushConstant.depthHeight){
        return 1.0;
    }
    return texelFetch(GetSampler2D(uHiZData.depthImg),ivec2(x,y),0).r;
}

shared uint sharedCounter;

void main(){
    uint tgX = gl_WorkGroupID.x;
    uint tgY = gl_WorkGroupID.y;
    uint tX = gl_LocalInvocationID.x;

    // mips 0: 4096x4096, 1: 2048x2048, 2: 1024x1024, 3: 512x512, 4: 256x256, 5: 128x128, 6: 64x64
    // mips 0: 64x64, 1: 32x32, 2: 16x16, 3: 8x8, 4: 4x4, 5: 2x2, 6: 1x1

    uint firstPartMipLvl = min(uHiZPushConstant.mipLevels, 6u);
    for(uint i=0;i<firstPartMipLvl;++i){
        // a thread is responsible for 2x2 pixels.
        uint tileSize = 1u << (6u-i);
        uint blockSize = 32u;
        uvec2 inTileXY = mortonDecode(tX*4);
        uint numBlocksX = max(1,tileSize/blockSize);
        if(inTileXY.x >= tileSize || inTileXY.y >= tileSize){
            // Do nothing
        }else{
            for(uint fx=0;fx<numBlocksX;fx++){
                for(uint fy=0;fy<numBlocksX;fy++){
                    uint tileX = tgX * tileSize + fx * blockSize + inTileXY.x;
                    uint tileY = tgY * tileSize + fy * blockSize + inTileXY.y;
                    
                    // fetch 4 pixels
                    if(i==0){
                        float d1 = depthFetch(tileX,tileY);
                        float d2 = depthFetch(tileX+1,tileY);
                        float d3 = depthFetch(tileX,tileY+1);
                        float d4 = depthFetch(tileX+1,tileY+1);

                        float maxDepth = max(max(d1,d2),max(d3,d4));

                        // for compatibility, we store the depth in the first mip level.
                        imageStore(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX,tileY),vec4(d1,0.0,0.0,0.0));
                        imageStore(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX+1,tileY),vec4(d2,0.0,0.0,0.0));
                        imageStore(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX,tileY+1),vec4(d3,0.0,0.0,0.0));
                        imageStore(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX+1,tileY+1),vec4(d4,0.0,0.0,0.0));

                        // Mip0 -> Mip1
                        imageStore(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i+1]),ivec2(tileX/2,tileY/2),vec4(maxDepth,0.0,0.0,0.0));
                    }else{
                        float d1 = imageLoad(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX,tileY)).r;
                        float d2 = imageLoad(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX+1,tileY)).r;
                        float d3 = imageLoad(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX,tileY+1)).r;
                        float d4 = imageLoad(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX+1,tileY+1)).r;

                        float maxDepth = max(max(d1,d2),max(d3,d4));
                        imageStore(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i+1]),ivec2(tileX/2,tileY/2),vec4(maxDepth,0.0,0.0,0.0));
                    }
                }
            }
        }
        barrier();
    }
    bool isFirstLane = (tX == 0);
    if(isFirstLane){
        sharedCounter = atomicAdd(GetResource(bHizAtomics,uHiZData.hizAtomics).globalAtomics,1)+1;
    }
    barrier();
    uint totalTiles = uHiZPushConstant.width * uHiZPushConstant.height / (64*64);
    if(sharedCounter != totalTiles){
        return;
    }
    if(sharedCounter == 0){
        imageStore(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[7]),ivec2(15,15),vec4(1.0,0.0,0.0,0.0));
    }

    // Last thread group, generate remaining mips
    for(uint i=6;i<uHiZPushConstant.mipLevels;++i){
        uint mipWidth = uHiZPushConstant.width >> i;
        uint mipHeight = uHiZPushConstant.height >> i;
        uint blockSize = 32u;
        uint numBlocksX = max(1,mipWidth/blockSize);
        uint numBlocksY = max(1,mipHeight/blockSize);
        uvec2 inTileXY = mortonDecode(tX*4);
        if(inTileXY.x >= mipWidth || inTileXY.y >= mipHeight){
            // do nothing
        }else{
            for(uint fx=0;fx<numBlocksX;fx++){
                for(uint fy=0;fy<numBlocksY;fy++){
                    uint tileX = fx * blockSize + inTileXY.x;
                    uint tileY = fy * blockSize + inTileXY.y;

                    uint tileX2 = min(tileX +1,mipWidth-1);
                    uint tileY2 = min(tileY +1,mipHeight-1);
                    float d1 = imageLoad(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX,tileY)).r;
                    float d2 = imageLoad(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX2,tileY)).r;
                    float d3 = imageLoad(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX,tileY2)).r;
                    float d4 = imageLoad(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i]),ivec2(tileX2,tileY2)).r;

                    float maxDepth = max(max(d1,d2),max(d3,d4));
                    imageStore(GetUAVImage2DR32F(GetResource(bHiZStorage,uHiZData.hizRefs).mipRefs[i+1]),ivec2(tileX/2,tileY/2),vec4(maxDepth,0.0,0.0,0.0));
                }
            }
        }
        barrier();
    }
    if(isFirstLane){
        atomicExchange(GetResource(bHizAtomics,uHiZData.hizAtomics).globalAtomics,0);
    }
}