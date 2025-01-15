
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
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

layout(local_size_x = cCombineVisBufferThreadGroupSizeX, local_size_y = cCombineVisBufferThreadGroupSizeY, local_size_z = 1) in;

layout(push_constant) uniform CombineVisBufferPushConstant{
    uint hwVisBufferId;
    uint hwDepthBufferId; // This is depth image
    uint swVisBufferId;
    uint swDepthBufferId; // This is just a ssbo
    uint rtWidth;
    uint rtHeight;
    uint outVisBufferId;
    uint outDepthBufferId; // color image
    uint outputMode; // 0-default, 1-sw/hw coloring
}pc;

RegisterStorage(bDepth,{
    uint64_t data[];
});

void main(){
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if(x>=pc.rtWidth || y>=pc.rtHeight) return;

    uint offset = y*pc.rtWidth + x;
    float hwDepth = texelFetch(GetSampler2D(pc.hwDepthBufferId), ivec2(x,y), 0).r;
    float swDepth = 1e9;
    uint swVis = 0;
    uint hwVis = 0;
    if(pc.swDepthBufferId!=~0){
        uint64_t packedZBuffer = GetResource(bDepth, pc.swDepthBufferId).data[offset];
        uint visBufferVal = uint(packedZBuffer & 0xFFFFFFFF);
        uint depthVal = uint((packedZBuffer>>32) & 0xFFFFFFFF);
        float depth = float(depthVal) / 4294967295.0;
        swDepth = depth;
        swVis = visBufferVal;
    } 

    if(pc.hwVisBufferId!=~0){
        hwVis = imageLoad(GetUAVImage2DR32UI(pc.hwVisBufferId), ivec2(x,y)).r;
    }

    float maxDepth = min(hwDepth, swDepth); 
    // Test: force output all sw, if swDepth is less than 1
    // if(swDepth<1.0){
    //     if(pc.outVisBufferId!=~0) imageStore(GetUAVImage2DR32UI(pc.outVisBufferId), ivec2(x,y), uvec4(swVis));
    //     imageStore(GetUAVImage2DR32F(pc.outDepthBufferId), ivec2(x,y), vec4(swDepth));
    //     return;
    // }

    if(maxDepth>=1.0){
        // discard this pixel, set all to 0, and depth to 1.0
        if(pc.outVisBufferId!=~0) imageStore(GetUAVImage2DR32UI(pc.outVisBufferId), ivec2(x,y), uvec4(0));
        imageStore(GetUAVImage2DR32F(pc.outDepthBufferId), ivec2(x,y), vec4(1.0));
        return;
    }
    if(pc.outputMode==1){
        if(swDepth<hwDepth){
            if(pc.outVisBufferId!=~0) imageStore(GetUAVImage2DR32UI(pc.outVisBufferId), ivec2(x,y), uvec4(2));
            imageStore(GetUAVImage2DR32F(pc.outDepthBufferId), ivec2(x,y), vec4(swDepth));
        }else{
            if(pc.outVisBufferId!=~0) imageStore(GetUAVImage2DR32UI(pc.outVisBufferId), ivec2(x,y), uvec4(3));
            imageStore(GetUAVImage2DR32F(pc.outDepthBufferId), ivec2(x,y), vec4(hwDepth));
        }
    }else{
        if(swDepth<hwDepth){
            if(pc.outVisBufferId!=~0) imageStore(GetUAVImage2DR32UI(pc.outVisBufferId), ivec2(x,y), uvec4(swVis));
            imageStore(GetUAVImage2DR32F(pc.outDepthBufferId), ivec2(x,y), vec4(swDepth));
        }else{
            if(pc.outVisBufferId!=~0) imageStore(GetUAVImage2DR32UI(pc.outVisBufferId), ivec2(x,y), uvec4(hwVis));
            imageStore(GetUAVImage2DR32F(pc.outDepthBufferId), ivec2(x,y), vec4(hwDepth));
        }
    }
    
    
}

