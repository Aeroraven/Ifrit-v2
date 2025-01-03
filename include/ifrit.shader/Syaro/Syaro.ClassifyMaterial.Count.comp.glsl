
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

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

// Material Pass / Counting Pass
// [Unlike HLSL, GLSL only have ONE entry point for compute shader, so,
// the entry point for each pass is defined by the main function name.]
//
// I do not think this implementation is correct. 
// The general idea is to maintain a list for each material type.
// 
// GAMES104's slide shows that a (NumMaterials x NumTiles) texture is used.
// However, the location sparsity of material occurence mentioned in Nanite's
// SIGGRAPH 2021 and GDC 2024 slides is not considered here, making a huge waste
// of memory.
// Some one mentioned that Material Range is used in Nanite, but I do not know
// what exactly the 64bit Material Range is. If it's exactly range, there might be
// something that can be optimized. For materials id in the middle, this implementation
// makes extra depth test for material depth.
//
// The count pass here count materials referenced by pixels, incrementing the 
// global counter for each material.
// The reserve pass here reserve space for each material, and write the offset
// to the global counter.
// The scatter pass here scatter the pixel to the correct position.

// Assumptions: NumMaterials <= NumScreenPixels
// And one material requires 1x uint for counter and 1x uint for offset.
// Total memory usage: uint3 * NumScreenPixels 
// For simplicity, larger memory is used in this implementation.

layout(local_size_x = cClassifyMaterialCountThreadGroupSizeX, local_size_y = cClassifyMaterialCountThreadGroupSizeY, local_size_z = 1) in;

#include "Syaro/Syaro.ClassifyMaterial.Shared.glsl"

void main(){
    uint tX = gl_GlobalInvocationID.x;
    uint tY = gl_GlobalInvocationID.y;
    
    // A thread is responsible for a 2x2 quad
    if(2*tX >= uMaterialPassPushConstant.renderWidth || 2*tY >= uMaterialPassPushConstant.renderHeight){
        return;
    }
    for(uint i = 0; i < 2; i++){
        for(uint j = 0; j < 2; j++){
            uvec2 pos = uvec2(2*tX+i, 2*tY+j);
            if(pos.x >= uMaterialPassPushConstant.renderWidth || pos.y >= uMaterialPassPushConstant.renderHeight){
                continue;
            }
            // Percision might be a problem here
            // float materialIdFloat = texelFetch(GetSampler2D(uMaterialPassData.materialDepthRef), ivec2(pos), 0).a;
            float materialIdFloat = imageLoad(GetUAVImage2DRGBA32F(uMaterialPassData.materialDepthRef), ivec2(pos)).a;

            uint materialId = uint(materialIdFloat);
            if(materialId == 0){
                uint posId = pos.y * uMaterialPassPushConstant.renderWidth + pos.x;
                GetResource(bPerPixelCounterOffset, uMaterialPassData.perPixelCounterOffsetRef).data[posId] = ~0u;
                continue;
            }
            materialId = materialId - 1;
            uint inMaterialCounterOffset = atomicAdd(GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[materialId].counter, 1);
            uint posId = pos.y * uMaterialPassPushConstant.renderWidth + pos.x;
            GetResource(bPerPixelCounterOffset, uMaterialPassData.perPixelCounterOffsetRef).data[posId] = inMaterialCounterOffset;
        }
    }
}