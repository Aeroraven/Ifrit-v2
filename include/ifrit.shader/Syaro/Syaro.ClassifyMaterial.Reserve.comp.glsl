
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

// Material Pass / Reserving Pass

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

#include "Syaro/Syaro.ClassifyMaterial.Shared.glsl"

void main(){
    uint materialId = gl_GlobalInvocationID.x;
    if(materialId >= uMaterialPassPushConstant.totalMaterials){
        return;
    }
    uint thisMaterialCounter = GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[materialId].counter;
    uint offset = atomicAdd(GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).totalCounter, thisMaterialCounter);
    GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[materialId].offset = offset;

    // DONE: Write indirect command, for simplicity one pixel one workgroup (slow, might be optimized later)
    uint totalTGX = (thisMaterialCounter + cEmitGbufThreadGroupSizeX - 1)/cEmitGbufThreadGroupSizeX;
    GetResource(bMaterialPassIndirectCommand, uMaterialPassData.indirectCommandRef).data[materialId].x = totalTGX;
    GetResource(bMaterialPassIndirectCommand, uMaterialPassData.indirectCommandRef).data[materialId].y = 1;
    GetResource(bMaterialPassIndirectCommand, uMaterialPassData.indirectCommandRef).data[materialId].z = 1;
}