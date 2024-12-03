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