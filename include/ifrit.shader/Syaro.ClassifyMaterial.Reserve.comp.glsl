#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro.Shared.glsl"

// Material Pass / Reserving Pass

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

#include "Syaro.ClassifyMaterial.Shared.glsl"

void main(){
    uint materialId = gl_GlobalInvocationID.x;
    if(materialId >= uMaterialPassPushConstant.totalMaterials){
        return;
    }
    uint thisMaterialCounter = GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[materialId].counter;
    uint offset = atomicAdd(GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).totalCounter, thisMaterialCounter);
    GetResource(bMaterialCounter, uMaterialPassData.materialCounterRef).data[materialId].offset = offset;

    // Write indirect command, for simplicity one pixel one workgroup (slow, might be optimized later)
    GetResource(bMaterialPassIndirectCommand, uMaterialPassData.indirectCommandRef).data[materialId] = MaterialPassIndirectCommand(thisMaterialCounter,1,1);
}