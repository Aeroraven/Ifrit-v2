#version 450
#extension GL_EXT_mesh_shader: require
#extension GL_GOOGLE_include_directive: require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

layout(local_size_x = cMeshRasterizeTaskThreadGroupSize, local_size_y = 1, local_size_z = 1) in;

// According to NVIDIA's documentation, AS's payload should be small
// to avoid perf penalty.
// https://developer.nvidia.com/blog/advanced-api-performance-mesh-shaders/
struct TaskSharedData{
    uint base;
    uint subIds[cMeshRasterizeTaskPayloadSize];
};

taskPayloadSharedEXT  TaskSharedData taskSharedData;

RegisterStorage(bMeshlet,{
    Meshlet data[];
});


RegisterStorage(bFilteredMeshlets2,{
    uvec2 data[];
});

RegisterStorage(bDrawCallSize,{
    uint x2;
    uint y2;
    uint z2; 
    uint x1;
    uint y1;
    uint z1;
    uint completedWorkGroups1;
    uint completedWorkGroups2;
    uint meshletsToDraw1;
    uint meshletsToDraw2; 
    uint pad1;
    uint pad2;
});

layout(binding = 0, set = 1) uniform PerframeViewData{
    uint refCurFrame;
    uint refPrevFrame;
}uPerframeView;

layout(binding = 0, set = 2) uniform InstanceData{
    uvec4 ref;
}uInstanceData;


layout(binding = 0, set = 3) uniform IndirectDrawData2{
    uint allMeshletsRef;
    uint indDrawCmdRef;
}uIndirectDrawData2;

layout(push_constant) uniform CullingPass{
    uint passNo;
} pConst;

shared uint sAcceptedMeshlets;

bool isSecondCullingPass(){
    return pConst.passNo == 1;
}

uint getClusterID(){
    uint actualWorkGroup = gl_GlobalInvocationID.x;
    if(!isSecondCullingPass()){
        return actualWorkGroup;
    }else{
        return GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1 + actualWorkGroup;
    }
}

uint getObjId(){
    uint actualWorkGroup = gl_GlobalInvocationID.x;
    if(!isSecondCullingPass()){
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[actualWorkGroup].x;
    }else{
        uint baseOffset = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1;
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[baseOffset + actualWorkGroup].x;
    }
}

uint getMeshletId(){
    uint actualWorkGroup = gl_GlobalInvocationID.x;
    if(!isSecondCullingPass()){
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[actualWorkGroup].y;
    }else{
        uint baseOffset = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1;
        return GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[baseOffset + actualWorkGroup].y;
    }
}

uint getTotalMeshletsToDraw(){
    if(isSecondCullingPass()){
        return GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw2;
    }else{
        return GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1;
    }
}

bool frustumCullLRTB(vec4 left, vec4 right, vec4 top, vec4 bottom, vec4 boundBall, float radius){
    float distLeft = ifrit_signedDistToPlane(left,boundBall);
    float distRight = ifrit_signedDistToPlane(right,boundBall);
    float distTop = ifrit_signedDistToPlane(top,boundBall);
    float distBottom = ifrit_signedDistToPlane(bottom,boundBall);

    if(distLeft + radius < 0.0 || distRight + radius < 0.0 || distTop + radius < 0.0 || distBottom + radius < 0.0){
        return true;
    }
    return false;
}

// If the object should be culled, return true
bool frustumCull(vec4 boundBall, float radius, float tanHalfFovY){
    float camFar = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraFar;
    float camNear = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraNear;
    
    float camAspect = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraAspect;
    float z = boundBall.z;
    if(z+radius < camNear || z-radius > camFar){
        return true;
    }
    vec4 leftNorm = normalize(vec4(1.0, 0.0,tanHalfFovY * camAspect, 0.0));
    vec4 rightNorm = normalize(vec4(-1.0, 0.0,tanHalfFovY * camAspect, 0.0));
    vec4 topNorm = normalize(vec4(0.0, -1.0,tanHalfFovY, 0.0));
    vec4 bottomNorm = normalize(vec4(0.0, 1.0,tanHalfFovY, 0.0));
    if(frustumCullLRTB(leftNorm,rightNorm,topNorm,bottomNorm,boundBall,radius)){
        return true;
    }
    return false;
}


void main(){
    // I make meshlet culling from persist cull stage into task shader.
    // It's expected that all warps in a workgroup are doing the same thing.
    // Place the culling code in persist cull stage makes many threads idle.
    // And some specific data (like cone apex) should be read, which introduces
    // LSB stalls.

    // This subjects to change. With 2 reasons:
    // 1. Almost no performance gain
    // 2. Cannot be integrated into SW rasterizer

    if(gl_LocalInvocationIndex == 0){
        sAcceptedMeshlets = 0;
        taskSharedData.base = gl_GlobalInvocationID.x;
    }
    if(gl_LocalInvocationID.x < cMeshRasterizeTaskPayloadSize){
        taskSharedData.subIds[gl_LocalInvocationID.x] = 0;
    }
    barrier();
    uint meshSlot;
    
    uint globalInvo = gl_GlobalInvocationID.x;
    if(globalInvo < getTotalMeshletsToDraw()){
        bool isAccepted = true;
        uint objId = getObjId();
        uint meshletId = getMeshletId();
        uint clusterId = getClusterID();

        uint trans = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].transformRef;
        mat4 model = GetResource(bLocalTransform,trans).m_localToWorld;
        mat4 worldToView = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_worldToView;
        float fovy = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraFovY;
        float tanHalfFovY = tan(fovy * 0.5);
        mat4 mv = worldToView * model;

        uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
        uint meshletRef =  GetResource(bMeshDataRef,obj).meshletBuffer;
        vec4 normalConeAxis = GetResource(bMeshlet,meshletRef).data[meshletId].normalCone;
        vec4 normalConeApex = GetResource(bMeshlet,meshletRef).data[meshletId].normalConeApex;

        // TODO: this transform is incorrect, but the demo scene only uses translation, it's fine for now
        vec4 viewConeAxis = mv * vec4(normalConeAxis.xyz,0.0);
        vec4 viewConeApex = mv * vec4(normalConeApex.xyz,1.0);
        float coneAngle = dot(viewConeAxis.xyz,viewConeApex.xyz);
        if(coneAngle > normalConeAxis.w){
            isAccepted = false;
        }

        if(isAccepted){
            vec4 boundBall = GetResource(bMeshlet,meshletRef).data[meshletId].boundSphere;
            vec4 boundBallCenter = vec4(boundBall.xyz,1.0); 
            float boundBallRadius = boundBall.w;

            vec4 viewBoundBallCenter = mv * boundBallCenter;
            if(frustumCull(viewBoundBallCenter,boundBallRadius,tanHalfFovY)){
                isAccepted = false;
            }
        }
        if(isAccepted){
            meshSlot = atomicAdd(sAcceptedMeshlets,1);
            uint slotIndex = meshSlot / 4;
            uint slotBits = (meshSlot % 4) * 8;
            uint addVal = (gl_LocalInvocationID.x << slotBits);
            atomicOr(taskSharedData.subIds[slotIndex],addVal);
        }
    }
    barrier();

    if(gl_LocalInvocationIndex == 0){
        EmitMeshTasksEXT(sAcceptedMeshlets,1,1);
    }
}