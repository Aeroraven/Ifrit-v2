#version 450
#extension GL_GOOGLE_include_directive : require

// Instance culling typically sends the instance's root BVH node
// to the persistent culling stage.
// It's said that instance culling uses mesh as the smallest unit of culling.
// From: https://www.reddit.com/r/unrealengine4/comments/sycyof/analysis_of_ue5_rendering_technology_nanite/

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro.Shared.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

RegisterStorage(bInstanceAccepted,{
    uint data[];
});
RegisterStorage(bInstanceRejected,{
    uint data[];
});
RegisterStorage(bHierCullDispatch,{
    uint accepted;
    uint compY;
    uint compZ;
    uint rejected;
});

layout(binding = 0, set = 1) uniform PerframeViewData{
    uvec4 ref;
}uPerframeView;

layout(binding = 0, set = 2) uniform InstanceData{
    uvec4 ref;
}uInstanceData;

layout(binding = 0, set = 3) uniform IndirectCompData{
    uint acceptRef;
    uint rejectRef;
    uint indRef;
    uint pad;
}uIndirectComp;

float signedDistToPlane(vec4 plane, vec4 point){
    return dot(plane.xyz,point.xyz) + plane.w;
}

bool frustumCullLRTB(vec4 left, vec4 right, vec4 top, vec4 bottom, vec4 boundBall, float radius){
    float distLeft = signedDistToPlane(left,boundBall);
    float distRight = signedDistToPlane(right,boundBall);
    float distTop = signedDistToPlane(top,boundBall);
    float distBottom = signedDistToPlane(bottom,boundBall);

    if(distLeft + radius < 0.0 || distRight + radius < 0.0 || distTop + radius < 0.0 || distBottom + radius < 0.0){
        return true;
    }
    return false;
}

// If the object should be culled, return true
bool frustumCull(vec4 boundBall, float radius){
    float camFar = GetResource(bPerframeView,uPerframeView.ref.x).data.m_cameraFar;
    float camNear = GetResource(bPerframeView,uPerframeView.ref.x).data.m_cameraNear;
    float camFovY = GetResource(bPerframeView,uPerframeView.ref.x).data.m_cameraFovY;
    float halfFovY = camFovY * 0.5;
    float camAspect = GetResource(bPerframeView,uPerframeView.ref.x).data.m_cameraAspect;
    float z = boundBall.z;
    if(z+radius < camNear || z-radius > camFar){
        return true;
    }
    vec4 leftNorm = normalize(vec4(1.0, 0.0,tan(halfFovY) * camAspect, 0.0));
    vec4 rightNorm = normalize(vec4(-1.0, 0.0,tan(halfFovY) * camAspect, 0.0));
    vec4 topNorm = normalize(vec4(0.0, -1.0,tan(halfFovY), 0.0));
    vec4 bottomNorm = normalize(vec4(0.0, 1.0,tan(halfFovY), 0.0));
    if(frustumCullLRTB(leftNorm,rightNorm,topNorm,bottomNorm,boundBall,radius)){
        return true;
    }
    return false;
}

void main(){
    uint instanceIndex = gl_GlobalInvocationID.x;
    uint indirectDispatchRef = uIndirectComp.indRef;
    uint objRef = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].objectDataRef;
    uint transRef = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].transformRef;
    uint perframeRef = uPerframeView.ref.x;

    mat4 worldToView = GetResource(bPerframeView,perframeRef).data.m_worldToView;
    mat4 localToWorld = GetResource(bLocalTransform,transRef).m_localToWorld;
    vec4 boundBall = GetResource(bMeshDataRef,objRef).boundingSphere;

    vec4 worldBoundBall = localToWorld * vec4(boundBall.xyz,1.0);
    vec4 viewBoundBall = worldToView * worldBoundBall;
    

    bool frustumCulled = frustumCull(viewBoundBall,boundBall.w);
    if(!frustumCulled){
        // if accept
        uint acceptRef = uIndirectComp.acceptRef;
        uint acceptIndex = atomicAdd(GetResource(bHierCullDispatch,indirectDispatchRef).accepted,1);
        GetResource(bInstanceAccepted,acceptRef).data[acceptIndex] = instanceIndex;
    }else{
        uint rejectRef = uIndirectComp.rejectRef;
        uint rejectIndex = atomicAdd(GetResource(bHierCullDispatch,indirectDispatchRef).rejected,1);
        GetResource(bInstanceRejected,rejectRef).data[rejectIndex] = instanceIndex;
    }
    if(gl_GlobalInvocationID.x == 0){
        GetResource(bHierCullDispatch,indirectDispatchRef).compY = 1;
        GetResource(bHierCullDispatch,indirectDispatchRef).compZ = 1;
    }
}
