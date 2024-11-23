#version 450
#extension GL_GOOGLE_include_directive : require

// Instance culling typically sends the instance's root BVH node
// to the persistent culling stage.
// It's said that instance culling uses mesh as the smallest unit of culling.
// From: https://www.reddit.com/r/unrealengine4/comments/sycyof/analysis_of_ue5_rendering_technology_nanite/

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro.Shared.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

RegisterStorage(bInstanceAccepted,{
    uint data[];
});
RegisterStorage(bInstanceRejected,{
    uint data[];
});
RegisterStorage(bHierCullDispatch,{
    // Indirect command buffer for accepted
    uint accepted;
    uint compY;
    uint compZ;
    // Indirect command buffer for rejected
    uint rejected;
    uint compYR;
    uint compZR;
    // Indirect command buffer for accepted (2nd pass)
    uint accepted2;
    uint compY2;
    uint compZ2;
    // Total rejected instance
    uint totalRejected; 
});

RegisterStorage(bHizMipsReference,{
    uint pad;
    uint ref[];
});

layout(binding = 0, set = 1) uniform PerframeViewData{
    uint refCurFrame;
    uint refPrevFrame;
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

// layout(binding = 0,set=4) uniform HizMips{
//     uint ref;
// }uHizMipsRef;

layout(binding = 0, set = 4) uniform HiZData{
    uint depthImg; // Depth image, with samplers
    uint hizRefs; // Reference to image views, UAVs 
    uint hizAtomics; // Atomic counter
}uHiZData;

layout(push_constant) uniform CullingPass{
    uint passNo;
} pConst;

bool isSecondCullingPass(){
    return pConst.passNo == 1;
}

float signedDistToPlane(vec4 plane, vec4 point){
    return dot(plane.xyz,point.xyz) + plane.w;
}

uint getPerFrameRef(){
    if(isSecondCullingPass()){
        return uPerframeView.refCurFrame;
    }else{
        return uPerframeView.refPrevFrame;
    }
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
    float camFar = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraFar;
    float camNear = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraNear;
    float camFovY = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraFovY;
    float halfFovY = camFovY * 0.5;
    float camAspect = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraAspect;
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

// Hiz Test
float computeProjectedRadius(float fovy,float d,float r) {
  // https://stackoverflow.com/questions/21648630/radius-of-projected-sphere-in-screen-space
  float fov = fovy / 2;
  return 1.0 / tan(fov) * r / sqrt(d * d - r * r); 
}

float hizFetch(uint lodIndex, uint x, uint y){
    uint hizMip = GetResource(bHizMipsReference,uHiZData.hizRefs).ref[lodIndex];
    uint clampX = clamp(x,0,uint(GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_renderWidth) >> lodIndex);
    uint clampY = clamp(y,0,uint(GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_renderHeight) >> lodIndex);
    return  imageLoad(GetUAVImage2DR32F(hizMip),ivec2(clampX,clampY)).r;
}

uint roundUpPow2(uint x){
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}   

bool occlusionCull(vec4 boundBall, float radius){
    uint renderHeight = uint(GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_renderHeight);
    uint renderWidth = uint(GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_renderWidth);
    uint totalLods = uint(GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_hizLods);
    mat4 proj = GetResource(bPerframeView,getPerFrameRef()).data.m_perspective;
    float fovY = GetResource(bPerframeView,getPerFrameRef()).data.m_cameraFovY;

    // Make 8 AABB corners
    vec4 boundBallAABB[8];
    boundBallAABB[0] = boundBall + vec4(radius,radius,radius,0.0);
    boundBallAABB[1] = boundBall + vec4(-radius,radius,radius,0.0);
    boundBallAABB[2] = boundBall + vec4(radius,-radius,radius,0.0);
    boundBallAABB[3] = boundBall + vec4(-radius,-radius,radius,0.0);
    boundBallAABB[4] = boundBall + vec4(radius,radius,-radius,0.0);
    boundBallAABB[5] = boundBall + vec4(-radius,radius,-radius,0.0);
    boundBallAABB[6] = boundBall + vec4(radius,-radius,-radius,0.0);
    boundBallAABB[7] = boundBall + vec4(-radius,-radius,-radius,0.0);

    vec4 projBoundBallAABB[8];
    vec3 projBoundBallUVz[8];
    bool ableToCull = true;
    for(int i = 0; i < 8; i++){
        projBoundBallAABB[i] = proj * boundBallAABB[i];
        if(projBoundBallAABB[i].w <= 0.0){
            ableToCull = false;
            break;
        }
        vec3 projBoundBallAABB3 = projBoundBallAABB[i].xyz / projBoundBallAABB[i].w;
        float projZ = projBoundBallAABB3.z;
        projBoundBallAABB3 = projBoundBallAABB3 * 0.5 + 0.5;
        projBoundBallUVz[i] = projBoundBallAABB3;
        projBoundBallUVz[i].z = projZ;
        if(projBoundBallUVz[i].x < 0.0 || projBoundBallUVz[i].y < 0.0 || projBoundBallUVz[i].x > 1.0 || projBoundBallUVz[i].y > 1.0){
            ableToCull = false;
            break;
        }
    }
    if(!ableToCull){
        return false;
    }
    // get lod to use
    vec2 minUV = vec2(1.0,1.0);
    vec2 maxUV = vec2(0.0,0.0);
    float minZ = 1.0;
    for(int i = 0; i < 8; i++){
        minUV = min(minUV,projBoundBallUVz[i].xy);
        maxUV = max(maxUV,projBoundBallUVz[i].xy);
        minZ = min(minZ,projBoundBallUVz[i].z);
    }
    if(minZ < 0.0){
        return false;
    }
    uint rectW = uint((maxUV.x - minUV.x) * float(renderWidth));
    uint rectH = uint((maxUV.y - minUV.y) * float(renderHeight));
    float lod = clamp(log2(float(max(rectW,rectH))) ,0.0,float(totalLods - 1));
    uint lodIndex = uint(lod);

    // then fetch the hiz value
    uint lod0Width = roundUpPow2(renderWidth);
    uint lod0Height = roundUpPow2(renderHeight);
    float uvfactorX = float(renderWidth) / float(lod0Width);
    float uvfactorY = float(renderHeight) / float(lod0Height);
    vec2 uvfactor = vec2(uvfactorX,uvfactorY);

    uint lodWidth = lod0Width >> lodIndex;
    uint lodHeight = lod0Height >> lodIndex;
    uvec2 minUVInt = uvec2(minUV * uvfactor * vec2(lodWidth,lodHeight));
    
    float depth1 = hizFetch(lodIndex,minUVInt.x,minUVInt.y);
    float depth2 = hizFetch(lodIndex,minUVInt.x + 1,minUVInt.y);
    float depth3 = hizFetch(lodIndex,minUVInt.x,minUVInt.y + 1);
    float depth4 = hizFetch(lodIndex,minUVInt.x + 1,minUVInt.y + 1);

    float maxDepth = max(max(depth1,depth2),max(depth3,depth4));

    // check if the aabb is occluded
    if(minZ > maxDepth){
        return true;
    }
    return false;
}

void main(){
    bool isFirstPass = !isSecondCullingPass();

    uint instanceIndex = 0;
    uint objRef = 0;
    uint transRef = 0;
    uint transRefLast = 0;

    if(isFirstPass){
        instanceIndex = gl_GlobalInvocationID.x;
        objRef = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].objectDataRef;
        transRef = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].transformRef;
        transRefLast = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].transformRefLast;
    }else{
        // read from rejected buffer
        instanceIndex = GetResource(bInstanceRejected,uIndirectComp.rejectRef).data[gl_GlobalInvocationID.x];
        objRef = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].objectDataRef;
        transRef = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].transformRef;
        transRefLast = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].transformRefLast;
    }

    uint indirectDispatchRef = uIndirectComp.indRef;
    uint perframeRef = uPerframeView.refCurFrame;

    mat4 worldToView = GetResource(bPerframeView,perframeRef).data.m_worldToView;
    mat4 worldToViewOccl = GetResource(bPerframeView,getPerFrameRef()).data.m_worldToView;

    mat4 localToWorld = GetResource(bLocalTransform,transRef).m_localToWorld;
    vec4 boundBall = GetResource(bMeshDataRef,objRef).boundingSphere;
    vec4 worldBoundBall = localToWorld * vec4(boundBall.xyz,1.0);
    vec4 viewBoundBall = worldToView * worldBoundBall;

    mat4 localToWorldOccl;
    if(isFirstPass){
        localToWorldOccl = localToWorld;
    }else{
        localToWorldOccl = GetResource(bLocalTransform,transRefLast).m_localToWorld;
    }
    vec4 worldBoundBallOccl = localToWorldOccl * vec4(boundBall.xyz,1.0);
    vec4 viewBoundBallOccl = worldToViewOccl * worldBoundBallOccl;

    
    bool occlusionCulled = occlusionCull(viewBoundBallOccl,boundBall.w);
    if(isFirstPass){
        bool frustumCulled = frustumCull(viewBoundBall,boundBall.w);
        if(!frustumCulled && !occlusionCulled ){
            // if accept
            uint acceptRef = uIndirectComp.acceptRef;
            uint acceptIndex = atomicAdd(GetResource(bHierCullDispatch,indirectDispatchRef).accepted,1);
            GetResource(bInstanceAccepted,acceptRef).data[acceptIndex] = instanceIndex;
        }else if(!frustumCulled){
            uint rejectRef = uIndirectComp.rejectRef;
            uint rejectIndex = atomicAdd(GetResource(bHierCullDispatch,indirectDispatchRef).totalRejected,1);
            GetResource(bInstanceRejected,rejectRef).data[rejectIndex] = instanceIndex;
        }
    }else{
        // second pass, not care about the frustum and the rejection
        if(!occlusionCulled){
            // if accept
            uint acceptRef = uIndirectComp.acceptRef;
            uint acceptIndex = atomicAdd(GetResource(bHierCullDispatch,indirectDispatchRef).accepted2,1);
            GetResource(bInstanceAccepted,acceptRef).data[acceptIndex] = instanceIndex;
        }
    }
    
    if(gl_GlobalInvocationID.x == 0){
        GetResource(bHierCullDispatch,indirectDispatchRef).compY = 1;
        GetResource(bHierCullDispatch,indirectDispatchRef).compZ = 1;
        GetResource(bHierCullDispatch,indirectDispatchRef).compYR = 1;
        GetResource(bHierCullDispatch,indirectDispatchRef).compZR = 1;
        if(isSecondCullingPass()){
            GetResource(bHierCullDispatch,indirectDispatchRef).compY2 = 1;
            GetResource(bHierCullDispatch,indirectDispatchRef).compZ2 = 1;
        }
    }

}
