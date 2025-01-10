
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

// Instance culling typically sends the instance's root BVH node
// to the persistent culling stage.
// It's said that instance culling uses mesh as the smallest unit of culling.
// From: https://www.reddit.com/r/unrealengine4/comments/sycyof/analysis_of_ue5_rendering_technology_nanite/

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro/Syaro.Shared.glsl"
#include "Syaro/Syaro.SharedConst.h"

layout(local_size_x = cInstanceCullingThreadGroupSizeX, local_size_y = 1, local_size_z = 1) in;

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
}uIndirectComp;

layout(binding = 0, set = 4) uniform HiZData{
    uint depthImg; // Depth image, with samplers
    uint hizRefs; // Reference to image views, UAVs 
    uint hizAtomics; // Atomic counter
}uHiZData;

layout(push_constant) uniform CullingPass{
    uint passNo;
    uint totalInstances;
} pConst;

bool isSecondCullingPass(){
    return pConst.passNo == 1;
}

uint getPerFrameRef(){
    if(isSecondCullingPass()){
        return uPerframeView.refCurFrame;
    }else{
        return uPerframeView.refPrevFrame;
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

bool frustumCullOrtho(vec4 boundBall, float radius){
    float camFar = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraFar;
    float camNear = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraNear;
    float camFovY = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraFovY;
    float halfFovY = camFovY * 0.5;
    float camAspect = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraAspect;
    float z = boundBall.z;
    if(z+radius < camNear || z-radius > camFar){
        return true;
    }
    float camOrthoSize = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraOrthoSize;
    float camOrthoSizeCullX = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cullCamOrthoSizeX;
    float camOrthoSizeCullY = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cullCamOrthoSizeY;

    float camOrthoHalfSize = camOrthoSize;

#if SYARO_SHADER_SHARED_EXPLICIT_ORTHO_FRUSTUM_CULL
    float left = -camOrthoSizeCullX * 0.5;
    float right = camOrthoSizeCullX * 0.5;
    float top = -camOrthoSizeCullY * 0.5;
    float bottom = camOrthoSizeCullY * 0.5;
#else
    float left = -camOrthoHalfSize * camAspect;
    float right = camOrthoHalfSize * camAspect;
    float top = -camOrthoHalfSize;
    float bottom = camOrthoHalfSize;
#endif

    bool leftCull = boundBall.x - radius > right;
    bool rightCull = boundBall.x + radius < left;
    bool topCull = boundBall.y - radius > bottom;
    bool bottomCull = boundBall.y + radius < top;

    if(leftCull || rightCull || topCull || bottomCull){
        return true;
    }
    return false;
}

// Hiz Test
float computeProjectedRadius(float fovy,float d,float r) {
  // https://stackoverflow.com/questions/21648630/radius-of-projected-sphere-in-screen-space
  float fov = fovy / 2;
  if(d<=r) return 1e10;
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
    float maxScale = 0.0;

    if(isFirstPass){
        instanceIndex = gl_GlobalInvocationID.x;
        if(instanceIndex >= pConst.totalInstances){
            return;
        }
        objRef = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].objectDataRef;
        transRef = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].transformRef;
        transRefLast = GetResource(bPerObjectRef,uInstanceData.ref.x).data[instanceIndex].transformRefLast;
    }else{
        // read from rejected buffer
        if(gl_GlobalInvocationID.x >= GetResource(bHierCullDispatch,uIndirectComp.indRef).totalRejected){
            return;
        }
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
    if(!isFirstPass){
        localToWorldOccl = localToWorld;
        maxScale = GetResource(bLocalTransform,transRef).m_maxScale;
    }else{
        localToWorldOccl = GetResource(bLocalTransform,transRefLast).m_localToWorld;
        maxScale = GetResource(bLocalTransform,transRefLast).m_maxScale;
    }
    vec4 worldBoundBallOccl = localToWorldOccl * vec4(boundBall.xyz,1.0);
    vec4 viewBoundBallOccl = worldToViewOccl * worldBoundBallOccl;

    
    bool occlusionCulled = occlusionCull(viewBoundBallOccl,boundBall.w*maxScale);
    if(isFirstPass){
        bool frustumCulled;
        float camViewType = GetResource(bPerframeView,perframeRef).data.m_viewCameraType;
        if(camViewType < 0.5){
            frustumCulled = frustumCull(viewBoundBall,boundBall.w*maxScale);
        }else{
            frustumCulled = frustumCullOrtho(viewBoundBall,boundBall.w*maxScale);
        }
        if(!frustumCulled && !occlusionCulled ){
            // if accept
            uint acceptRef = uIndirectComp.acceptRef;
            uint acceptIndex = atomicAdd(GetResource(bHierCullDispatch,indirectDispatchRef).accepted,1);
            GetResource(bInstanceAccepted,acceptRef).data[acceptIndex] = instanceIndex;
        }else if(!frustumCulled && occlusionCulled){ 
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

    // An extra workgroup should be launched to handle if all instances are culled using prev hzb
    if(gl_GlobalInvocationID.x==0){
       // atomicAdd(GetResource(bHierCullDispatch,indirectDispatchRef).accepted,1);
    }

}
