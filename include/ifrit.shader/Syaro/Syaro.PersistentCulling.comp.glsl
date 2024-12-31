
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

layout(local_size_x = cPersistentCullThreadGroupSizeX, local_size_y = 1, local_size_z = 1) in;

struct ClusterGroup{
    vec4 selfBoundSphere;
    vec4 parentBoundSphere;
    uint childMeshletStart;
    uint childMeshletCount;
    uint lod;
    uint dummy1;
};

struct BVHNode{
    vec4 boundingSphere;
    int numChildNodes;
    uint clusterGroupStart;
    uint clusterGroupCount;
    int subTreeSize;
    int childNodes[8];
    float maxClusterError;
    int dummy1;
    int dummy2;
    int dummy3;
};
RegisterStorage(bMeshlet,{
    Meshlet data[];
});

RegisterStorage(bClusterGroup,{
    ClusterGroup data[];
});
RegisterStorage(bBVHNode,{
    BVHNode data[];
});
RegisterStorage(bCpQueue,{ int data[]; });
RegisterStorage(bCpCounterMesh,{ 
    int totalBvh;
    int totalCluster;
    int totalLods;
    int pad1;
});
RegisterStorage(bCpCounterInstance,{ 
    int con;
    int prod;
    int remain;
    int pad1;
});

RegisterStorage(bInstanceAccepted,{
    uint data[];
});

RegisterStorage(bMeshletsInClusterGroup,{
    uint data[];
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

RegisterStorage(bFilteredMeshlets2,{
    ivec2 data[];
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


layout(binding = 0, set = 1) uniform PerframeViewData{
    uint refCurFrame;
    uint refPrevFrame;
}uPerframeView;

layout(binding = 0, set = 2) uniform InstanceData{
    uvec4 ref;
}uInstanceData;

layout(binding = 0, set = 3) uniform IndirectDrawData{
    uvec4 ref;
}uIndirectDrawData;

layout(binding = 0, set = 4) uniform IndirectDrawData2{
    uint allMeshletsRef;
    uint indDrawCmdRef;
}uIndirectDrawData2;

layout(binding = 0, set = 5) uniform IndirectCompData{
    uint acceptRef;
    uint rejectRef;
    uint indRef;
    uint pad;
}uIndirectCompInstCull;

layout(push_constant) uniform CullingPass{
    uint passNo;
} pConst;

shared uint sConsumer;
shared uint sProducer;
shared int sRemain;


bool isSecondCullingPass(){
    return pConst.passNo == 1;
}

float computeProjectedRadius(float tanfovy,float d,float r) {
  // https://stackoverflow.com/questions/21648630/radius-of-projected-sphere-in-screen-space
  if(d<r){
    return 1e10;
  }
  return 1.0 / tanfovy * r / sqrt(d * d - r * r); 
}

bool isBVHNodeVisible(uint id){
    // Currently, the impl is wrong, so we just return true.
    // The correct impl will be added later.
    return true;
}

uint getObjId(){
    return GetResource(bInstanceAccepted,uIndirectCompInstCull.acceptRef).data[gl_WorkGroupID.x];
}

bool isClusterGroupVisible(uint id, mat4 mvMat,float rtHeight,float tanfovy,float viewCamType,
    float camAspect, float orthoSize, float cullOrthoX, float cullOrthoY){

    uint objId = getObjId();
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint cpcntRefMesh = GetResource(bMeshDataRef,obj).cpCounterBuffer;

    uint clusterRef = GetResource(bMeshDataRef,obj).clusterGroupBuffer;
    uint totalLod = GetResource(bCpCounterMesh,cpcntRefMesh).totalLods;
    
    vec3 camPos = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraPosition.xyz;   

    ClusterGroup group = GetResource(bClusterGroup,clusterRef).data[id];
    vec3 selfSphereCenter = group.selfBoundSphere.xyz;
    float selfSphereRadius = group.selfBoundSphere.w;
    vec3 parentSphereCenter = group.parentBoundSphere.xyz;
    float parentSphereRadius = group.parentBoundSphere.w;

    if(totalLod == 1){
        return true;
    }

    bool parentRejected = true;
    if(group.lod != totalLod-1){
        vec4 viewSpaceCenter = mvMat * vec4(parentSphereCenter,1.0);
        vec3 viewSpaceCenter3 = viewSpaceCenter.xyz / viewSpaceCenter.w;
        float parentProjectedRadius;
        
        if(viewCamType == 0){
            parentProjectedRadius = computeProjectedRadius(tanfovy,length(viewSpaceCenter3),parentSphereRadius);
        }else{
            parentProjectedRadius = parentSphereRadius * camAspect * orthoSize;
        }
        parentProjectedRadius*=rtHeight;
        parentRejected = parentProjectedRadius > 0.5;
    }
    if(!parentRejected){
        return false;
    }

    bool selfRejected = false;
    if(group.lod != 0){
        vec4 viewSpaceCenter = mvMat * vec4(selfSphereCenter,1.0);
        vec3 viewSpaceCenter3 = viewSpaceCenter.xyz / viewSpaceCenter.w;
        float selfProjectedRadius;
        if(viewCamType == 0){
            selfProjectedRadius = computeProjectedRadius(tanfovy,length(viewSpaceCenter3),selfSphereRadius);
        }else{
            selfProjectedRadius = selfSphereRadius * camAspect * orthoSize;
        }
        selfProjectedRadius*=rtHeight;
        selfRejected = selfProjectedRadius > 0.5;
    }
    return !selfRejected;

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

void enqueueClusterGroup(uint id, uint clusterRef, uint micRef, float tanHalfFovY){
    ClusterGroup group = GetResource(bClusterGroup,clusterRef).data[id];
    uint objId =getObjId();
    int numMeshlets = 1; //int(group.childMeshletCount);
    uint pos = 0;

    bool bMeshletCulled = false;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint meshletRef = GetResource(bMeshDataRef,obj).meshletBuffer;
    uint meshletId = GetResource(bMeshletsInClusterGroup,micRef).data[group.childMeshletStart];

#if SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL
    uint instId = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].instanceDataRef;
    uint trans = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].transformRef;
    mat4 model = GetResource(bLocalTransform,trans).m_localToWorld;
    mat4 view = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_worldToView;
    mat4 mv = view * model;
    
    // cone culling
    Meshlet meshlet = GetResource(bMeshlet,meshletRef).data[meshletId];
    vec4 normalConeAxis = model * vec4(meshlet.normalCone.xyz,0.0);
    vec4 normalConeApex = model * vec4(meshlet.normalConeApex.xyz,1.0);
    vec3 cameraPos = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraPosition.xyz;
    float camViewType = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_viewCameraType;
    float coneAngle = dot(normalize(normalConeApex.xyz - cameraPos),normalize(normalConeAxis.xyz));
    if(coneAngle > meshlet.normalCone.w+1e-6){
        return;
    }

    // frustum culling
    vec4 boundBall = mv * vec4(meshlet.boundSphere.xyz,1.0);
    float radius = meshlet.boundSphere.w;
    if(camViewType > 0.5){
        bMeshletCulled = frustumCullOrtho(boundBall,radius);
    }else{
        bMeshletCulled = frustumCull(boundBall,radius,tanHalfFovY);
    }
    if(bMeshletCulled){
        //return;
    }
#endif

    // End culling
    if(isSecondCullingPass()){
        uint basePos = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1;
        pos = atomicAdd(GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw2,1);
        pos += basePos;
    }else{
        pos = atomicAdd(GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1,1);
    }

    GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[pos] = ivec2(objId,meshletId);
}

void main(){
    if(gl_GlobalInvocationID.x == 0){
        uint rejected = GetResource(bHierCullDispatch,uIndirectCompInstCull.indRef).totalRejected;
        uint instanceCullTGSZ = 64;
        GetResource(bHierCullDispatch,uIndirectCompInstCull.indRef).rejected = (rejected + instanceCullTGSZ - 1) / instanceCullTGSZ;
    }

    if(gl_WorkGroupID.x==0){
        GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).y2 = 1;
        GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).z2 = 1;
        GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).y1 = 1;
        GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).z1 = 1;
    }
    
    uint threadId = gl_LocalInvocationID.x;
    uint objId = getObjId();
    uint groupSize = gl_WorkGroupSize.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint instId = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].instanceDataRef;
    uint micRef = GetResource(bMeshDataRef,obj).meshletInClusterBuffer;
    float viewCamType = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_viewCameraType;

    uint trans = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].transformRef;
    mat4 model = GetResource(bLocalTransform,trans).m_localToWorld;
    mat4 view = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_worldToView;
    mat4 mv = view * model;

    float rtHeight =  GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_renderHeight;

    uint cpcntRefMesh = GetResource(bMeshDataRef,obj).cpCounterBuffer;
    uint cpcntRefInst = GetResource(bInstanceDataRef,instId).cpCounterBuffer;
    uint cpqueueRef = GetResource(bInstanceDataRef,instId).cpQueueBuffer;

    uint bvhRef = GetResource(bMeshDataRef,obj).bvhNodeBuffer;
    uint clusterRef = GetResource(bMeshDataRef,obj).clusterGroupBuffer;
    const int UNSET = 0x7FFFFFFF;

    uint totalBVHNodes = GetResource(bCpCounterMesh,cpcntRefMesh).totalBvh;
    uint totalClusterGroups = GetResource(bCpCounterMesh,cpcntRefMesh).totalCluster;
    float fov = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraFovY;
    float tanfovy = tan(fov*0.5);
    float camAspect = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraAspect;
    float orthoSize = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraOrthoSize;
    float cullOrthoX = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cullCamOrthoSizeX;
    float cullOrthoY = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cullCamOrthoSizeY;

    if(cPersistentCullParallelStg == cPersistentCullParallelStg_PersistThread){
        if(threadId == 0){
            sConsumer = 0;
            sProducer = 0;
            sRemain = int(totalBVHNodes);
        }
        
        for(uint i= 0;i<totalClusterGroups;i+=groupSize){
            GetResource(bCpQueue,cpqueueRef).data[i+threadId] = UNSET;
        }

        int chosenBVHNodeInd = UNSET;
        int chosenBVHNodePos = UNSET;
        if(threadId == 0){
            GetResource(bCpQueue,cpqueueRef).data[0] = 0;
            chosenBVHNodeInd = 0;
        }
        barrier();
        while(true){
            int remaining = sRemain;
            if(remaining <= 0){
                break;
            }
            if(chosenBVHNodeInd == UNSET){
                chosenBVHNodeInd = int(atomicAdd(sConsumer,1));
                if(chosenBVHNodeInd >= totalBVHNodes){
                    break;
                }
            }
            if(chosenBVHNodeInd != UNSET && chosenBVHNodePos == UNSET){
                int temp = UNSET;
                int retVal = atomicExchange(GetResource(bCpQueue,cpqueueRef).data[chosenBVHNodeInd],temp);
                if(retVal != UNSET){
                    chosenBVHNodePos = retVal;
                }
            }
            if(chosenBVHNodePos != UNSET){
                bool bvhNodeVisible = isBVHNodeVisible(chosenBVHNodePos);
                if(bvhNodeVisible){
                    atomicAdd(sRemain,-1);
                    BVHNode node = GetResource(bBVHNode,bvhRef).data[chosenBVHNodePos];
                    for(uint i = 0;i < node.numChildNodes ; i++){
                        int childNode = node.childNodes[i];
                        int pos = int(atomicAdd(sProducer,1));
                        atomicExchange(GetResource(bCpQueue,cpqueueRef).data[pos],childNode); 
                    }
                    for(uint i = 0 ; i < node.clusterGroupCount;i++){
                        ClusterGroup group = GetResource(bClusterGroup,clusterRef).data[node.clusterGroupStart + i];
                        bool clusterGroupVisible = isClusterGroupVisible(node.clusterGroupStart + i,mv,rtHeight,tanfovy,
                            viewCamType,camAspect,orthoSize,cullOrthoX,cullOrthoY);
                        if(clusterGroupVisible){ 
                            enqueueClusterGroup(node.clusterGroupStart + i,clusterRef,micRef,tanfovy);
                        }
                    }
                }else{
                    int subTreeSize = GetResource(bBVHNode,bvhRef).data[chosenBVHNodePos].subTreeSize;
                    atomicAdd(sRemain,-subTreeSize);
                }
                chosenBVHNodePos = UNSET;
                chosenBVHNodeInd = UNSET;
            }
        }
    }else if(cPersistentCullParallelStg == cPersistentCullParallelStg_StridedLoop_BVH){
        for(uint k=threadId;k<totalBVHNodes;k+=cPersistentCullThreadGroupSizeX){
            uint chosenBVHNodePos = k;
            bool bvhNodeVisible = isBVHNodeVisible(chosenBVHNodePos);
            if(bvhNodeVisible){
                BVHNode node = GetResource(bBVHNode,bvhRef).data[chosenBVHNodePos];
                for(uint i = 0 ; i < node.clusterGroupCount;i++){
                    ClusterGroup group = GetResource(bClusterGroup,clusterRef).data[node.clusterGroupStart + i];
                    bool clusterGroupVisible = isClusterGroupVisible(node.clusterGroupStart + i,mv,rtHeight,tanfovy,
                        viewCamType,camAspect,orthoSize,cullOrthoX,cullOrthoY);
                    if(clusterGroupVisible){ 
                        enqueueClusterGroup(node.clusterGroupStart + i,clusterRef,micRef,tanfovy);
                    }
                }
            }
        }
    }else if(cPersistentCullParallelStg == cPersistentCullParallelStg_StridedLoop_ClusterGroup){
        for(uint k=threadId;k<totalClusterGroups;k+=cPersistentCullThreadGroupSizeX){
            ClusterGroup group = GetResource(bClusterGroup,clusterRef).data[k];
            bool clusterGroupVisible = isClusterGroupVisible(k,mv,rtHeight,tanfovy,
                viewCamType,camAspect,orthoSize,cullOrthoX,cullOrthoY);
            if(clusterGroupVisible){ 
                enqueueClusterGroup(k,clusterRef,micRef,tanfovy);
            }
        }
    }
    
    //GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).x2 = 104829;

    barrier();
    if(threadId == 0){
        if(isSecondCullingPass()){
            uint v = atomicAdd(GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).completedWorkGroups2,1) + 1;
            if(v == gl_NumWorkGroups.x){
                uint m2 = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw2;
#if SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL
                uint issuedTasks = m2;
#else
                uint issuedTasks = (m2 + cMeshRasterizeTaskThreadGroupSize-1) / cMeshRasterizeTaskThreadGroupSize;
#endif
                GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).x2 = issuedTasks;
            }
        }
        else{
            uint v = atomicAdd(GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).completedWorkGroups1,1) + 1;
            if(v == gl_NumWorkGroups.x){
                uint m1 = GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).meshletsToDraw1;
#if SYARO_SHADER_MESHLET_CULL_IN_PERSISTENT_CULL
                uint issuedTasks = m1;
#else
                uint issuedTasks = (m1 + cMeshRasterizeTaskThreadGroupSize-1) / cMeshRasterizeTaskThreadGroupSize;
#endif
                GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).x1 = issuedTasks;
            }
        }
    }
}