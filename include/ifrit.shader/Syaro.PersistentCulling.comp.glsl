#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Syaro.Shared.glsl"


layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

struct ClusterGroup{
    vec4 selfBoundSphere;
    vec4 parentBoundSphere;
    float selfError;
    float parentError;
    uint childMeshletStart;
    uint childMeshletCount;
    uint lod;
    uint dummy1;
    uint dummy2;
    uint dummy3;
};

struct BVHNode{
    vec4 boundingSphere;
    int numChildNodes;
    uint clusterGroupStart;
    uint clusterGroupCount;
    int subTreeSize;
    int childNodes[8];
};


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
    int data[];
});
RegisterStorage(bFilteredMeshlets, { 
    int data[]; 
});

struct MeshletDesc{
    uint instanceId;
    uint meshletId;
};
RegisterStorage(bFilteredMeshlets2,{
    ivec2 data[];
});


layout(binding = 0, set = 1) uniform PerframeViewData{
    uvec4 ref;
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

float computeProjectedRadius(float fovy,float d,float r) {
  // https://stackoverflow.com/questions/21648630/radius-of-projected-sphere-in-screen-space
  float fov = fovy / 2;
  return 1.0 / tan(fov) * r / sqrt(d * d - r * r); 
}

bool isBVHNodeVisible(uint id){
    return true;
}

uint getObjId(){
    return GetResource(bInstanceAccepted,uIndirectCompInstCull.acceptRef).data[gl_WorkGroupID.x];
}

bool isClusterGroupVisible(uint id){
    uint objId = getObjId();
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint instId = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].instanceDataRef;
    uint trans = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].transformRef;
    uint cpcntRefMesh = GetResource(bMeshDataRef,obj).cpCounterBuffer;
    uint cpcntRefInst = GetResource(bInstanceDataRef,instId).cpCounterBuffer;

    uint clusterRef = GetResource(bMeshDataRef,obj).clusterGroupBuffer;
    uint totalLod = GetResource(bCpCounterMesh,cpcntRefMesh).totalLods;

    float fov = GetResource(bPerframeView,uPerframeView.ref.x).data.m_cameraFovY;
    vec3 camPos = GetResource(bPerframeView,uPerframeView.ref.x).data.m_cameraPosition.xyz;   

    ClusterGroup group = GetResource(bClusterGroup,clusterRef).data[id];
    vec3 selfSphereCenter = group.selfBoundSphere.xyz;
    float selfSphereRadius = group.selfError;
    vec3 parentSphereCenter = group.parentBoundSphere.xyz;
    float parentSphereRadius = group.parentError;

    bool parentRejected = true;
    if(group.lod != totalLod-1){
        mat4 model = GetResource(bLocalTransform,trans).m_localToWorld;
        mat4 view = GetResource(bPerframeView,uPerframeView.ref.x).data.m_worldToView;
        mat4 mv = view * model;

        vec4 viewSpaceCenter = mv * vec4(parentSphereCenter,1.0);
        vec3 viewSpaceCenter3 = viewSpaceCenter.xyz / viewSpaceCenter.w;
        float parentProjectedRadius = computeProjectedRadius(fov,length(viewSpaceCenter3),parentSphereRadius);
        parentProjectedRadius*=1080.0;
        parentRejected = parentProjectedRadius > 1.0;
    }
    if(!parentRejected){
        return false;
    }

    bool selfRejected = false;
    if(group.lod != 0){
        mat4 model = GetResource(bLocalTransform,trans).m_localToWorld;
        mat4 view = GetResource(bPerframeView,uPerframeView.ref.x).data.m_worldToView;
        mat4 mv = view * model;

        vec4 viewSpaceCenter = mv * vec4(selfSphereCenter,1.0);
        vec3 viewSpaceCenter3 = viewSpaceCenter.xyz / viewSpaceCenter.w;
        float selfProjectedRadius = computeProjectedRadius(fov,length(viewSpaceCenter3),selfSphereRadius);
        selfProjectedRadius*=1080.0;
        selfRejected = selfProjectedRadius > 1.0;
    }
    return !selfRejected;

}

void enqueueClusterGroup(uint id, uint clusterRef){
    ClusterGroup group = GetResource(bClusterGroup,clusterRef).data[id];
    uint objId =getObjId();
    uint objMesh = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint objInst = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].instanceDataRef;
    uint filteredRef = GetResource(bInstanceDataRef,objInst).filteredMeshletsBuffer;
    uint micRef = GetResource(bMeshDataRef,objMesh).meshletInClusterBuffer;
    int numMeshlets = int(group.childMeshletCount);

    // Seems atomicity is guaranteed across workgroups
    int pos = atomicAdd(GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).data[0],numMeshlets);

    for(uint i = 0;i<group.childMeshletCount;i++){
        uint meshletId = GetResource(bMeshletsInClusterGroup,micRef).data[group.childMeshletStart + i];
        GetResource(bFilteredMeshlets2,uIndirectDrawData2.allMeshletsRef).data[pos+i] = ivec2(objId,meshletId);
    }
}

void main(){
    
    uint threadId = gl_LocalInvocationID.x;
    uint objId = getObjId();
    uint groupSize = gl_WorkGroupSize.x;
    uint obj = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].objectDataRef;
    uint instId = GetResource(bPerObjectRef,uInstanceData.ref.x).data[objId].instanceDataRef;

    uint cpcntRefMesh = GetResource(bMeshDataRef,obj).cpCounterBuffer;
    uint cpcntRefInst = GetResource(bInstanceDataRef,instId).cpCounterBuffer;
    uint cpqueueRef = GetResource(bInstanceDataRef,instId).cpQueueBuffer;

    uint bvhRef = GetResource(bMeshDataRef,obj).bvhNodeBuffer;
    uint clusterRef = GetResource(bMeshDataRef,obj).clusterGroupBuffer;
    const int UNSET = 0x7FFFFFFF;

    uint totalBVHNodes = GetResource(bCpCounterMesh,cpcntRefMesh).totalBvh;
    uint totalClusterGroups = GetResource(bCpCounterMesh,cpcntRefMesh).totalCluster;
    if(threadId == 0){
        GetResource(bCpCounterInstance, cpcntRefInst).con = 0;
        GetResource(bCpCounterInstance, cpcntRefInst).prod = 0;
        GetResource(bCpCounterInstance, cpcntRefInst).remain = int(totalBVHNodes);
    }
    

    for(uint i= 0;i<totalClusterGroups;i+=groupSize){
        GetResource(bCpQueue,cpqueueRef).data[i+threadId] = UNSET;
    }

    int chosenBVHNodeInd = UNSET;
    int chosenBVHNodePos = UNSET;
    if(threadId == 0){
        GetResource(bCpQueue,cpqueueRef).data[0] = 0;
        // TODO: this should hold num of instances instead of num of meshlets in the future
        GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).data[1] = 1;
        GetResource(bDrawCallSize,uIndirectDrawData2.indDrawCmdRef).data[2] = 1;
        chosenBVHNodeInd = 0;
    }
    memoryBarrier();

    while(true){
        int remaining = GetResource(bCpCounterInstance,cpcntRefInst).remain;
        if(remaining <= 0){
            break;
        }
        if(chosenBVHNodeInd == UNSET){
            chosenBVHNodeInd = atomicAdd(GetResource(bCpCounterInstance,cpcntRefInst).con,1);
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
                atomicAdd(GetResource(bCpCounterInstance,cpcntRefInst).remain,-1);
                BVHNode node = GetResource(bBVHNode,bvhRef).data[chosenBVHNodePos];
                for(uint i = 0;i < node.numChildNodes ; i++){
                    int childNode = node.childNodes[i];
                    int pos = atomicAdd(GetResource(bCpCounterInstance,cpcntRefInst).prod,1);
                    atomicExchange(GetResource(bCpQueue,cpqueueRef).data[pos],childNode); 
                }
                for(uint i = 0 ; i < node.clusterGroupCount;i++){
                    ClusterGroup group = GetResource(bClusterGroup,clusterRef).data[node.clusterGroupStart + i];
                    bool clusterGroupVisible = isClusterGroupVisible(node.clusterGroupStart + i);
                    if(clusterGroupVisible){
                        enqueueClusterGroup(node.clusterGroupStart + i,clusterRef);
                    }
                }
            }else{
                int subTreeSize = GetResource(bBVHNode,bvhRef).data[chosenBVHNodePos].subTreeSize;
                atomicAdd(GetResource(bCpCounterInstance,cpcntRefInst).remain,-subTreeSize);
            }
            chosenBVHNodePos = UNSET;
            chosenBVHNodeInd = UNSET;
        }
    }
}