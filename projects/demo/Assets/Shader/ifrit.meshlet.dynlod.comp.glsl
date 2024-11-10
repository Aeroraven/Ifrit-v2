#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "Bindless.glsl"
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct BVHNode{
    vec4 boundingSphere;
    int numChildNodes;
    uint clusterGroupStart;
    uint clusterGroupCount;
    int subTreeSize;
    int childNodes[8];
};

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

// Consumer-Producer queue
RegisterStorage(consumer,{ int v; });
RegisterStorage(producer,{ int v; });
RegisterStorage(remainingCluster,{ int v; });
RegisterStorage(queue,{ int data[]; });

// Meshlet storage buffers
RegisterStorage(clusterGroups,{ 
    ClusterGroup data[]; 
});
RegisterStorage(drawCallSize,{
    uvec3 counter;
});
RegisterStorage(meshletsInClusterGroup,{
    uint data[];
});
RegisterStorage(bvhNodes,{
    BVHNode data[];
});
RegisterStorage(filteredMeshlets, { 
    uint data[]; 
});

// Uniform bindings
RegisterUniform(ubo,{
    uint clusterGroupCounts;
    uint totalBvhNodes;
    uint totalLods;
});
RegisterUniform(ubocam,{
    mat4 mvp;
    mat4 mv;
    vec4 cameraPos;
    uint meshletCount;
    float fov;
});

layout(set = 1, binding = 0) uniform BindlessBuffer { 
    uint counterId; // Num of meshlets filtered
    uint clusterGroupId; // Cluster group buffer
    uint bvhId; // BVH buffer
    uint meshletId; // Filtered meshlets buffer
    uint meshletInClusterGroupId; // Meshlets in cluster group buffer
    uint uniformId; // Uniform buffer
    uint unicamId;  // Camera uniform buffer
    uint consumerId; // Consumer counter
    uint producerId;  // Producer counter
    uint queueId; // Queue buffer
    uint remainingId; // Remaining cluster counter
    uint dummy2;
} bindless;

float computeProjectedRadius(float fovy,float d,float r) {
  // https://stackoverflow.com/questions/21648630/radius-of-projected-sphere-in-screen-space
  float fov = fovy / 2;
  return 1.0 / tan(fov) * r / sqrt(d * d - r * r); 
}

bool isBVHNodeVisible(uint id){
    return true;
}

bool isClusterGroupVisible(uint id){
    // Seems the original paper just use `parentError` to determine whether a cluster group is visible
    // This implementation uses both `parentError` and `selfError` to determine visibility
    // which is more slow
    const float MESHLET_CULL_ERROR_THRESH = 1.0;
    const float RENDERTARGET_HEIGHT = 1080.0;

    uint totalLod = GetResource(ubo,bindless.uniformId).totalLods;
    float fov = GetResource(ubocam,bindless.unicamId).fov;
    vec3 cameraPos = GetResource(ubocam,bindless.unicamId).cameraPos.xyz;

    ClusterGroup group = GetResource(clusterGroups,bindless.clusterGroupId).data[id];
    vec3 selfSphereCenter = group.selfBoundSphere.xyz;
    float selfSphereRaidus = group.selfError;
    vec3 parentSphereCenter = group.parentBoundSphere.xyz;
    float parentSphereRadius = group.parentError;
    bool parentRejected = true;

    if(group.lod != totalLod-1){
        vec4 viewSpaceCenter = GetResource(ubocam,bindless.unicamId).mv * vec4(parentSphereCenter, 1.0);
        vec3 viewSpaceCenter3 = viewSpaceCenter.xyz / viewSpaceCenter.w;
        float parentProjectedRadius = computeProjectedRadius(fov, length(viewSpaceCenter3), parentSphereRadius);
        parentProjectedRadius*=RENDERTARGET_HEIGHT;
        parentRejected = parentProjectedRadius > MESHLET_CULL_ERROR_THRESH;
    }
    if(!parentRejected) return false;

    bool selfRejected = false;
    if(group.lod != 0){
        vec4 viewSpaceCenter = GetResource(ubocam,bindless.unicamId).mv * vec4(selfSphereCenter, 1.0);
        vec3 viewSpaceCenter3 = viewSpaceCenter.xyz / viewSpaceCenter.w;
        float selfProjectedRadius = computeProjectedRadius(fov, length(viewSpaceCenter3), selfSphereRaidus);
        selfProjectedRadius*=RENDERTARGET_HEIGHT;
        selfRejected = selfProjectedRadius > MESHLET_CULL_ERROR_THRESH;
    }
    return !selfRejected;
}

void enqueueClusterGroup(uint id){
    // add all meshlets in the cluster group to the filtered meshlets buffer
    ClusterGroup group = GetResource(clusterGroups,bindless.clusterGroupId).data[id];
    uint pos = atomicAdd(GetResource(drawCallSize,bindless.counterId).counter.x, group.childMeshletCount);
    for(uint i = 0; i < group.childMeshletCount; i++){
        // get meshlet in cluster group
        uint meshletId = GetResource(meshletsInClusterGroup,bindless.meshletInClusterGroupId).data[group.childMeshletStart + i];
        GetResource(filteredMeshlets,bindless.meshletId).data[pos + i] = meshletId;
    }
}

void main(){
    // TODO: Only 1 workgroup is used, low GPU utilization (better sync strategy required)
    uint threadId = gl_GlobalInvocationID.x;
    uint groupSize = gl_WorkGroupSize.x;

    const int UNSET = 0x7FFFFFFF;
    
    int chosenBVHNodeInd = UNSET;
    int chosenBVHNodePos = UNSET;
    uint totalBVHNodes = GetResource(ubo,bindless.uniformId).totalBvhNodes;
    uint totalClusters = GetResource(ubo,bindless.uniformId).clusterGroupCounts;

    uint testCounter = 0;
    
    // Reset queue
    if(threadId == 0){
        GetResource(consumer,bindless.consumerId).v = 1;
        GetResource(producer,bindless.producerId).v = 1;
        GetResource(remainingCluster,bindless.remainingId).v = int(totalBVHNodes);
        GetResource(drawCallSize,bindless.counterId).counter = uvec3(0,1,1);
    }
    
    for(uint i = 0; i < totalClusters ; i += groupSize){
        GetResource(queue,bindless.queueId).data[i+threadId] = UNSET;
    }

    // Enqueue root cluster, and assign it to the first thread
    if(threadId == 0){
        GetResource(queue,bindless.queueId).data[0] = 0;
        chosenBVHNodeInd = 0;
    }
    memoryBarrier();
    uint deadlock =0;
    
    while(true){
        deadlock++;
        if(deadlock > 10000){
            break;
        }
        // Check if all BVH Nodes have been processed
        int remaining = GetResource(remainingCluster,bindless.remainingId).v;
        if(remaining <= 0){
            break;
        }
        // If current thread does not have a cluster to process, try to get one
        if(chosenBVHNodeInd == UNSET){
            chosenBVHNodeInd = atomicAdd(GetResource(consumer,bindless.consumerId).v, 1);
            if(chosenBVHNodeInd >= totalBVHNodes){
                break;
            }
        }
        if(chosenBVHNodeInd != UNSET && chosenBVHNodePos == UNSET){
            int temp = UNSET;
            int retVal = atomicExchange(GetResource(queue,bindless.queueId).data[chosenBVHNodeInd], temp);
            if(retVal != UNSET){
                chosenBVHNodePos = retVal;
            }
        }
        
        if(chosenBVHNodePos != UNSET){
            // Check whether the BVH node is visible
            bool bvhNodeVisible = isBVHNodeVisible(chosenBVHNodePos);
            if(bvhNodeVisible){
                // If the BVH node is visible, enqueue its children
                atomicAdd(GetResource(remainingCluster,bindless.remainingId).v, -1);
                BVHNode node = GetResource(bvhNodes,bindless.bvhId).data[chosenBVHNodePos];
                
                for(uint i = 0; i < node.numChildNodes; i++){
                    int childNode = node.childNodes[i];
                    int pos = atomicAdd(GetResource(producer,bindless.producerId).v, 1);
                    atomicExchange(GetResource(queue,bindless.queueId).data[pos], childNode);
                }
                // Check whether the cluster group is visible
                for(uint i = 0; i < node.clusterGroupCount; i++){
                    ClusterGroup group = GetResource(clusterGroups,bindless.clusterGroupId).data[node.clusterGroupStart + i];
                    bool clusterGroupVisible = isClusterGroupVisible(node.clusterGroupStart + i);  
                    if(clusterGroupVisible){
                        enqueueClusterGroup(node.clusterGroupStart + i);
                    }
                }
            }else{
                // Reject all subtrees
                int subTreeSize = GetResource(bvhNodes,bindless.bvhId).data[chosenBVHNodePos].subTreeSize;
                atomicAdd(GetResource(remainingCluster,bindless.remainingId).v, -subTreeSize);
            }
            // Reset the chosen BVH node
            chosenBVHNodeInd = UNSET;
            chosenBVHNodePos = UNSET;
        }
    }
}