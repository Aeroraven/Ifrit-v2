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

RegisterStorage(bHizMipsReference,{
    uint ref[];
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

layout(binding = 0,set=4) uniform HizMips{
    uint ref;
}uHizMipsRef;

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

// Hiz Test
float computeProjectedRadius(float fovy,float d,float r) {
  // https://stackoverflow.com/questions/21648630/radius-of-projected-sphere-in-screen-space
  float fov = fovy / 2;
  return 1.0 / tan(fov) * r / sqrt(d * d - r * r); 
}

float hizFetch(uint lodIndex, uint x, uint y){
    uint hizMip = GetResource(bHizMipsReference,uHizMipsRef.ref).ref[lodIndex];
    uint clampX = clamp(x,0,uint(GetResource(bPerframeView,uPerframeView.ref.x).data.m_renderWidth) >> lodIndex);
    uint clampY = clamp(y,0,uint(GetResource(bPerframeView,uPerframeView.ref.x).data.m_renderHeight) >> lodIndex);
    return  imageLoad(GetUAVImage2DR32F(hizMip),ivec2(clampX,clampY)).r;
        //texelFetch(uHizMipsRef.ref,hizMip,ivec2(clampX,clampY),0).r;
}

bool occlusionCull(vec4 boundBall, float radius){
    uint renderHeight = uint(GetResource(bPerframeView,uPerframeView.ref.x).data.m_renderHeight);
    uint renderWidth = uint(GetResource(bPerframeView,uPerframeView.ref.x).data.m_renderWidth);
    uint totalLods = uint(GetResource(bPerframeView,uPerframeView.ref.x).data.m_hizLods);
    mat4 proj = GetResource(bPerframeView,uPerframeView.ref.x).data.m_perspective;
    float fovY = GetResource(bPerframeView,uPerframeView.ref.x).data.m_cameraFovY;
    float distBall = length(boundBall.xyz);
    float radiusProjected = computeProjectedRadius(fovY,distBall,radius);
    vec4 projBall = proj * boundBall;
    vec3 projBall3 = projBall.xyz / projBall.w;

    projBall3 = projBall3 * 0.5 + 0.5;

    uint ballDiameter = uint(radiusProjected * 2.0 * float(max(renderWidth,renderHeight)));
    float lod = clamp(log2(float(ballDiameter)) - 1.0,0.0,float(totalLods - 1));
    uint lodIndex = uint(lod);

    uint texW = max(1,renderWidth >> lodIndex);
    uint texH = max(1,renderHeight >> lodIndex);

    float ballSx = projBall3.x - radiusProjected;
    float ballSy = projBall3.y - radiusProjected;

    // get starting pixels for ballSx, ballSy
    uint ballSxPx = uint(ballSx * float(texW));
    uint ballSyPx = uint(ballSy * float(texH));

    // fetch depth value from hiz mip
    uint hizMip = GetResource(bHizMipsReference,uHizMipsRef.ref).ref[lodIndex];
    float depth1 = hizFetch(lodIndex,ballSxPx,ballSyPx);
    float depth2 = hizFetch(lodIndex,ballSxPx + 1,ballSyPx);
    float depth3 = hizFetch(lodIndex,ballSxPx,ballSyPx + 1);
    float depth4 = hizFetch(lodIndex,ballSxPx + ballDiameter,ballSyPx + ballDiameter);
    float maxDepth = max(max(depth1,depth2),max(depth3,depth4));

    if(projBall3.z > maxDepth){
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
    bool occlusionCulled = occlusionCull(viewBoundBall,boundBall.w);
    if(!frustumCulled && !occlusionCulled){
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