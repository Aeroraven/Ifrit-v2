
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

#include "Base.glsl"
#include "Bindless.glsl"
#include "AmbientOcclusion/AmbientOcclusion.Shared.h"
#include "Random/Random.WNoise2D.glsl"

RegisterUniform(bPerframe,{
    PerFramePerViewData data;
});

layout(local_size_x = cHBAOThreadGroupSizeX, local_size_y = cHBAOThreadGroupSizeY, local_size_z = 1) in;

layout(push_constant) uniform PushConstantHBAO{
    uint perframe;
    // texFetch using samplers
    uint normalTex;
    uint depthTex;
    uint aoTex;
    float radius;
    float maxRadius;
} pushConst;

mat2 getRotationMatrix(float angle){
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c,-s,s,c);
}

vec3 toViewspace(vec2 uv, float depth, mat4 invPerspective,float nearZ, float farZ){
    float vsDepth = ifrit_recoverViewSpaceDepth(depth,nearZ,farZ);
    vec4 clipPos = vec4(uv * 2.0 - 1.0,depth,1.0)*vsDepth;
    vec4 viewPos = invPerspective * clipPos;
    return viewPos.xyz / viewPos.w;
}

// https://www.derschmale.com/2013/12/20/an-alternative-implementation-for-hbao-2/
float computeAO(vec3 vpos, vec3 stepVpos, vec3 normal, inout float topOcclusion) {
    vec3 horizonVector = stepVpos - vpos;
    float horizonVectorLength = length(horizonVector);
    float occlusion = dot(normal, horizonVector) / horizonVectorLength;
    float diff = max(occlusion - topOcclusion, 0);
    topOcclusion = max(occlusion, topOcclusion);
    float distanceFactor = clamp(horizonVectorLength / 0.5, 0, 1);
    distanceFactor = 1 - distanceFactor * distanceFactor;
    return diff * distanceFactor;
}

void main(){
    uint renderHeight = uint(GetResource(bPerframe,pushConst.perframe).data.m_renderHeight);
    uint renderWidth = uint(GetResource(bPerframe,pushConst.perframe).data.m_renderWidth);
    uint threadX = gl_GlobalInvocationID.x;
    uint threadY = gl_GlobalInvocationID.y;
    if(threadX >= renderWidth || threadY >= renderHeight){
        return;
    }
    vec2 uv = (vec2(0.5)+ vec2(threadX,threadY)) / vec2(renderWidth,renderHeight);
    vec2 tUV = vec2(threadX,threadY);

    vec3 vsNormal = texture(GetSampler2D(pushConst.normalTex),uv).xyz;
    vsNormal = normalize(vsNormal * 2.0 - 1.0);
    float vsDepth = texture(GetSampler2D(pushConst.depthTex),uv).x;

    mat4 invPerspective = GetResource(bPerframe,pushConst.perframe).data.m_invPerspective;
    float nearZ = GetResource(bPerframe,pushConst.perframe).data.m_cameraNear;
    float farZ = GetResource(bPerframe,pushConst.perframe).data.m_cameraFar;
    vec3 vsPos = toViewspace(uv,vsDepth,invPerspective,nearZ,farZ);

    // Create a random coef for sampling direction disturbance
    float rand = ifrit_wnoise2(tUV).x * kPI * 2.0 / float(cHBAODirections);

    float weightedAO = 0.0;
    float totalWeight = 0.0;
    for(uint i=0;i<cHBAODirections;i++){
        float dirAng = float(i) * kPI * 2.0 / float(cHBAODirections) + rand;
        vec2 dir = getRotationMatrix(dirAng) * vec2(1.0,0.0);
        float sampleStep = min(3e-2/vsPos.z,5e-3);// min(pushConst.radius / vsPos.z, pushConst.maxRadius);

        float accAO = 0.0;
        float maxAO = 0.03;
        for(uint j=0;j<cHBAOSampleSteps;j++){
            vec2 sampUV = uv + dir * sampleStep;
            float sampDepth = texture(GetSampler2D(pushConst.depthTex),sampUV).x;
            vec3 sampVSPos = toViewspace(sampUV,sampDepth,invPerspective,nearZ,farZ);
            float ao = computeAO(vsPos,sampVSPos,vsNormal,maxAO);
            accAO += ao;
        }    
        weightedAO += accAO;
        totalWeight += 1.0;
    }

    weightedAO /= totalWeight;

    // store ao in alpha channel
    vec4 rawSmoothAO = imageLoad(GetUAVImage2DR32F(pushConst.aoTex),ivec2(threadX,threadY));
    rawSmoothAO.a = 1.0 - weightedAO; //= vec4(1.0 - weightedAO,0.0,0.0,0.0);
    imageStore(GetUAVImage2DRGBA32F(pushConst.aoTex),ivec2(threadX,threadY),rawSmoothAO);

}