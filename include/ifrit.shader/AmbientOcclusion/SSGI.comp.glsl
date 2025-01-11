
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

RegisterStorage(bHiZStorage,{
    uint pad;
    uint mipRefs[];
});

layout(local_size_x = cSSGIThreadGroupSizeX, local_size_y = cSSGIThreadGroupSizeY, local_size_z = 1) in;

layout(push_constant) uniform PushConstantSSGI{
    uint perframe;
    uint normalTex; //SRV
    uint depthHizTex; //Ref->UAVs
    uint aoTex; //UAV
    uint albedoTex; //SRV
    uint hizTexW;
    uint hizTexH;
    uint rtW;
    uint rtH;
    uint maxMips;
} pushConst;


struct RayHitPayload{
    bool hit;
    vec3 hitPos;
};

vec3 toViewspace(vec2 uv, float depth, mat4 invPerspective,float nearZ, float farZ){
    float vsDepth = ifrit_recoverViewSpaceDepth(depth,nearZ,farZ);
    vec4 clipPos = vec4(uv * 2.0 - 1.0,depth,1.0)*vsDepth;
    vec4 viewPos = invPerspective * clipPos;
    return viewPos.xyz / viewPos.w;
}

float getHizDepth(ivec2 uv, uint mip, bool ranged){
    // convert uv to mip
    ivec2 mipUV = uv >> mip;
    uint mipId = GetResource(bHiZStorage,pushConst.depthHizTex).mipRefs[mip];
    if(!ranged){
        return imageLoad(GetUAVImage2DR32F(mipId), mipUV).r;
    }
    float t0 = imageLoad(GetUAVImage2DR32F(mipId), mipUV).r;
    float t1 = imageLoad(GetUAVImage2DR32F(mipId), mipUV+ivec2(1,0)).r;
    float t2 = imageLoad(GetUAVImage2DR32F(mipId), mipUV+ivec2(0,1)).r;
    float t3 = imageLoad(GetUAVImage2DR32F(mipId), mipUV+ivec2(1,1)).r;

    return min(min(t0,t1),min(t2,t3));
}

float ndcToUV(float ndc){
    return (ndc + 1.0) * 0.5;
}

float uvToNdc(float uv){
    return uv * 2.0 - 1.0;
}   

// For simplicity, roughness is not used here.
vec3 getWoRay(vec3 normal, vec3 rayDir, float roughness, uint seed){
    float threadX = gl_GlobalInvocationID.x;
    float threadY = gl_GlobalInvocationID.y;
    vec2 randParam1 = vec2(threadX,threadY) * 7 * float(seed);
    vec2 randParam2 = vec2(threadX,threadY) * 13 * float(seed);

    float rand1 = ifrit_wnoise2(randParam1);
    float rand2 = ifrit_wnoise2(randParam2);

    // make a random vector on sphere
    float phi = rand1 * 2.0 * 3.1415926;
    float cosTheta = 1.0 - rand2;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec3 randVec = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    vec3 finalOutRay = normal+randVec;

    return normalize(finalOutRay);
}

float getWoSamplingWeight(vec3 normal, vec3 rayDir, vec3 outRay){
    float newDistWeight = dot(normal,outRay) / k_PI;
    float uniqDistWeight = 1.0 / (4.0 * k_PI);
    return uniqDistWeight / newDistWeight;
}

void ssgiTraverse(mat4 invP, vec3 startPosVS,vec3 rayDirVS,vec3 clipStartSS,
     vec3 clipNextSS, out RayHitPayload payload){

    vec3 nextPosVS = startPosVS + rayDirVS;
    float ssFactorX = clipNextSS.x - clipStartSS.x;
    float ssFactorY = clipNextSS.y - clipStartSS.y;

    bool useY = abs(ssFactorY) > abs(ssFactorX);
    float ssFactor = useY ? ssFactorY : ssFactorX;
    float ssScaleFactor = useY ? ssFactorX/ssFactorY : ssFactorY/ssFactorX;
    
    int currentMips = int(pushConst.maxMips-1);
    uint pixelAdvance = (1<<(pushConst.maxMips-1));

    int clipStartSSPx = int(ndcToUV(clipStartSS.x) * pushConst.rtW);
    int clipStartSSPy = int(ndcToUV(clipStartSS.y) * pushConst.rtH);

    float clipCurSSPx = clipStartSSPx;    
    float clipCurSSPy = clipStartSSPy;
    float compareDepth;
    // starting with a small offset
    if(!useY){
        clipCurSSPx += pixelAdvance>>1;
        clipCurSSPy += (pixelAdvance>>1) * ssScaleFactor;
    }else{
        clipCurSSPy += pixelAdvance>>1;
        clipCurSSPx += (pixelAdvance>>1) * ssScaleFactor;
    }

    while(currentMips>=0){
        // note that the  hiz here use MIN instead of MAX. so we need another hiz.
        // pass to get the MIN.
        compareDepth = getHizDepth(ivec2(clipStartSSPx,clipStartSSPy),uint(currentMips),currentMips!=0);
        bool hit = false;
        // get current depth
        if(!useY){
            float deltaSx = float(clipCurSSPx - clipStartSSPx) / pushConst.rtW;
            float deltaSxNdc = uvToNdc(deltaSx);
            float tX = deltaSxNdc / ssFactorX;
            float ndcZ = ifrit_perspectiveLerp(clipStartSS.z,clipNextSS.z,startPosVS.z,nextPosVS.z,tX);
            if(ndcZ > compareDepth+1e-3){
                hit = true;
            }
        }else{
            float deltaSy = float(clipCurSSPy - clipStartSSPy) / pushConst.rtH;
            float deltaSyNdc = uvToNdc(deltaSy);
            float tY = deltaSyNdc / ssFactorY;
            float ndcZ = ifrit_perspectiveLerp(clipStartSS.z,clipNextSS.z,startPosVS.z,nextPosVS.z,tY);
            if(ndcZ > compareDepth+1e-3){
                hit = true;
            }
        }

        if(!hit){
            if(!useY){
                clipCurSSPx += pixelAdvance;
                clipCurSSPy += pixelAdvance * ssScaleFactor;
            }
            else{
                clipCurSSPy += pixelAdvance;
                clipCurSSPx += pixelAdvance * ssScaleFactor;
            }
            if(clipCurSSPx >= pushConst.rtW || clipCurSSPy >= pushConst.rtH ||
                clipCurSSPx < 0 || clipCurSSPy < 0){
                payload.hit = false;
                return;
            }
        }else{
            // Narrow down the hit position
            hit = false;
            currentMips--;
            pixelAdvance>>=1;
        }
    }

    payload.hit = true;

    vec2 hituv = vec2(clipCurSSPx,clipCurSSPy)/vec2(pushConst.rtW,pushConst.rtH);
    vec2 hitNdc = vec2(uvToNdc(hituv.x),uvToNdc(hituv.y));
    float hitZ = compareDepth;
    vec3 hitNdc3 = vec3(hitNdc,hitZ);

    // inverse transform this into view space
    vec4 clipPos = vec4(hitNdc3,1.0);
    vec4 viewPos = invP * clipPos;
    payload.hitPos = viewPos.xyz / viewPos.w;
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

    uint depthMip0Id = GetResource(bHiZStorage,pushConst.depthHizTex).mipRefs[0];
    float vsDepth = imageLoad(GetUAVImage2DR32F(depthMip0Id), ivec2(threadX,threadY)).r;

    mat4 invPerspective = GetResource(bPerframe,pushConst.perframe).data.m_invPerspective;
    mat4 perspective = GetResource(bPerframe,pushConst.perframe).data.m_perspective;
    float nearZ = GetResource(bPerframe,pushConst.perframe).data.m_cameraNear;
    float farZ = GetResource(bPerframe,pushConst.perframe).data.m_cameraFar;
    vec3 vsPos = toViewspace(uv,vsDepth,invPerspective,nearZ,farZ);
    vec3 normal = normalize(texelFetch(GetSampler2D(pushConst.normalTex), ivec2(threadX,threadY), 0).xyz);
    vec3 startAlbedo = texelFetch(GetSampler2D(pushConst.albedoTex), ivec2(threadX,threadY), 0).xyz;
    vec3 inRay = normalize(vsPos);

    vec3 finalRadiance = vec3(0.0);

    // rendering equation: L_o = L_e + L_i * BRDF * cos(theta)
    // lambertian BRDF: BRDF = 1/pi
    // lamertian consine distribution: cos(theta) = dot(N,L)

    vec3 finalGI = vec3(0.0);
    for(uint S=0;S<cSSGISamples;S++){
        vec3 outRay = getWoRay(normal,inRay,0.0,threadX%19+S);

        vec3 historyOutRay[cSSGIBounces];
        vec3 historyInRay[cSSGIBounces];
        float historyWeight[cSSGIBounces];
        vec3  historyColor[cSSGIBounces];
        vec3  startupColor = vec3(0.1);
        int historyCount = 0;

        historyInRay[0] = inRay;
        historyOutRay[0] = outRay;
        historyWeight[0] = getWoSamplingWeight(normal,inRay,outRay);
        historyAlbedo[0] = startAlbedo;
        historyCount++;


        // Not recursive, only using 1 bounce.
        for(uint T=0;T<cSSGIBounces;T++){

            vec4 clipPos1 = perspective * vec4(vsPos,1.0);
            vec2 uv1 = clipPos1.xy / clipPos1.w;
            vec3 vsPos2 = vsPos + outRay;
            vec4 clipPos2 = perspective * vec4(vsPos2,1.0);
            vec2 uv2 = clipPos2.xy / clipPos2.w;

            RayHitPayload payload;
            ssgiTraverse(invPerspective,vsPos,outRay,uv1,uv2,payload);
            if(!payload.hit){
                startupColor = vec3(4.0);
                break;
            }

            // update next iteration
            vec3 clipPosHit = perspective * vec4(payload.hitPos,1.0);
            vec2 uvHit = clipPosHit.xy / clipPosHit.w;
            uvHit = (uvHit + 1.0) * 0.5;
            ivec2 hitUV = ivec2(uvHit * vec2(pushConst.hizTexW,pushConst.hizTexH));
            vec3 hitNormal = normalize(texelFetch(GetSampler2D(pushConst.normalTex), hitUV, 0).xyz);
            vec3 hitAlbedo = texelFetch(GetSampler2D(pushConst.albedoTex), hitUV, 0).xyz;


            inRay = outRay;
            outRay = getWoRay(hitNormal,inRay,0.0,threadX%19+S+T);
            vsPos = payload.hitPos;

            historyInRay[T+1] = inRay;
            historyOutRay[T+1] = outRay;
            historyWeight[T+1] = getWoSamplingWeight(hitNormal,inRay,outRay);
            historyAlbedo[T+1] = hitAlbedo;
            historyCount++;
        }

        // calculate the final radiance, for simplicity, only 1 bounce is used.
        // T=0 is direct illumination, so we skip it.
        for(int T=historyCount-1;T>=1;T--){
            finalRadiance += historyAlbedo[T] * historyWeight[T-1];
        }
        finalGI += finalRadiance;
    }

    finalGI /= float(cSSGISamples);

    imageStore(GetUAVImage2DRGBA(pushConst.aoTex), ivec2(threadX,threadY), vec4(finalGI,1.0));
}
