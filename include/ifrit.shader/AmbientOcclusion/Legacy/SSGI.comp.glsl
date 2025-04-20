
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
#include "Random/Random.BlueNoise2D.glsl"

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
    uint depthHizTexMin; //Ref->UAVs
    uint depthHizTexMax; //Ref->UAVs
    uint aoTex; //UAV
    uint albedoTex; //SRV
    uint hizTexW;
    uint hizTexH;
    uint rtW;
    uint rtH;
    uint maxMips;
    uint blueNoiseSRV;
} pushConst;


struct RayHitPayload{
    bool hit;
    vec3 hitPos;
    float hitDepth;
    float cmpDepth;
    vec2 pdir;
};

vec3 toViewspace(vec2 uv, float depth, mat4 invPerspective,float nearZ, float farZ){
    float vsDepth = ifrit_recoverViewSpaceDepth(depth,nearZ,farZ);
    vec4 clipPos = vec4(uv * 2.0 - 1.0,depth,1.0)*vsDepth;
    vec4 viewPos = invPerspective * clipPos;
    return viewPos.xyz / viewPos.w;
}

float getHizDepth(ivec2 uv, uint mip, bool ranged, uint uavId){
    // convert uv to mip
    ivec2 mipUV = uv >> mip;
    uint mipId = GetResource(bHiZStorage,uavId).mipRefs[mip];
    if(!ranged){
        return imageLoad(GetUAVImage2DR32F(mipId), mipUV).r;
    }
    float t0 = imageLoad(GetUAVImage2DR32F(mipId), mipUV).r;
    float t1 = imageLoad(GetUAVImage2DR32F(mipId), mipUV+ivec2(1,0)).r;
    float t2 = imageLoad(GetUAVImage2DR32F(mipId), mipUV+ivec2(0,1)).r;
    float t3 = imageLoad(GetUAVImage2DR32F(mipId), mipUV+ivec2(1,1)).r;
    return min(min(t0,t1),min(t2,t3));
}

float getHizDepthMin(ivec2 uv, uint mip, bool ranged){
    return getHizDepth(uv,mip,ranged,pushConst.depthHizTexMin);
}

float getHizDepthMax(ivec2 uv, uint mip, bool ranged){
    return getHizDepth(uv,mip,ranged,pushConst.depthHizTexMax);
}

float ndcToUV(float ndc){
    return (ndc + 1.0) * 0.5;
}

float uvToNdc(float uv){
    return uv * 2.0 - 1.0;
}   
float uvToNdcDelta(float uv){
    return uv * 2.0;
}

// For simplicity, roughness is not used here.
vec3 getWoRay(vec3 normal, vec3 rayDir, float roughness, uint seed){

    // just reflect the ray
    // return reflect(rayDir,normal);
    
    float threadX = gl_GlobalInvocationID.x;
    float threadY = gl_GlobalInvocationID.y;
    float rtWidth = float(pushConst.rtW);
    float rtHeight = float(pushConst.rtH);

    vec2 randParam1 = vec2(threadX,threadY) * 7 * float(seed)/float(rtWidth);
    vec2 randParam2 = vec2(threadX,threadY) * 13 * float(seed)/float(rtHeight);

    randParam1 = fract(randParam1);
    randParam2 = fract(randParam2);

    vec4 bnoise = ifrit_bnoise2d(pushConst.blueNoiseSRV,vec2(randParam1.x,randParam1.y));

    float rand1 = bnoise.r;
    float rand2 = bnoise.g;

    rand1 = clamp(rand1,0.0,1.0);
    rand2 = clamp(rand2,0.0,1.0);

    // make a random vector on sphere
    float phi = rand1 * 2.0 * 3.1415926;
    float cosTheta = 1.0 - rand2;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec3 randVec = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    vec3 finalOutRay = normal+randVec;

    return normalize(finalOutRay);
}

float getLambertianBRDF(vec3 normal, vec3 rayDir, vec3 outRay){
    return 1.0 / kPI;
}

float getWoSamplingWeightInv(vec3 normal, vec3 rayDir, vec3 outRay){
    float newDistWeight = dot(normal,outRay);
    float uniqDistWeight = 1.0 / 4.0;
    return (newDistWeight) / uniqDistWeight;
}

void ssgiRayMarch(mat4 invP, vec3 startPosVS,vec3 rayDirVS,vec3 clipStartSS,
     vec3 clipNextSS, float camNear, float camFar, out RayHitPayload payload){
        
    vec3 nextPosVS = startPosVS + rayDirVS * SSGI_RAY_MAX_DISTANCE;
    float ssFactorX = clipNextSS.x - clipStartSS.x;
    float ssFactorY = clipNextSS.y - clipStartSS.y;

    payload.pdir = vec2(ssFactorX,ssFactorY);
    payload.pdir = normalize(payload.pdir)*0.5+0.5;

    bool useY = abs(ssFactorY) > abs(ssFactorX);
    int ssSign = int(useY ? sign(ssFactorY) : sign(ssFactorX));
    float ssFactor = useY ? ssFactorY : ssFactorX;
    float ssScaleFactor = useY ? ssFactorX/ssFactorY : ssFactorY/ssFactorX;
    
    int currentMips = 0;
    int pixelAdvance = (1<<currentMips)*ssSign;
    int initPixelAdvance = 1;

    int clipStartSSPx = int(ndcToUV(clipStartSS.x) * pushConst.rtW);
    int clipStartSSPy = int(ndcToUV(clipStartSS.y) * pushConst.rtH);

    float clipCurSSPx = clipStartSSPx;    
    float clipCurSSPy = clipStartSSPy;
    float compareDepthMin;
    float compareDepthMax;

    // Advance some distances first
    if(!useY){
        clipCurSSPx = clipCurSSPx + initPixelAdvance;
        clipCurSSPy = clipCurSSPy + initPixelAdvance * ssScaleFactor;
    }else{
        clipCurSSPy = clipCurSSPy + initPixelAdvance;
        clipCurSSPx = clipCurSSPx + initPixelAdvance * ssScaleFactor;
    }
    uint remainSteps = 40;
    while(currentMips>=0 && remainSteps-- > 0){
        // note that the  hiz here use MIN instead of MAX. so we need another hiz.
        // pass to get the MIN.
        float clipCurSSPxNext = 0;
        float clipCurSSPyNext = 0;

        if(!useY){
            clipCurSSPxNext = clipCurSSPx + pixelAdvance;
            clipCurSSPyNext = clipCurSSPy + pixelAdvance * ssScaleFactor;
        }else{
            clipCurSSPyNext = clipCurSSPy + pixelAdvance;
            clipCurSSPxNext = clipCurSSPx + pixelAdvance * ssScaleFactor;
        }

        compareDepthMin = getHizDepthMin(ivec2(clipCurSSPxNext,clipCurSSPyNext),uint(currentMips),currentMips!=0);
        compareDepthMax = getHizDepthMax(ivec2(clipCurSSPxNext,clipCurSSPyNext),uint(currentMips),currentMips!=0);

        float cmpDepthMinView = ifrit_clipZToViewZ(compareDepthMin,camNear,camFar);
        float cmpDepthMaxView = ifrit_clipZToViewZ(compareDepthMax,camNear,camFar);

        bool hit = false;
        bool nextOutBound = false;
        if(clipCurSSPxNext >= pushConst.rtW || clipCurSSPyNext >= pushConst.rtH ||
            clipCurSSPxNext < 0 || clipCurSSPyNext < 0){
            nextOutBound = true;
        }
        // get current depth
        if(!useY){
            float deltaSx = float(clipCurSSPxNext - clipStartSSPx) / pushConst.rtW;
            float deltaSxNdc = uvToNdcDelta(deltaSx);
            float tX = deltaSxNdc / ssFactorX;
            float viewZ = ifrit_perspectiveLerp(startPosVS.z,nextPosVS.z,startPosVS.z,nextPosVS.z,tX);
            float ndcZ = ifrit_viewZToClipZ(viewZ,camNear,camFar);
            if(viewZ>=cmpDepthMinView && viewZ < cmpDepthMaxView+1e-1){
                hit = true;
                payload.hitDepth = viewZ;
                payload.cmpDepth = cmpDepthMinView;
            }
        }else{
            float deltaSy = float(clipCurSSPyNext - clipStartSSPy) / pushConst.rtH;
            float deltaSyNdc = uvToNdcDelta(deltaSy);
            float tY = deltaSyNdc / ssFactorY;
            float viewZ = ifrit_perspectiveLerp(startPosVS.z,nextPosVS.z,startPosVS.z,nextPosVS.z,tY);
            float ndcZ = ifrit_viewZToClipZ(viewZ,camNear,camFar);
            if(viewZ>=cmpDepthMinView && viewZ < cmpDepthMaxView+1e-1){
                hit = true;
                payload.hitDepth = viewZ;
                payload.cmpDepth = cmpDepthMinView;
            }
        }
        if(nextOutBound){
            if(currentMips == 0){
                payload.hit = false;
                return;
            }else{
                currentMips--;
                pixelAdvance /= 2;
                continue;
            }
        }

        if(!hit){
            if(!useY){
                // check whether out of bound after advancing
                float nxtClipCurSSPx = clipCurSSPx + pixelAdvance;
                float nxtClipCurSSPy = clipCurSSPy + pixelAdvance * ssScaleFactor;
                if((nxtClipCurSSPx >= pushConst.rtW || nxtClipCurSSPy >= pushConst.rtH ||
                    nxtClipCurSSPx < 0 || nxtClipCurSSPy < 0) && currentMips > 0){
                    // narrow lods
                    hit = false;
                    currentMips--;
                    pixelAdvance = (1<<currentMips)*ssSign;;
                    continue;
                }else{
                    clipCurSSPx += pixelAdvance;
                    clipCurSSPy += pixelAdvance * ssScaleFactor;
                }
            }
            else{
                float nxtClipCurSSPx = clipCurSSPx + pixelAdvance * ssScaleFactor;
                float nxtClipCurSSPy = clipCurSSPy + pixelAdvance;
                if((nxtClipCurSSPx >= pushConst.rtW || nxtClipCurSSPy >= pushConst.rtH ||
                    nxtClipCurSSPx < 0 || nxtClipCurSSPy < 0) && currentMips > 0){
                    // narrow lods
                    hit = false;
                    currentMips--;
                    pixelAdvance=(1<<currentMips)*ssSign;
                    continue;
                }else{
                    clipCurSSPy += pixelAdvance;
                    clipCurSSPx += pixelAdvance * ssScaleFactor;
                }
            }
            if(clipCurSSPx >= pushConst.rtW || clipCurSSPy >= pushConst.rtH ||
                clipCurSSPx < 0 || clipCurSSPy < 0){
                payload.hit = false;
                return;
            }
        }else{
            if(currentMips == 0){
                break;
            }
            // Narrow down the hit position
            hit = false;
            currentMips--;
            pixelAdvance =  (1<<currentMips)*ssSign;
        }
    }

    if(remainSteps <= 0){
        payload.hit = false;
        return;
    }

    payload.hit = true;
    vec2 hituv = vec2(clipCurSSPx,clipCurSSPy)/vec2(pushConst.rtW,pushConst.rtH);
    vec2 hitNdc = vec2(uvToNdc(hituv.x),uvToNdc(hituv.y));
    float hitZ = compareDepthMin;
    vec3 hitNdc3 = vec3(hitNdc,hitZ);

    // inverse transform this into view space
    vec4 clipPos = vec4(hitNdc3,1.0);
    vec4 viewPos = invP * clipPos;
    payload.hitPos = viewPos.xyz / viewPos.w;
}

void ssgiTraverse(mat4 invP, vec3 startPosVS,vec3 rayDirVS,vec3 clipStartSS,
     vec3 clipNextSS, float camNear, float camFar, out RayHitPayload payload){

    ssgiRayMarch(invP,startPosVS,rayDirVS,clipStartSS,clipNextSS,camNear,camFar,payload);
} 

void ssgiMainSingleBounce(){
    uint renderHeight = uint(GetResource(bPerframe,pushConst.perframe).data.m_renderHeight);
    uint renderWidth = uint(GetResource(bPerframe,pushConst.perframe).data.m_renderWidth);
    uint threadX = gl_GlobalInvocationID.x;
    uint threadY = gl_GlobalInvocationID.y;
    if(threadX >= renderWidth || threadY >= renderHeight){
        return;
    }
    vec2 uv = (vec2(0.5)+ vec2(threadX,threadY)) / vec2(renderWidth,renderHeight);
    vec2 tUV = vec2(threadX,threadY);

    uint depthMip0Id = GetResource(bHiZStorage,pushConst.depthHizTexMin).mipRefs[0];
    float vsDepth = imageLoad(GetUAVImage2DR32F(depthMip0Id), ivec2(threadX,threadY)).r;

    mat4 invPerspective = GetResource(bPerframe,pushConst.perframe).data.m_invPerspective;
    mat4 perspective = GetResource(bPerframe,pushConst.perframe).data.m_perspective;
    float nearZ = GetResource(bPerframe,pushConst.perframe).data.m_cameraNear;
    float farZ = GetResource(bPerframe,pushConst.perframe).data.m_cameraFar;

    
    vec3 vsPos = toViewspace(uv,vsDepth,invPerspective,nearZ,farZ);
    vec3 normal = texelFetch(GetSampler2D(pushConst.normalTex), ivec2(threadX,threadY), 0).xyz;
    normal = normalize(normal * 2.0 - 1.0);
    vec3 startAlbedo = texelFetch(GetSampler2D(pushConst.albedoTex), ivec2(threadX,threadY), 0).xyz;
    vec3 inRay = normalize(vsPos);

    vec3 finalGI = vec3(0.0);
    bool nanFlag = false;
    for(uint S=0;S<cSSGISamples;S++){
        vec3 outRay = getWoRay(normal,inRay,0.0,threadX%149+S);
        float invpdf = getWoSamplingWeightInv(normal,inRay,outRay);
        float cosTheta = dot(normal,outRay);
        float brdf = getLambertianBRDF(normal,inRay,outRay);
        vec3 brdfA = startAlbedo*brdf;
    

        // begin ssgi traverse
        vec4 clipPos1 = perspective * vec4(vsPos,1.0);
        vec3 ndc1 = clipPos1.xyz / clipPos1.w;
        vec3 vsPos2 = vsPos + outRay * SSGI_RAY_MAX_DISTANCE;
        vec4 clipPos2 = perspective * vec4(vsPos2,1.0);
        vec3 ndc2 = clipPos2.xyz / clipPos2.w;

        RayHitPayload payload;
        ssgiTraverse(invPerspective,vsPos,outRay,ndc1,ndc2,nearZ,farZ,payload);

        vec3 Li = vec3(1.0);
        if(payload.hit){
            vec4 clipPosHit = perspective * vec4(payload.hitPos,1.0);
            vec2 uvHit = clipPosHit.xy / clipPosHit.w;
            uvHit = (uvHit + 1.0) * 0.5;
            vec3 hitAlbedo = texture(GetSampler2D(pushConst.albedoTex), uvHit).xyz;
            Li = hitAlbedo;
        }
        vec3 Ls = brdfA * Li * cosTheta * invpdf;

        finalGI += Ls;
    }
    finalGI /= float(cSSGISamples);

    // Store GI
    vec4 aoRaw = imageLoad(GetUAVImage2DRGBA32F(pushConst.aoTex), ivec2(threadX,threadY));
    aoRaw.rgb = finalGI;
    imageStore(GetUAVImage2DRGBA32F(pushConst.aoTex), ivec2(threadX,threadY),aoRaw);
}

void main(){
    ssgiMainSingleBounce();
}
