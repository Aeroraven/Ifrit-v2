
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

// Deferred Shading
// Although, I think every full-screen pass can be replaced with compute shader.
// And with the raster pipeline before, conventional graphics pipeline (vertex shader)
// seems to be a bit redundant.

#include "Base.glsl"
#include "Bindless.glsl"
#include "DeferredPBR.glsl"
#include "Random/Random.WNoise2D.glsl"
#include "Syaro/Syaro.SharedConst.h"

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 outColor;

RegisterUniform(bPerframeView,{
    PerFramePerViewData data;
});

layout(binding = 0, set = 1) uniform EmitGBufferRefs{
    uint albedo_materialFlags;
    uint specular_occlusion;
    uint normal_smoothness;
    uint emissive;
    uint shadowMask;
} uGBufferRefs;

layout(binding = 0, set = 2) uniform MotionDepthRefs{
    uint ref;
} uMotionDepthRefs;

layout(binding = 0, set = 3) uniform PerframeViewData{
    uint refCurFrame;
    uint refPrevFrame;
}uPerframeView;

struct ShadowMaps{
    uint viewRef;
    uint shadowRef;
};

RegisterStorage(bShadowMaps,{
    ShadowMaps data[];
});

layout(push_constant) uniform PushConstant{
    uint shadowRef;
    uint numShadowMaps;
    uint depthTexRef;
} pc;

mat2 rotate2d(float angle){
    return mat2(cos(angle),-sin(angle),sin(angle),cos(angle));
}

float shadowMappingSingle(uint lightId, vec3 worldPos, float pcfRadius){
    // 1=no shadow, 0=shadow
    const float kHalton2[16] = float[16](0.5,0.25,0.75,0.125,0.625,0.375,0.875,0.0625,0.5625,0.3125,0.8125,0.1875,0.6875,0.4375,0.9375,0.03125);
    const float kHalton3[16] = float[16](0.333333,0.666667,0.111111,0.444444,0.777778,0.222222,0.555556,0.888889,0.037037,0.370370,0.703704,0.148148,0.481481,0.814815,0.259259,0.592593);

    const uint spX = uint(gl_FragCoord.x);
    const uint spY = uint(gl_FragCoord.y);
    const float rand = ifrit_wnoise2(vec2(spX,spY));

    
    ShadowMaps shadowMap = GetResource(bShadowMaps,pc.shadowRef).data[lightId];
    mat4 lightView = GetResource(bPerframeView,shadowMap.viewRef).data.m_worldToView;
    mat4 lightProj = GetResource(bPerframeView,shadowMap.viewRef).data.m_perspective;
    mat4 vp = lightProj * lightView;
    vec4 lightPos = vp * vec4(worldPos,1.0);
    lightPos.xyz /= lightPos.w;
    vec2 uv = lightPos.xy * 0.5 + 0.5;
    float avgShadow = 0.0;
    
    
    float kSearchRadiusPx = pcfRadius;

#if SYARO_DEFERRED_SHADOW_MAPPING_HALTON_PCF_SAMPLING
    for(int k=0;k<SYARO_DEFERRED_SHADOW_MAPPING_HALTON_PCF_NUM_SAMPLES;k++){
        vec2 offset = vec2(kHalton2[k%16]-0.5,kHalton3[k%16]-0.5) * 2.0;
        vec2 rot = rotate2d(rand * kPI + float(k/16) * 2.0) * offset * kSearchRadiusPx / 2048.0;
        vec2 sampPos = uv + rot;
        sampPos = clamp(sampPos,vec2(0.0),vec2(1.0));
        float depth = texture(GetSampler2D(shadowMap.shadowRef),sampPos).r;
        if(depth - lightPos.z < -1e-3 ){
            avgShadow += 0.0;
        }else{
            avgShadow += 1.0;
        }
    }
    avgShadow /= float(SYARO_DEFERRED_SHADOW_MAPPING_HALTON_PCF_NUM_SAMPLES);
#else
    int sampleRadius = 3;
    for(int kx = -sampleRadius ; kx <= sampleRadius; kx++){
        for(int ky = -sampleRadius; ky <= sampleRadius; ky++){
            vec2 offset = vec2(kx,ky) / 2048.0 * kSearchRadiusPx/3.0;
            vec2 sampPos = uv + offset;
            sampPos = clamp(sampPos,vec2(0.0),vec2(1.0));
            float depth = texture(GetSampler2D(shadowMap.shadowRef),sampPos).r;
            if(depth - lightPos.z < -1e-3 ){
                avgShadow += 0.0;
            }else{
                avgShadow += 1.0;
            }
        }
    }
    avgShadow /= float(49);
#endif
    return avgShadow;
}

float pcssBlockerAvgDepth(vec2 uv, float lightDepth,uint shadowTexRef){
    float avgDepth = 0.0;
    float numSamples = 0.0;
    for(int k = -2; k <= 2; k++){
        for(int l = -2; l <= 2; l++){
            vec2 offset = vec2(k,l) / 2048.0 * 5.0;
            vec2 sampPos = uv + offset;
            float depth = texture(GetSampler2D(shadowTexRef),sampPos).r;
            if(depth - lightDepth < -1e-5 ){
                avgDepth += depth;
                numSamples += 1.0;
            }
        }
    }
    if(numSamples == 0.0){
        return 1e9;
    }
    return avgDepth / numSamples;
}

float pcssShadowMapSingle(uint lightId, vec3 worldPos){
    ShadowMaps shadowMap = GetResource(bShadowMaps,pc.shadowRef).data[lightId];
    mat4 lightView = GetResource(bPerframeView,shadowMap.viewRef).data.m_worldToView;
    mat4 lightProj = GetResource(bPerframeView,shadowMap.viewRef).data.m_perspective;
    mat4 vp = lightProj * lightView;

    vec3 posViewSpace = (lightView * vec4(worldPos,1.0)).xyz;
    vec4 posClip = vp * vec4(worldPos,1.0);
    posClip.xyz /= posClip.w;
    vec2 uv = posClip.xy * 0.5 + 0.5;

    float dBias = 1e-8;
    float dBlocker = pcssBlockerAvgDepth(uv,posClip.z,shadowMap.shadowRef) + dBias;
    float dReceiver = posClip.z + dBias;
    if(dReceiver - dBlocker < 1e-5 || dBlocker > 1e8){
        return 1.0;
    }
    
    float rPenumbra = max(0.0,35.0 * (dReceiver - dBlocker)/dBlocker);

    float avgShadow = 0.0;
    avgShadow = shadowMappingSingle(lightId,worldPos,rPenumbra);
    return avgShadow;
}


float globalShadowMapping(vec3 worldPos){
    float avgShadow = 10000.0;
    for(int i = 0; i < pc.numShadowMaps; i++){
        float shadow = pcssShadowMapSingle(i,worldPos);
        avgShadow = min(avgShadow,shadow);
    }
    return avgShadow;
}

void main(){
    vec3 albedo = texture(GetSampler2D(uGBufferRefs.albedo_materialFlags),texCoord).rgb;
    vec3 normal = texture(GetSampler2D(uGBufferRefs.normal_smoothness),texCoord).rgb;
    vec4 motion_depth = texture(GetSampler2D(uMotionDepthRefs.ref),texCoord).rgba;
    mat4 invproj = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_invPerspective;
    float camNear = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraNear;
    float camFar = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_cameraFar;
    float vsDepth = ifrit_recoverViewSpaceDepth(motion_depth.b,camNear,camFar);
    float ao = texture(GetSampler2D(uGBufferRefs.specular_occlusion),texCoord).a;
    mat4 clipToWorld = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_clipToWorld;
    float depth = texture(GetSampler2D(pc.depthTexRef),texCoord).r;

    if(motion_depth.a < 0.5){
        outColor = vec4(0.0);
        return;
    }

    vec4 ndcPos = vec4(texCoord * 2.0 - 1.0, depth, 1.0) * vsDepth;
    vec4 viewPos = invproj * ndcPos;
    vec4 worldPos = clipToWorld * ndcPos;
    worldPos /= worldPos.w;
    viewPos /= viewPos.w;
    normal = normalize(normal * 2.0 - 1.0);
    vec3 lightDir = normalize(vec3(0.612372,0.500000,0.612372));

    float NdotL = max(dot(normal,lightDir),0.0);
    vec3 V = normalize(-viewPos.xyz);
    vec3 H = normalize(lightDir + V);
    float NdotH = max(dot(normal,H),0.0);
    float roughness = 0.6;
    float D = dpbr_trowbridgeReitzGGX(NdotH,roughness);

    float NdotV = max(dot(normal,V),0.0);
    float G = dpbr_smithSchlickGGX(NdotV,NdotL,roughness);

    vec3 F0 = vec3(0.04);
    float metallic = 0.1;
    float HdotV = max(dot(H,V),0.0);
    vec3 F = dpbr_fresnelSchlickMetallic(F0,albedo,metallic,HdotV);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    float PIx = 3.14159265359;
    vec3 specular = dpbr_cookTorranceBRDF(F,G,D,NdotV,NdotL);
    vec3 Lo = (kD * albedo / PIx + specular) * NdotL;

    vec3 ambient = vec3(0.10) * albedo * ao;
    float shadow = globalShadowMapping(worldPos.xyz);

    //This is incorrect, but it's used for test if shadow mapping works
    vec3 color = ambient + Lo * shadow; 
    //color = vec3(shadow);
    outColor = vec4(color,1.0);
}