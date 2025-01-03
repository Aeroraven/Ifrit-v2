
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
    uvec4 viewRef;
    uvec4 shadowRef;
    uvec4 viewMapping; //Useless for shader, should be optimized
    vec4 csmStart;
    vec4 csmEnd;
    uint csmNumSplits;
};

RegisterStorage(bShadowMaps,{
    ShadowMaps data[];
});

layout(push_constant) uniform PushConstant{
    vec4 sundir;
    uint shadowRef;
    uint numShadowMaps;
    uint depthTexRef;
    uint shadowTexRef;
} pc;

const vec4 testLightPos = vec4(0.61237, -0.50, -0.61237, 0.00) * 3.0;

mat2 rotate2d(float angle){
    return mat2(cos(angle),-sin(angle),sin(angle),cos(angle));
}

void main(){
    vec3 albedo = texture(GetSampler2D(uGBufferRefs.albedo_materialFlags),texCoord).rgb;
    vec3 normal = texture(GetSampler2D(uGBufferRefs.normal_smoothness),texCoord).rgb;
    vec4 motion_depth = texture(GetSampler2D(uMotionDepthRefs.ref),texCoord).rgba;
    mat4 invproj = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_invPerspective;
    mat4 worldToView = GetResource(bPerframeView,uPerframeView.refCurFrame).data.m_worldToView;
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
    normal = (worldToView * vec4(normalize(normal * 2.0 - 1.0),0.0)).xyz;

    vec3 sundir = -normalize(pc.sundir.xyz);
    vec3 lightDir = (worldToView * vec4(sundir,0.0)).xyz;
    //normalize(vec3(0.612372,0.500000,0.612372));

    float NdotL = max(dot(normal,lightDir),0.0);
    vec3 V = normalize(-viewPos.xyz);
    vec3 H = normalize(lightDir + V);
    float NdotH = max(dot(normal,H),0.0);
    float roughness = 0.62;
    float D = dpbr_trowbridgeReitzGGX(NdotH,roughness);

    float NdotV = max(dot(normal,V),0.0);
    float G = dpbr_smithSchlickGGX(NdotV,NdotL,roughness);

    vec3 F0 = vec3(0.04);
    float metallic = 0.03;
    float HdotV = max(dot(H,V),0.0);
    vec3 F = dpbr_fresnelSchlickMetallic(F0,albedo,metallic,HdotV);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    float PIx = 3.14159265359;
    vec3 specular = dpbr_cookTorranceBRDF(F,G,D,NdotV,NdotL);
    vec3 Lo = (kD * albedo / PIx + specular) * NdotL * 2.3;

    vec3 ambient = vec3(0.12) * albedo * pow(ao,1.5);
    float shadow = texture(GetSampler2D(pc.shadowTexRef),texCoord).r;

    //This is incorrect, but it's used for test if shadow mapping works
    vec3 color = ambient + Lo * shadow;
    outColor = vec4(color,1.0);
}