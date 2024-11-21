#version 450
#extension GL_GOOGLE_include_directive : require

// Deferred Shading
// Although, I think every full-screen pass can be replaced with compute shader.
// And with the raster pipeline before, conventional graphics pipeline (vertex shader)
// seems to be a bit redundant.

#include "Bindless.glsl"

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 outColor;

layout(binding = 0, set = 1) uniform EmitGBufferRefs{
    uint albedo_materialFlags;
    uint specular_occlusion;
    uint normal_smoothness;
    uint emissive;
    uint shadowMask;
} uGBufferRefs;

void main(){
    vec3 albedo = texture(GetSampler2D(uGBufferRefs.albedo_materialFlags),texCoord).rgb;
    vec3 normal = texture(GetSampler2D(uGBufferRefs.normal_smoothness),texCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    vec3 lightDir = normalize(vec3(1.0,0.0,-1.0));

    float NdotL = max(dot(normal,lightDir),0.0);
    vec3 diffuse = albedo * NdotL;

    outColor = vec4(diffuse, 1.0);
}