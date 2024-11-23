// References:
// https://learnopengl-cn.github.io/07%20PBR/02%20Lighting/
// https://zhuanlan.zhihu.com/p/69380665

// Utilties for Deferred PBR rendering

float dpbr_trowbridgeReitzGGX(float NdotH, float roughness){
    float PI = 3.14159265359;
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

float dpbr_smithSchlickGGX(float NdotV, float NdotL, float roughness){
    float alphaRemap = (roughness + 1.0) * (roughness + 1.0) / 4.0;
    float k = alphaRemap / 2.0;
    float G1 = NdotV / (NdotV * (1.0 - k) + k);
    float G2 = NdotL / (NdotL * (1.0 - k) + k);
    return G1 * G2;
}

vec3 dpbr_fresnelSchlick(float cosTheta, vec3 F0){
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 dpbr_fresnelSchlickMetallic(vec3 F0, vec3 albedo, float metallic, float HdotV){
    vec3 F0mix = mix(F0, albedo, metallic);
    vec3 F = dpbr_fresnelSchlick(HdotV, F0mix);
    return F;
}

vec3 dpbr_cookTorranceBRDF(vec3 F, float G, float D, float NdotV, float NdotL){
    return F * G * D / (4.0 * NdotV * NdotL + 0.0001);
}