// Layout reference:
// https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@13.1/manual/rendering/deferred-rendering-path.html

struct GBufferRefs{
    uint albedo_materialFlags;
    uint specular_occlusion;
    uint normal_smoothness;
    uint emissive;
    uint shadowMask;
};

struct GBufferPixel{
    vec3 albedo;
    float materialFlags;
    vec3 normal;
    float smoothness;
    vec3 specular;
    float occlusion;
    vec4 emissive;
    vec4 shadowMask;
};

RegisterStorage(bGBufferRefs,{
    GBufferRefs data[];
});

// Gbuffer operations for compute shaders
void gbcomp_WriteAlbedo(uint gBuffer, uint2 pixel, vec3 albedo){
    uint albedo_materialFlags_ref = GetResource(bGBufferRefs, gBuffer).albedo_materialFlags;
    vec4 albedo_materialFlags = imageRead(GetUAVImage2DRGBA8(albedo_materialFlags_ref), ivec2(pixel));
    albedo_materialFlags.xyz = albedo;
    imageStore(GetUAVImage2DRGBA8(albedo_materialFlags_ref), ivec2(pixel), albedo_materialFlags);
}
vec3 gbcomp_ReadAlbedo(uint gBuffer, uint2 pixel){
    uint albedo_materialFlags_ref = GetResource(bGBufferRefs, gBuffer).albedo_materialFlags;
    vec4 albedo_materialFlags = imageRead(GetUAVImage2DRGBA8(albedo_materialFlags_ref), ivec2(pixel));
    albedo = albedo_materialFlags.xyz;
    return albedo;
}
void gbcomp_WriteAlbedoMaterialFlags(uint gBuffer, uint2 pixel,vec3 albedo, float materialFlags){
    uint albedo_materialFlags_ref = GetResource(bGBufferRefs, gBuffer).albedo_materialFlags;
    vec4 albedo_materialFlags = imageRead(GetUAVImage2DRGBA8(albedo_materialFlags_ref), ivec2(pixel));
    albedo_materialFlags.xyz = albedo;
    albedo_materialFlags.w = materialFlags;
    imageStore(GetUAVImage2DRGBA8(albedo_materialFlags_ref), ivec2(pixel), albedo_materialFlags);
}


void gbcomp_WriteNormal(uint gBuffer, uint2 pixel, vec3 normal){
    uint normal_smoothness_ref = GetResource(bGBufferRefs, gBuffer).normal_smoothness;
    vec4 normal_smoothness = imageRead(GetUAVImage2DRGBA8(normal_smoothness_ref), ivec2(pixel));
    normal_smoothness.xyz = normal;
    imageStore(GetUAVImage2DRGBA8(normal_smoothness_ref), ivec2(pixel), normal_smoothness);
}
vec3 gbcomp_ReadNormal(uint gBuffer, uint2 pixel){
    uint normal_smoothness_ref = GetResource(bGBufferRefs, gBuffer).normal_smoothness;
    vec4 normal_smoothness = imageRead(GetUAVImage2DRGBA8(normal_smoothness_ref), ivec2(pixel));
    normal = normal_smoothness.xyz;
    return normal;
}
void gbcomp_WriteNormalSmoothness(uint gBuffer, uint2 pixel, vec3 normal, float smoothness){
    uint normal_smoothness_ref = GetResource(bGBufferRefs, gBuffer).normal_smoothness;
    vec4 normal_smoothness = imageRead(GetUAVImage2DRGBA8(normal_smoothness_ref), ivec2(pixel));
    normal_smoothness.xyz = normal;
    normal_smoothness.w = smoothness;
    imageStore(GetUAVImage2DRGBA8(normal_smoothness_ref), ivec2(pixel), normal_smoothness);
}

void gbcomp_WriteSpecularOcclusion(uint gBuffer, uint2 pixel, vec3 specular, float occlusion){
    uint specular_occlusion_ref = GetResource(bGBufferRefs, gBuffer).specular_occlusion;
    vec4 specular_occlusion = imageRead(GetUAVImage2DRGBA8(specular_occlusion_ref), ivec2(pixel));
    specular_occlusion.xyz = specular;
    specular_occlusion.w = occlusion;
    imageStore(GetUAVImage2DRGBA8(specular_occlusion_ref), ivec2(pixel), specular_occlusion);
}


void gbcomp_WriteEmissive(uint gBuffer, uint2 pixel, vec4 emissive){
    uint emissive_ref = GetResource(bGBufferRefs, gBuffer).emissive;
    vec4 emissive = imageRead(GetUAVImage2DRGBA8(emissive_ref), ivec2(pixel));
    imageStore(GetUAVImage2DRGBA8(emissive_ref), ivec2(pixel), emissive);
}
vec4 gbcomp_ReadEmissive(uint gBuffer, uint2 pixel){
    uint emissive_ref = GetResource(bGBufferRefs, gBuffer).emissive;
    vec4 emissive = imageRead(GetUAVImage2DRGBA8(emissive_ref), ivec2(pixel));
    return emissive;
}

void gbcomp_WriteGBuffer(uint gBuffer, uint2 pixel, GBufferPixel pixelData){
    gbcomp_WriteAlbedoMaterialFlags(gBuffer, pixel, pixelData.albedo, pixelData.materialFlags);
    gbcomp_WriteNormalSmoothness(gBuffer, pixel, pixelData.normal, pixelData.smoothness);
    gbcomp_WriteSpecularOcclusion(gBuffer, pixel, pixelData.specular, pixelData.occlusion);
    gbcomp_WriteEmissive(gBuffer, pixel, pixelData.emissive);
}