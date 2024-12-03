#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Atmosphere/PAS.SharedConst.h"
#include "Atmosphere/PAS.Definition.glsl"
#include "Atmosphere/PAS.Function.glsl"
#include "Atmosphere/PAS.Shared.glsl"

layout(local_size_x = cPasIrradianceTGX, local_size_y = cPasIrradianceTGY, local_size_z = 1) in;

layout(push_constant) uniform PASPushConst {
    mat4 luminanceFromRadiance;
    uint atmoData;
    uint deltaIrradiance;
    uint irradiance;
    uint singleRayleighScattering;
    uint singleMieScattering;
    uint multipleScattering;
    int scatteringOrder;
} pConst;

void main(){
    uvec2 thread = uvec2(gl_GlobalInvocationID.xy);
    vec2 px = vec2(thread) + vec2(0.5);

    AtmosphereParameters atmo = GetResource(bAtmo, pConst.atmoData).data;

    vec3 deltaIrradiance = ComputeIndirectIrradianceTexture(atmo,
        GetSampler3D(pConst.singleRayleighScattering),
        GetSampler3D(pConst.singleMieScattering),
        GetSampler3D(pConst.multipleScattering),
        px,pConst.scatteringOrder);
    vec4 irradiance = pConst.luminanceFromRadiance * vec4(deltaIrradiance,0.0);

    imageStore(GetUAVImage2DRGBA32F(pConst.deltaIrradiance), ivec2(thread), vec4(deltaIrradiance, 0.0));

    vec4 oldIrradiance = imageLoad(GetUAVImage2DRGBA32F(pConst.irradiance), ivec2(thread));
    irradiance += oldIrradiance;
    imageStore(GetUAVImage2DRGBA32F(pConst.irradiance), ivec2(thread), irradiance);
}