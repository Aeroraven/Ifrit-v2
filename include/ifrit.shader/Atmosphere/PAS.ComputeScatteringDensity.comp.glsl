#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Atmosphere/PAS.SharedConst.h"
#include "Atmosphere/PAS.Definition.glsl"
#include "Atmosphere/PAS.Function.glsl"
#include "Atmosphere/PAS.Shared.glsl"

layout(local_size_x = cPasScatteringDensityTGX, local_size_y = cPasScatteringDensityTGY, local_size_z = cPasScatteringDensityTGZ) in;

layout(push_constant) uniform PASPushConst {
    uint atmoData;
    uint transmittance;
    uint singleRayleighScattering;
    uint singleMieScattering;
    uint multipleScattering;
    uint irradiance;
    uint scatterDensity;
    int scatteringOrder;
} pConst;

void main(){
    uvec3 thread = uvec3(gl_GlobalInvocationID.xyz);
    vec3 px = vec3(thread) + vec3(0.5);

    AtmosphereParameters atmo = GetResource(bAtmo, pConst.atmoData).data;

    vec3 scatterDensity = ComputeScatteringDensityTexture(atmo,
        GetSampler2D(pConst.transmittance),
        GetSampler3D(pConst.singleRayleighScattering),
        GetSampler3D(pConst.singleMieScattering),
        GetSampler3D(pConst.multipleScattering),
        GetSampler2D(pConst.irradiance),
        px,pConst.scatteringOrder);  

    imageStore(GetUAVImage3DRGBA32F(pConst.scatterDensity), ivec3(thread), vec4(scatterDensity, 1.0));
}