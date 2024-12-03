#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Atmosphere/PAS.SharedConst.h"
#include "Atmosphere/PAS.Definition.glsl"
#include "Atmosphere/PAS.Function.glsl"
#include "Atmosphere/PAS.Shared.glsl"

layout(local_size_x = cPasMultipleScatteringTGX, local_size_y = cPasMultipleScatteringTGY, local_size_z = cPasMultipleScatteringTGZ) in;

layout(push_constant) uniform PASPushConst {
    mat4 luminanceFromRadiance;
    uint atmoData;
    uint deltaMultipleScattering;
    uint scattering;
    uint transmittance;
    uint scatteringDensity;
} pConst;

void main(){
    uvec3 thread = uvec3(gl_GlobalInvocationID.xyz);
    vec3 px = vec3(thread) + vec3(0.5);

    AtmosphereParameters atmo = GetResource(bAtmo, pConst.atmoData).data;
    float nu;
    vec3 deltaMultipleScattering = ComputeMultipleScatteringTexture(atmo,
        GetSampler2D(pConst.transmittance),
        GetSampler3D(pConst.scatteringDensity), px,nu);
    float tmp = RayleighPhaseFunction(nu);
    vec4 scatteringRgb = pConst.luminanceFromRadiance * vec4(deltaMultipleScattering/tmp,0.0);
    vec4 scatteringOld = imageLoad(GetUAVImage3DRGBA32F(pConst.scattering), ivec3(thread));
    scatteringRgb += scatteringOld;

    imageStore(GetUAVImage3DRGBA32F(pConst.deltaMultipleScattering), ivec3(thread), vec4(deltaMultipleScattering,0.0));
    imageStore(GetUAVImage3DRGBA32F(pConst.scattering), ivec3(thread), scatteringRgb);

}