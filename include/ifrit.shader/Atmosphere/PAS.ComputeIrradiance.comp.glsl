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
    uint atmoData;
    uint uavTransmittance;
    uint uavIrradiance;
    uint uavIrradianceDelta;
} pConst;

void main(){
    uvec2 thread = uvec2(gl_GlobalInvocationID.xy);
    vec2 px = vec2(thread) + vec2(0.5);
    AtmosphereParameters atmo = GetResource(bAtmo, pConst.atmoData).data;
    vec3 deltaIrr = ComputeDirectIrradianceTexture(atmo, GetSampler2D(pConst.uavTransmittance),px);

    imageStore(GetUAVImage2DRGBA32F(pConst.uavIrradianceDelta), ivec2(thread), vec4(deltaIrr, 1.0));
    imageStore(GetUAVImage2DRGBA32F(pConst.uavIrradiance), ivec2(thread), vec4(0.0));
}