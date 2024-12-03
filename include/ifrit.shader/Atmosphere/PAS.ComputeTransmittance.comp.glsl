#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"
#include "Atmosphere/PAS.SharedConst.h"
#include "Atmosphere/PAS.Definition.glsl"
#include "Atmosphere/PAS.Function.glsl"
#include "Atmosphere/PAS.Shared.glsl"

layout(local_size_x = cPasTransmittanceTGX, local_size_y = cPasTransmittanceTGY, local_size_z = 1) in;

layout(push_constant) uniform PASComputeTransmittancePushConstant {
    uint atmoData;
    uint uavTransmittance;
} pConst;

void main(){
    uvec2 thread = uvec2(gl_GlobalInvocationID.xy);
    vec2 px = vec2(thread) + vec2(0.5);
    AtmosphereParameters atmo = GetResource(bAtmo, pConst.atmoData).data;
    vec3 transmittance = ComputeTransmittanceToTopAtmosphereBoundaryTexture(atmo, px);
    imageStore(GetUAVImage2DRGBA32F(pConst.uavTransmittance), ivec2(thread), vec4(transmittance, 1.0));
}