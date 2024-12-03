
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