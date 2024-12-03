
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


layout(local_size_x = cPasSingleScatteringTGX, local_size_y = cPasSingleScatteringTGY, local_size_z = cPasSingleScatteringTGZ) in;

layout(push_constant) uniform PASPushConst {
    mat4 luminanceFromRadiance;
    uint atmoData;
    uint deltaRayleigh;
    uint deltaMie;
    uint scattering;
    uint singleMieScattering;
    uint transmittance;
} pConst;

void main(){
    uvec3 thread = uvec3(gl_GlobalInvocationID.xyz);
    vec3 px = vec3(thread) + vec3(0.5);

    AtmosphereParameters atmo = GetResource(bAtmo, pConst.atmoData).data;
    vec3 dRayleigh = vec3(0.0);
    vec3 dMie = vec3(0.0);
    ComputeSingleScatteringTexture(atmo, GetSampler2D(pConst.transmittance), 
        px, dRayleigh, dMie);

    vec4 scatteringRgb = pConst.luminanceFromRadiance * vec4(dRayleigh.rgb,0.0);
    vec4 singleMieScatteringRgb = pConst.luminanceFromRadiance * vec4(dMie.rgb,0.0);
    scatteringRgb.a = singleMieScatteringRgb.r;

    imageStore(GetUAVImage3DRGBA32F(pConst.deltaRayleigh), ivec3(thread), vec4(dRayleigh,0.0));
    imageStore(GetUAVImage3DRGBA32F(pConst.deltaMie), ivec3(thread), vec4(dMie,0.0));
    imageStore(GetUAVImage3DRGBA32F(pConst.scattering), ivec3(thread), scatteringRgb);
    imageStore(GetUAVImage3DRGBA32F(pConst.singleMieScattering), ivec3(thread), singleMieScatteringRgb);
}