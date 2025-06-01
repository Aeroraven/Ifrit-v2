
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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


#extension GL_GOOGLE_include_directive : require
#include "Base.glsl"
#include "Bindless.glsl"
#include "SamplerUtils.SharedConst.h"
#include "DeferredPBR.glsl"

#include "Ayanami/Ayanami.SharedConst.h"
#include "Ayanami/Ayanami.Shared.glsl"

#include "Random/Random.WNoise2D.glsl"

layout(push_constant)  uniform PushConstData{
    uint m_DirectLightingSRV;
    uint m_IndirectLightingSRV; 
    uint m_GAlbedoSRV;
} PushConst;

layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 oFinalColor;

void main(){
    vec3 Albedo = SampleTexture2D(PushConst.m_GAlbedoSRV, sNearestClamp, vTexCoord).xyz;
    vec4 DirectLighting = SampleTexture2D(PushConst.m_DirectLightingSRV,sLinearClamp,vTexCoord);
    vec4 IndirectLighting = SampleTexture2D(PushConst.m_IndirectLightingSRV,sLinearClamp,vTexCoord);
    vec3 LambertianBRDF = Albedo / kPI;

    // Note that direct lighting is already multiplied by the Lambertian BRDF in prev passes >w<
    vec4 FinalColor = DirectLighting + IndirectLighting * vec4(LambertianBRDF,1.0);
    oFinalColor = FinalColor;
}