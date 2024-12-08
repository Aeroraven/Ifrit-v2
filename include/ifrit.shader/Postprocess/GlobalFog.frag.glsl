
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
#include "Bindless.glsl"
#include "Base.glsl"

RegisterUniform(bPerframeView,{
    PerFramePerViewData data;
});

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform GlobalFogParams{
    uint inputTexture;
    uint depthTexture;
    uint viewDataRef;
} pc;

void main(){
    // Temporarily, celestial depths are not written into the depth buffer in atmospheric scattering.
    // So, the depth values here are not the actual depth values.
    
    float depth = texture(GetSampler2D(pc.depthTexture), texCoord).r;
    mat4 inverseProj = GetResource(bPerframeView, pc.viewDataRef).data.m_invPerspective;
    float camNear = GetResource(bPerframeView, pc.viewDataRef).data.m_cameraNear;
    float camFar = GetResource(bPerframeView, pc.viewDataRef).data.m_cameraFar;
    float vsDepth = ifrit_recoverViewSpaceDepth(depth,camNear,camFar);
    vec3 clipPos = vec3(2.0 * texCoord - 1.0, depth);
    vec4 clipPos4 = vec4(clipPos, 1.0) * vsDepth;

    vec4 viewPos = inverseProj * clipPos4;
    viewPos /= viewPos.w;

    float height = max(0.0,viewPos.y);
    float posDist = length(viewPos.xyz);

    // transmittance = X ; attenuation = Y ; 
    // optical thickness = X*exp(-Y*h)*((1-exp(-Y*dist*sinv))/(Y*sinv))
    const float transmittance = 4e-3;
    const float attenuation = 0.5;
    float sinv = height / posDist;

    posDist = min(posDist, 100.0);
    float thickness;
    
    if(sinv<1e-6){
        thickness = transmittance * exp(-attenuation * 0.0) * posDist;
    }else{
        thickness = transmittance * exp(-attenuation * 0.0) * ((1.0 - exp(-attenuation * posDist * sinv)) / (attenuation * sinv));
    }

    float coef = 1.0 - exp(-thickness);

    vec4 color = texture(GetSampler2D(pc.inputTexture), texCoord).rgba;
    float colorAlpha = color.a;
    vec3 fogColor = vec3(0.5, 0.5, 0.5);
    color = mix(color, vec4(fogColor, 1.0), coef);
    color.a = clamp(colorAlpha, 0.0, 1.0);

    outColor = color;
}