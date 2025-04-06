
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

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform UPushConstant{
    vec4 m_LightDir; // In WS
    uint m_NormalSRV; // In VS
    uint m_PerFrameId;
    uint m_ShadowMapSRV;
} PushConst;

RegisterUniform(BPerFrameData,{
    PerFramePerViewData data;
});


void main(){
    mat4 viewToWorld = GetResource(BPerFrameData, PushConst.m_PerFrameId).data.m_viewToWorld;
    vec3 normal = texture(GetSampler2D(PushConst.m_NormalSRV), texCoord).xyz;
    if(normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0){
        outColor = vec4(1.0, 1.0, 0.0, 1.0);
        return;
    }
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize((viewToWorld * vec4(normal, 0.0)).xyz);

    float shadow = texture(GetSampler2D(PushConst.m_ShadowMapSRV), texCoord).r;
    vec3 light = normalize(PushConst.m_LightDir.xyz);
    float dotProduct = max(0.0, dot(normal, -light))* shadow;

    vec3 color = vec3(dotProduct, dotProduct, dotProduct)+ 0.1;
    outColor = vec4(color, 1.0);
}