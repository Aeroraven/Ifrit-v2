
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


layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

#include "Bindless.glsl"
#include "SamplerUtils.SharedConst.h"

layout(push_constant) uniform PushConstant{
    uint rayMarchResult;
} pc;

void main(){
    vec4 color = SampleTexture2D(pc.rayMarchResult,sLinearClamp,texCoord);
    outColor = vec4(color.xyz, 1.0);
}