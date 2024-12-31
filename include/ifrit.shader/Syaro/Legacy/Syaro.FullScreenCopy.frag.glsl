
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
layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 outColor;
layout(binding = 0, set = 1) uniform SamplerLocation{
    uint ref;
    uint pad0;
    uint pad1;
    uint pad2;
} uSampler;

void main(){
    ivec2 size = textureSize(GetSampler2D(uSampler.ref), 0);
    ivec2 coordInt = ivec2(texCoord * vec2(size));
    uint sampledVal = texelFetch(GetSampler2DU(uSampler.ref), coordInt, 0).r;

    uint x = sampledVal & 0x0000007Fu;
    float color = float(x) / 127.0;

    uint y = (sampledVal >> 7);
    float color2 = float(y) / 500.0;
    outColor = vec4(color, color2, 0.0, 1.0);
}