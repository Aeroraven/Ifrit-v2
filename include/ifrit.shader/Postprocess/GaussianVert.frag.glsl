
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

layout(push_constant) uniform GaussianHoriParams{
    uint inputTexture;
    uint kernelSize;
} pc;

void main(){
    vec3 color = vec3(0.0);

    vec2 fragCoord = vec2(gl_FragCoord.x, gl_FragCoord.y);
    //texelFetch
    float offset = 1.0;
    float totalWeights = 0.0;
    for(int i = -int(pc.kernelSize) / 2; i <= int(pc.kernelSize) / 2; i++){
        float x = fragCoord.x;
        float y = fragCoord.y + float(i) * offset;
        float gaussianWeight = 1.0 / sqrt(2.0 * 3.14159265359 * 1.0) * exp(-float(i * i) / (2.0 * 1.0));
        totalWeights += gaussianWeight;
        color += texelFetch(GetSampler2D(pc.inputTexture), ivec2(x, y), 0).rgb * gaussianWeight;
    }
    outColor = vec4(color / totalWeights, 1.0);
}

