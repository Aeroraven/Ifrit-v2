
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

// Simple Temporary Anti-Aliasing
// Implementation might be incorrect, but it works for now.
// 
// Reference: https://zhuanlan.zhihu.com/p/425233743

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColorNextHistory;
layout(location = 1) out vec4 outColorFrameBuffer;

#include "Bindless.glsl"

layout(binding = 0, set = 1) uniform TAAHistory {
    uint frameUnresolved;
    uint frame1;
    uint frame2;
} uHistory;

layout(binding = 0, set = 2) uniform MotionDepthRefs{
    uint ref;
} uMotionDepthRefs;

layout(push_constant) uniform TAA {
    uint frame;
    uint renderWidth;
    uint renderHeight;
    float jitterX;
    float jitterY;
} uTAA;

vec3 rgbToYCoCg(vec3 rgb){
    float Y = rgb.r * 0.25 + rgb.g * 0.5 + rgb.b * 0.25;
    float Co = rgb.r * 0.5 - rgb.b * 0.5;
    float Cg = rgb.g - Y;
    return vec3(Y, Co, Cg);
}

vec3 yCoCgToRGB(vec3 yCoCg){
    float r = yCoCg.x + yCoCg.y - yCoCg.z;
    float g = yCoCg.x + yCoCg.z;
    float b = yCoCg.x - yCoCg.y - yCoCg.z;
    return vec3(r, g, b);
}

vec4 colorClamp(vec4 srcColor, vec4 historyColor, vec2 uv){
    vec3 colorAABBMin = vec3(1.0);
    vec3 colorAABBMax = vec3(0.0);
    for(int dx = -1; dx <= 1; dx++){
        for(int dy = -1; dy <= 1; dy++){
            vec2 offset = vec2(dx, dy) / vec2(uTAA.renderWidth, uTAA.renderHeight);
            vec4 neighborColor = texture(GetSampler2D(uHistory.frameUnresolved), uv + offset);
            vec3 neighborYCoCg = rgbToYCoCg(neighborColor.rgb);
            colorAABBMin = min(colorAABBMin, neighborYCoCg);
            colorAABBMax = max(colorAABBMax, neighborYCoCg);
        }
    }
    vec3 historyColorYCoCg = rgbToYCoCg(historyColor.rgb);
    vec3 clampedHistoryColorYCoCg = clamp(historyColorYCoCg, colorAABBMin, colorAABBMax);
    return vec4(yCoCgToRGB(clampedHistoryColorYCoCg), historyColor.a);
}

void main(){
    vec2 curFrameJitter = vec2(uTAA.jitterX, uTAA.jitterY);
    vec2 motionVector = texture(GetSampler2D(uMotionDepthRefs.ref), texCoord).rg;
    vec2 lastTexCoord = (texCoord) - motionVector;
    vec4 resolvedColor;

    float blendFactor = clamp(0.05 + length(motionVector)*114.514, 0.0, 1.0);
    if(uTAA.frame % 2 == 0){
        // read from frame1
        vec4 historyColor = texture(GetSampler2D(uHistory.frame2), lastTexCoord);
        vec4 frameColor = texture(GetSampler2D(uHistory.frameUnresolved), texCoord);
        historyColor = colorClamp(frameColor, historyColor, texCoord);
        resolvedColor = mix(historyColor, frameColor, blendFactor);
    }else{
        // read from frame2
        vec4 historyColor = texture(GetSampler2D(uHistory.frame1), lastTexCoord);
        vec4 frameColor = texture(GetSampler2D(uHistory.frameUnresolved), texCoord);
        historyColor = colorClamp(frameColor, historyColor, texCoord);
        resolvedColor = mix(historyColor, frameColor, blendFactor);
    }
    outColorNextHistory = resolvedColor;
    outColorFrameBuffer = resolvedColor;
}