
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
    uint colorSRV;
    uint normalSRV;
    uint depthSRV;
    int halfKernSize;
} pc;

float getConvWeight(vec2 locA,vec2 locB,vec3 colA,vec3 colB, vec3 normA, vec3 normB, float depthA, float depthB){
    float posWeight = exp(-dot(locA-locB,locA-locB)/2.0);
    float colorWeight = exp(-dot(colA-colB,colA-colB)/2.0);
    float normalWeight = exp(-dot(normA,normB)/2.0);
    float depthWeight = exp(-dot(depthA-depthB,depthA-depthB)/2.0);
    return posWeight*colorWeight*normalWeight*depthWeight;
}

void main(){
    // Joint Bilaterial Filter
    // K(i,j) = Kp(i,j)*Kp(C[i],C[j])*Kp(N[i],N[j])*Kp(D[i],D[j])
    // https://zhuanlan.zhihu.com/p/607012514

    vec2 fragCoord = vec2(gl_FragCoord.x, gl_FragCoord.y);
    vec3 colorSelf = texelFetch(GetSampler2D(pc.colorSRV), ivec2(fragCoord), 0).rgb;
    vec3 normalSelf = texelFetch(GetSampler2D(pc.normalSRV), ivec2(fragCoord), 0).rgb;
    normalSelf = normalSelf*2.0-1.0;
    float depthSelf = texelFetch(GetSampler2D(pc.depthSRV), ivec2(fragCoord), 0).r;

    float totalWeights = 0.0;
    
    vec4 retColor = vec4(0.0);
    for(int i=-pc.halfKernSize;i<=pc.halfKernSize;i++){
        for(int j=-pc.halfKernSize;j<=pc.halfKernSize;j++){
            vec3 colorNeigh = texelFetch(GetSampler2D(pc.colorSRV), ivec2(fragCoord)+ivec2(i,j), 0).rgb;
            vec3 normalNeigh = texelFetch(GetSampler2D(pc.normalSRV), ivec2(fragCoord)+ivec2(i,j), 0).rgb;
            normalNeigh = normalNeigh*2.0-1.0;
            float depthNeigh = texelFetch(GetSampler2D(pc.depthSRV), ivec2(fragCoord)+ivec2(i,j), 0).r;
            float weight = getConvWeight(fragCoord,fragCoord+vec2(i,j),colorSelf,colorNeigh,normalSelf,normalNeigh,depthSelf,depthNeigh);
            retColor += texelFetch(GetSampler2D(pc.colorSRV), ivec2(fragCoord)+ivec2(i,j), 0).rgba*weight;
            totalWeights += weight;
        }
    }
    outColor = retColor/totalWeights;
    outColor.a = 1.0;
}