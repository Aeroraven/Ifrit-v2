#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"

layout(push_constant) uniform DebugTriangleView{
    uint visBufferSRV;   
}pc;

layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec4 outColor;

vec4 colorLut[12]={
    vec4(1.0, 0.0, 0.0, 1.0),
    vec4(0.0, 1.0, 0.0, 1.0),
    vec4(0.0, 0.0, 1.0, 1.0),
    vec4(1.0, 1.0, 0.0, 1.0),
    vec4(1.0, 0.0, 1.0, 1.0),
    vec4(0.0, 1.0, 1.0, 1.0),
    vec4(1.0, 1.0, 1.0, 1.0),
    vec4(0.5, 0.0, 0.0, 1.0),
    vec4(0.0, 0.5, 0.0, 1.0),
    vec4(0.0, 0.0, 0.5, 1.0),
    vec4(0.5, 0.5, 0.0, 1.0),
    vec4(0.5, 0.0, 0.5, 1.0)
};

void main(){
    uint vis = texture(GetSampler2DU(pc.visBufferSRV), inPosition).r;
    uint val = (vis & 0x7F) % 12;
    outColor = colorLut[val];
}