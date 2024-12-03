#version 450
#extension GL_GOOGLE_include_directive : require

#include "Base.glsl"
#include "Bindless.glsl"

layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec2 texCoord;

void main(){
    gl_Position = vec4(inPosition - vec2(1.0), 0.0, 1.0);
    texCoord = inPosition * 0.5 ;
}