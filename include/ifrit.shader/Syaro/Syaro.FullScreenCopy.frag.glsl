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