// #include "Bindless.glsl"
// #include "SamplerUtils.SharedConst.h"

// layout(location = 0) in vec2 texCoord;
// layout(location = 0) out vec4 outColor;

// layout(push_constant) uniform GaussianHoriParams{
//     uint inputTexture;
//     uint kernelSize;
// } pc;

// void main(){
//     vec4 color = vec4(0.0);
//     vec2 fragCoord = vec2(gl_FragCoord.x, gl_FragCoord.y);
//     float offset = 1.0;
//     float totalWeights = 0.0;
//     for(int i = -int(pc.kernelSize) / 2; i <= int(pc.kernelSize) / 2; i++){
//         float x = fragCoord.x + float(i) * offset;
//         float y = fragCoord.y;
//         float gaussianWeight = 1.0 / sqrt(2.0 * 3.14159265359 * 1.0) * exp(-float(i * i) / (2.0 * 1.0));
//         totalWeights += gaussianWeight;
//         color += SampleTexture2DLoad(pc.inputTexture,sLinearClamp, ivec2(x, y)).rgba * gaussianWeight;
//     }
//     outColor = vec4(color / totalWeights);
// }


#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"

namespace IfritShader
{
    struct PushConst_T
    {
        TTexture2DHandle<float4> m_SrcTex;
    };

    struct PSOutput
    {
        float4 m_Color : SV_Target0;
    };

    struct VSOutput
    {
        float2 m_TexCoord : TEXCOORD0;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PSOutput ResolveToSwapchainPS(VSOutput Input)
    {
        PSOutput Output;
        Output.m_Color = PushConst.m_SrcTex.Sample(ESamplerType::SLinearClamp, Input.m_TexCoord);
        Output.m_Color.a = 1.0f; 
        return Output;
    }

}