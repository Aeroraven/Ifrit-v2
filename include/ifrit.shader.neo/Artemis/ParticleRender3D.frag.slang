#include "ifrit.shader.neo/Bindless.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T{
        float4x4 m_Transform;
        TRWStructuredBufferHandle<float4> m_Positions;
            TStructuredBufferHandle<float4> mColors;
        float mPointSize;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    struct PSOutput
    {
        IFSHADER_LOCATION(0) float4 Color: SV_Target0;
    };

    struct VSOutput
    {
        float4 Position: SV_Position;
        IFSHADER_LOCATION(0) float4 Color: COLOR0;
        IFSHADER_POINTSIZE_DECORATE float PointSize: IFSHADER_POINTSIZE_SEMANTIC;
    };

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PSOutput ArtemisParticleRender3DPS(VSOutput VsOut)
    {
        PSOutput Output;
        Output.Color = VsOut.Color;
        return Output;
    }
}
} 