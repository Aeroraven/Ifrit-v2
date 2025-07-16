#include "ifrit.shader.neo/Bindless.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T{
        float4x4 m_Transform;
        TRWStructuredBufferHandle<float4> m_Positions;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    struct PSOutput
    {
        IFSHADER_LOCATION(0) float4 Color: SV_Target0;
    };

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PSOutput ArtemisParticleRender3DPS()
    {
        PSOutput Output;
        Output.Color = float4(1.0f, 1.0f, 1.0f, 1.0f); 
        return Output;
    }
}
} 