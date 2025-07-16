#include "ifrit.shader.neo/Bindless.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T{
        float4x4 m_Transform;
        TRWStructuredBufferHandle<float4> m_Positions;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    struct VSOutput
    {
        float4 Position: SV_Position;
        IFSHADER_POINTSIZE_DECORATE float PointSize: IFSHADER_POINTSIZE_SEMANTIC;
    };

    IFSHADER_ENTRY(IFSHADER_VS_ENTRY)
    VSOutput ArtemisParticleRender3DVS(uint VertexId: SV_VertexID)
    {
        float4 Position = PushConst.m_Positions.Load(VertexId);
        float4 TransformedPosition = mul(PushConst.m_Transform, float4(Position.xyz, 1.0f));

        VSOutput Output;
        Output.Position = TransformedPosition;
        Output.PointSize = 1.0f; 
        return Output;
    }
}
} 