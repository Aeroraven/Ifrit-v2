#include "ifrit.shader.neo/Artemis/MPM/CPIC/CPIC.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        TStructuredBufferHandle<float2> m_VertexBuffer;
    };

    struct VSOutput
    {
        float4 Position:SV_Position;
        nointerpolation int RigidId:TEXCOORD0; 
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_VS_ENTRY)
    VSOutput CpicPrimitiveBlockRasterize2DVS(uint VertexId:SV_VertexID)
    {
        VSOutput Output;
        float2 Vertex = PushConst.m_VertexBuffer.Load(VertexId);
        Vertex = Vertex * 2.0f - 1.0f; 
        Output.Position = float4(Vertex, 0.5f, 1.0f);
        Output.RigidId = VertexId / 6;
        return Output;
    }

}}}
