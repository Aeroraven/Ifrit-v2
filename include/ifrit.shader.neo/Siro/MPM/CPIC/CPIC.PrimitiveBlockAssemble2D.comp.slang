#include "ifrit.shader.neo/Siro/MPM/CPIC/CPIC.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        int m_NumPrimitives;
        TStructuredBufferHandle<int> m_IndirectNumPrimitives;
        TStructuredBufferHandle<FCpicRigidBoundary> m_RigidBoundaries;
        TRWStructuredBufferHandle<float2> m_VertexBuffer;
        TAtomicRWStructuredBufferHandle<int> m_DrawVertexCount; // view from indirect draw cmd

        FDenseGridStructureHandle m_Grid;
        FDenseBlockStructureHandle m_Block;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    float2x2 GetRot135PositiveMatrix()
    {
        return float2x2(-0.707106781f, -0.707106781f,
                        0.707106781f, -0.707106781f);
    }

    float2x2 GetRot135NegativeMatrix()
    {
        return float2x2(-0.707106781f, 0.707106781f,
                        -0.707106781f, -0.707106781f);
    }


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void CpicPrimitiveBlockAssemble2DCS(uint3 ThreadId:SV_DispatchThreadID,
                                        uint3 GroupId:SV_GroupID,
                                        uint3 GroupThreadId:SV_GroupThreadID)
    {
        // primitive assemble for rasterization:
        // line -> 2x triangle (6x vertices)

        int NumPrimitives = PushConst.m_NumPrimitives;
        if(NumPrimitives == 0x7fffffff)
        {
            NumPrimitives = PushConst.m_IndirectNumPrimitives.Load(0);
        }
        bool IsValidPrimitive = ThreadId.x < NumPrimitives;
        int ValidPrimitiveCnt = WaveActiveCountBits(IsValidPrimitive);
        int sWaveMarkedVertexListStart = 0;
        if(WaveIsFirstLane())
        {
            sWaveMarkedVertexListStart = PushConst.m_DrawVertexCount.AtomicAdd(0, ValidPrimitiveCnt * 6);
        }
        int sThreadMarkedVertexListStart = WaveReadLaneFirst(sWaveMarkedVertexListStart);
        int sThreadMarkedVertexListOffset = GroupThreadId.x * 6;
        int VertexBufferOffset = sThreadMarkedVertexListStart + sThreadMarkedVertexListOffset;

        if(!IsValidPrimitive)
        {
            return;
        }

        FCpicRigidBoundary RigidBoundary = PushConst.m_RigidBoundaries.Load(ThreadId.x);
        FDenseBlockStructure Block = PushConst.m_Block.Load();
        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        
        float2 FrameBufferSize = Block.m_BlockSize.xy;
        float2 DeltaUV_Block = rcp(FrameBufferSize);
        float2 DeltaUV_Grid = DeltaUV_Block * rcp(1.0f*kMpmBlockWidth);
        
        float2 SegCenter0 = RigidBoundary.m_BoundaryStart * DeltaUV_Block;
        float2 SegCenter1 = RigidBoundary.m_BoundaryEnd * DeltaUV_Block;

        float2 SegDir = normalize(SegCenter1 - SegCenter0);

        float2 SegDirRot135P = mul(GetRot135PositiveMatrix(), SegDir);
        float2 SegDirRot135N = mul(GetRot135NegativeMatrix(), SegDir);

        float2 V0 = SegCenter0 + SegDirRot135N * DeltaUV_Grid * 1.414f;
        float2 V1 = SegCenter0 + SegDirRot135P * DeltaUV_Grid * 1.414f;
        float2 V2 = SegCenter1 + SegDirRot135N * DeltaUV_Grid * 1.414f;
        float2 V3 = SegCenter1 + SegDirRot135P * DeltaUV_Grid * 1.414f;

        PushConst.m_VertexBuffer.Store(V0, VertexBufferOffset + 0);
        PushConst.m_VertexBuffer.Store(V1, VertexBufferOffset + 1);
        PushConst.m_VertexBuffer.Store(V2, VertexBufferOffset + 2);
        PushConst.m_VertexBuffer.Store(V0, VertexBufferOffset + 3);
        PushConst.m_VertexBuffer.Store(V2, VertexBufferOffset + 4);
        PushConst.m_VertexBuffer.Store(V3, VertexBufferOffset + 5);
    }

}}}