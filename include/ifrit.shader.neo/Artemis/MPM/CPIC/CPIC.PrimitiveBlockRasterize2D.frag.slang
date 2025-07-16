#include "ifrit.shader.neo/Artemis/MPM/CPIC/CPIC.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        TStructuredBufferHandle<float2> m_VertexBuffer;
        FDenseGridStructureHandle m_Grid;
        FDenseBlockStructureHandle m_Block;
        TAtomicRWStructuredBufferHandle<int> m_CpicBlockRigidCounter;
        TRWStructuredBufferHandle<int> m_CpicBlockRigidList;
    };

    struct VSOutput
    {
        float4 Position:SV_Position;
        nointerpolation int RigidId:TEXCOORD0; 
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    void CpicPrimitiveBlockRasterize2DPS(VSOutput Input)
    {
        float4 ScreenPos = Input.Position;
        FDenseBlockStructure Block = PushConst.m_Block.Load();
        FDenseGridStructure Grid = PushConst.m_Grid.Load();

        float2 RTSize = Block.m_BlockSize.xy;

        int2 BlockCoord = int2(int(ScreenPos.x),int(ScreenPos.y));
        int BlockId = Block.GetBlockId(BlockCoord);

        int Offset = PushConst.m_CpicBlockRigidCounter.AtomicAdd(BlockId, 1);
        if(Offset >= kCpicMaxRigidPerCell)
        {
            return; 
        }
        PushConst.m_CpicBlockRigidList.Store(EncodeBlockRigidIdList(BlockId, Input.RigidId), Offset);
    }

}}}
