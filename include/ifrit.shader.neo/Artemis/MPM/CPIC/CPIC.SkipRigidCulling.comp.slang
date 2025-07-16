#include "ifrit.shader.neo/Artemis/MPM/CPIC/CPIC.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        int m_RigidCount;
        TStructuredBufferHandle<int> m_IndirectRigidCount;
        FDenseGridStructureHandle m_Grid;
        FDenseBlockStructureHandle m_Block;
        TAtomicRWStructuredBufferHandle<int> m_CpicBlockRigidCounter;
        TRWStructuredBufferHandle<int> m_CpicBlockRigidList;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kCpicMaxRigidPerCell,1,1)
    void CpicSkipRigidCullingPS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        FSpatialIndex BlockCoord = ToSpatialIndex(GroupId.x, GroupId.y, GroupId.z);
        FDenseBlockStructure Block = PushConst.m_Block.Load();
        FDenseGridStructure Grid = PushConst.m_Grid.Load();

        int BlockId = Block.GetBlockId(BlockCoord);
        int RigidCount = PushConst.m_RigidCount;
        if(RigidCount == kInt32Max)
        {
            RigidCount = PushConst.m_IndirectRigidCount.Load(BlockId);
        }
        if(ThreadId.x == 0)
        {
            PushConst.m_CpicBlockRigidCounter.Store(RigidCount, BlockId);
        }

        PushConst.m_CpicBlockRigidList.Store(ThreadId.x, EncodeBlockRigidIdList(BlockId, ThreadId.x));
    }

}}}
