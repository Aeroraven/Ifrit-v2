#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS

#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        FParticleCounterHandle m_ParticleCounter;
        FDenseBlockStructureHandle m_Block;
        FDenseGridStructureHandle m_Grid;
        FSpatialVectorHandle m_ParticleLocation;
        TRWStructuredBufferHandle<int> m_ParticleOffset;
        FRWSpatialVectorHandle m_ParticleDebug;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmParticleScatterCountCS(uint3 ThreadId:SV_DispatchThreadID,
                                    uint3 GroupId:SV_GroupID,
                                    uint3 GroupThreadId:SV_GroupThreadID)
    {
        FDenseBlockStructure Block = PushConst.m_Block.Load();
        FDenseGridStructure Grid = PushConst.m_Grid.Load();

        if(ThreadId.x >= PushConst.m_ParticleCounter.GetNumParticles())
            return;

        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FSpatialIndex GridIndex = Grid.GetCentralGridRounded(ParticleLocation);
        FSpatialIndex BlockCoord = Block.GetBlockCoord(GridIndex);

        PushConst.m_ParticleDebug.Store(ThreadId.x, BlockCoord * 1.0f);

        int BlockId = Block.GetBlockId(BlockCoord);
        int Offset = Block.ScatterToBlock(BlockId);
        PushConst.m_ParticleOffset.Store(Offset, ThreadId.x);
    }

}}}
