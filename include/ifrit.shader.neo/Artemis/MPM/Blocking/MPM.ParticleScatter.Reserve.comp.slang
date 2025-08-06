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
        //TRWStructuredBufferHandle<int> m_BlockPageData; // store the block id
        TRWStructuredBufferHandle<int> m_BlockPageNumParticles; 
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmParticleScatterReserveCS(uint3 ThreadId:SV_DispatchThreadID,
                                    uint3 GroupId:SV_GroupID,
                                    uint3 GroupThreadId:SV_GroupThreadID)
    {
        FDenseBlockStructure Block = PushConst.m_Block.Load();
        int TotalBlockCount = Block.GetTotalBlockCount();
        int BlockId = ThreadId.x;

        if(ThreadId.x >= TotalBlockCount)
            return;

        int NumParticlesInBlock = Block.GetNumParticlesInBlock(BlockId);
        FMpmBlockPageAllocData AllocData = Block.AllocatePagesWithElementSize(NumParticlesInBlock, BlockId);
        Block.SetBlockStoreOffset(BlockId, AllocData.m_Offset);

        for(int i=0; i<AllocData.m_NumPages; ++i)
        {
            int PageId = AllocData.m_StartingPageId + i;
            int ParticleUsed = min(NumParticlesInBlock, kMpmBlockPageSize);
            NumParticlesInBlock -= ParticleUsed;
            
            PushConst.m_BlockPageNumParticles.Store(ParticleUsed, PageId);
            //PushConst.m_BlockPageData.Store(BlockId, PageId);
        }
    }

}}}
