#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TAtomicRWStructuredBufferHandle<int> m_ActiveGridBlockCounter; 
        TRWStructuredBufferHandle<int> m_ActiveGridBlockList; 
        TStructuredBufferHandle<FSurfReconGrid> m_Grid; //SRV
        float m_KernelRadius; //H
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
#ifdef IFSHADER_SURFRECON_2D
    IFSHADER_BLOCKSIZE(kSurfReconFilterBlockTGSz2D, kSurfReconFilterBlockTGSz2D, 1)
#else
    IFSHADER_BLOCKSIZE(kSurfReconFilterBlockTGSz3D, kSurfReconFilterBlockTGSz3D, kSurfReconFilterBlockTGSz3D)
#endif
    void SurfReconFilterBlocksCS(uint3 ThreadId:SV_DispatchThreadID,
                              uint3 GroupId:SV_GroupID,
                              uint3 GroupThreadId:SV_GroupThreadID)
    {
        if(ThreadId.x==0 && ThreadId.y==0 && ThreadId.z==0)
        {
            PushConst.m_ActiveGridBlockCounter.Store(1,1);
            PushConst.m_ActiveGridBlockCounter.Store(1,2);
        }
        
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        FSpatialIndex MaxBlockIndex = Grid.GetMaxBlockIndex();
        FSpatialIndex CurBlockindex = ToSpatialIndex(ThreadId.x, ThreadId.y, ThreadId.z);

        bool IsValid = true;
        if(any(CurBlockindex < FSpatialIndex(0)) || any(CurBlockindex > MaxBlockIndex))
        {
            IsValid = false;
        }
        int BlockId = Grid.GetBlockFlattenId(CurBlockindex);
        if(IsValid)
        {
            //IsValid = Grid.IsBlockActive(BlockId); //&& (!Grid.IsBlockInternalExpanded(BlockId, int(ceil(PushConst.m_KernelRadius))));
        }

        int sWaveMarkedBlockListOffset = WavePrefixCountBits(IsValid);
        int sWaveNumBlocksToMark = WaveActiveCountBits(IsValid);
        int sWaveMarkedBlockListStart = 0;
        if(WaveIsFirstLane())
        {
            sWaveMarkedBlockListStart = PushConst.m_ActiveGridBlockCounter.AtomicAdd(0, sWaveNumBlocksToMark);
        }
        int BlockListStorePlace = WaveReadLaneFirst(sWaveMarkedBlockListStart) + sWaveMarkedBlockListOffset;
        
        if(IsValid)
        {
            PushConst.m_ActiveGridBlockList.Store(BlockId, BlockListStorePlace);
        }
    }

}}}
