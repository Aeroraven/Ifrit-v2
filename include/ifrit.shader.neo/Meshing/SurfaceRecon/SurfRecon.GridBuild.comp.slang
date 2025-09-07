#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        uint m_ParticleCounter;
        TStructuredBufferHandle<int> m_ParticleCounterIndirect; //SRV
        FSpatialVectorHandle m_ParticleLocation; //SRV
        TStructuredBufferHandle<FSurfReconGrid> m_Grid; //SRV
        TRWStructuredBufferHandle<int2> m_DebugData;
        TRWStructuredBufferHandle<int> m_CellParticleIndices; //UAV
        float m_KernelRadius; //H
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    uint GetParticleCount()
    {
        if(PushConst.m_ParticleCounter != ~0u)
        {
            return PushConst.m_ParticleCounter;
        }
        else
        {
           return (uint)PushConst.m_ParticleCounterIndirect.Load(0);
        }
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSurfReconGridBuildTGSz, 1, 1)
    void SurfReconGridBuildCS(uint3 ThreadId:SV_DispatchThreadID,
                              uint3 GroupId:SV_GroupID,
                              uint3 GroupThreadId:SV_GroupThreadID)
    {
        if(ThreadId.x >= GetParticleCount())
            return;

        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);

        bool IsInRange = Grid.IsInRange(ParticleLocation);
        if(!IsInRange)
            return;

        FSpatialIndex CellCoord = Grid.GetCellIndex(ParticleLocation);
        FSpatialIndex CellInBlockBoundaryState = Grid.GetCellInBlockBoundaryStateExpanded(CellCoord, int(ceil(PushConst.m_KernelRadius)));
        FSpatialIndex BlockCoord = Grid.GetBlockIndex(CellCoord);

        int CellId = Grid.GetCellFlattenId(CellCoord);
        int ParticleCountInCell = Grid.MarkCellActive(CellId);

        if(ParticleCountInCell < kSurfReconMaxParticlesPerCell)
        {
            PushConst.m_CellParticleIndices.Store(ThreadId.x, CellId * kSurfReconMaxParticlesPerCell + ParticleCountInCell);
        }

        int2 DebugData = int2(ParticleCountInCell, CellId);
        PushConst.m_DebugData.Store(DebugData, ThreadId.x);

        FSpatialIndex MinBound = max(FSpatialIndex(0), BlockCoord+min(CellInBlockBoundaryState,FSpatialIndex(0)));
        FSpatialIndex MaxBound = min(Grid.GetMaxBlockIndex(), BlockCoord+max(CellInBlockBoundaryState,FSpatialIndex(0)));

        if(ParticleCountInCell == 1)
        {
            for(int dx = MinBound.x; dx <= MaxBound.x; ++dx)
            {
                for(int dy = MinBound.y; dy <= MaxBound.y; ++dy)
                {
                    int dz = 0;
    #ifndef IFSHADER_SURFRECON_2D
                    for(dz = MinBound.z; dz <= MaxBound.z; ++dz)
    #endif
                    {
                        FSpatialIndex BlockCoord = ToSpatialIndex(dx, dy, dz);
                        int BlockId = Grid.GetBlockFlattenId(BlockCoord);
                        Grid.MarkBlockActive(BlockId);
                    }
                }
            }
        }
        
    }

}}}
