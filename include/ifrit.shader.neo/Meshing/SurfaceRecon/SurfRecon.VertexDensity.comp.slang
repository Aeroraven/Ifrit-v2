#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TStructuredBufferHandle<FSurfReconGrid> m_Grid; //SRV
        TAtomicRWStructuredBufferHandle<int> m_ActiveVerticesCounter;
        TRWStructuredBufferHandle<int> m_ActiveVerticesList;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSurfReconVertexDensityTGSz, 1, 1)
    void SurfReconVertexDensityCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        FSpatialIndex CellsMaxBound = Grid.GetGridNumCellsInAxis()-1;
        int TotalActiveVertices = PushConst.m_ActiveVerticesCounter.Load(0);
        if(ThreadId.x >= TotalActiveVertices) return;
        int VertexIdx = PushConst.m_ActiveVerticesList.Load(ThreadId.x);

        FSpatialIndex VertexCoord = Grid.GetCellVertexFromFlattenId(VertexIdx);
        FSpatialIndex MaxBound = min(CellsMaxBound,VertexCoord);
        FSpatialIndex MinBound = max(FSpatialIndex(0),VertexCoord-1);
        
        float TotalDensity = 0.0f;
        for(int dx = MinBound.x; dx <= MaxBound.x; ++dx)
        {
            for(int dy = MinBound.y; dy <= MaxBound.y; ++dy)
            {
                int dz = 0;
#ifndef IFSHADER_SURFRECON_2D
                for(dz = MinBound.z; dz <= MaxBound.z; ++dz)
#endif
                {
                    FSpatialIndex GridCoord = ToSpatialIndex(dx, dy, dz);
                    
                }
            }
        }

    }

}}}