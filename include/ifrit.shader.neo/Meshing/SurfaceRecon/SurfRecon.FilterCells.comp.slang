#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TStructuredBufferHandle<int> m_ActiveGridBlockList; 
        TStructuredBufferHandle<FSurfReconGrid> m_Grid; //SRV
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    void MarkSurfaceCell(FSpatialIndex cellCoord)
    {
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        Grid.MarkVertexBufferIndex(Grid.GetCellVertexFlattenId(cellCoord+ToSpatialIndex(0, 0, 0)));
        Grid.MarkVertexBufferIndex(Grid.GetCellVertexFlattenId(cellCoord+ToSpatialIndex(0, 1, 0)));
        Grid.MarkVertexBufferIndex(Grid.GetCellVertexFlattenId(cellCoord+ToSpatialIndex(1, 0, 0)));
        Grid.MarkVertexBufferIndex(Grid.GetCellVertexFlattenId(cellCoord+ToSpatialIndex(1, 1, 0)));
#ifndef IFSHADER_SURFRECON_2D
        Grid.MarkVertexBufferIndex(Grid.GetCellVertexFlattenId(cellCoord+ToSpatialIndex(0, 0, 1)));
        Grid.MarkVertexBufferIndex(Grid.GetCellVertexFlattenId(cellCoord+ToSpatialIndex(0, 1, 1)));
        Grid.MarkVertexBufferIndex(Grid.GetCellVertexFlattenId(cellCoord+ToSpatialIndex(1, 0, 1)));
        Grid.MarkVertexBufferIndex(Grid.GetCellVertexFlattenId(cellCoord+ToSpatialIndex(1, 1, 1)));
#endif  
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
#ifdef IFSHADER_SURFRECON_2D
    IFSHADER_BLOCKSIZE(kSurfReconFilterCellTGSz2D, kSurfReconFilterCellTGSz2D, 1)
#else
    IFSHADER_BLOCKSIZE(kSurfReconFilterCellTGSz3D, kSurfReconFilterCellTGSz3D, kSurfReconFilterCellTGSz3D)
#endif
    void SurfReconFilterCellsCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        FSpatialIndex InBlockCellOffset = ToSpatialIndex(GroupThreadId.x, GroupThreadId.y, GroupThreadId.z);
        int BlockCandidateId = GroupId.x;
        int BlockId = PushConst.m_ActiveGridBlockList.Load(BlockCandidateId);
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        FSpatialIndex BlockCoord = Grid.GetBlockCoordFromFlattenId(BlockId);
        FSpatialIndex CellCoord = BlockCoord * Grid.m_BlockWidthInCellUnits + InBlockCellOffset;

        FSpatialIndex MinCellBound = FSpatialIndex(0);
        FSpatialIndex MaxCellBound = ToSpatialIndex(Grid.m_NumCells);
    
        FSpatialIndex SearchLBound = max(MinCellBound, CellCoord - 1);
        FSpatialIndex SearchUBound = min(MaxCellBound, CellCoord + 1);

        int NumActiveNeighbors = 0;
        for(int dx = SearchLBound.x; dx <= SearchUBound.x; ++dx)
        {
            for(int dy = SearchLBound.y; dy <= SearchUBound.y; ++dy)
            {
                int dz = 0;
#ifndef IFSHADER_SURFRECON_2D
                for(dz = SearchLBound.z; dz <= SearchUBound.z; ++dz)
#endif
                {
                    FSpatialIndex SearchCellCoord = ToSpatialIndex(dx, dy, dz);
                    int SearchCellId = Grid.GetCellFlattenId(SearchCellCoord);
                    bool IsCellActive = Grid.IsCellActive(SearchCellId);
                    NumActiveNeighbors += IsCellActive ? 1 : 0;
                }
            }
        }
        int MaxActiveNeighbors = (SearchUBound.y - SearchLBound.y + 1) * (SearchUBound.x - SearchLBound.x + 1);
#ifndef IFSHADER_SURFRECON_2D
        MaxActiveNeighbors *= (SearchUBound.z - SearchLBound.z + 1);
#endif
        bool IsSurfaceCell = (NumActiveNeighbors > 0) && (NumActiveNeighbors < MaxActiveNeighbors);

        if(IsSurfaceCell)
        {
            MarkSurfaceCell(CellCoord);
        }
    }

}}}