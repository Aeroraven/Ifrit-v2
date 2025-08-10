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
        TRWStructuredBufferHandle<int> m_CellParticleIndices;
        TRWStructuredBufferHandle<float> m_VertexDensity;
        FRWSpatialVectorHandle m_CellVertexNormal;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    float GetVertexDensity(FSpatialIndex Idx,FSpatialIndex Offset, FSurfReconGrid Grid)
    {
        FSpatialIndex GridVertsBound = Grid.GetGridNumCellsInAxis()+1;
        FSpatialIndex ClampedIdx = clamp(Idx + Offset, FSpatialIndex(0), GridVertsBound - 1);
        int GridFlattenId = Grid.GetCellVertexFlattenId(ClampedIdx);
        return PushConst.m_VertexDensity.Load(GridFlattenId);
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSurfReconVertexDensityTGSz, 1, 1)
    void SurfReconVertexNormalBuildCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        FSpatialIndex CellsMaxBound = Grid.GetGridNumCellsInAxis()-1;
        int TotalActiveVertices = PushConst.m_ActiveVerticesCounter.Load(0);
        if(ThreadId.x >= TotalActiveVertices) return;
        int VertexIdx = PushConst.m_ActiveVerticesList.Load(ThreadId.x);

        FSpatialIndex VertexCoord = Grid.GetCellVertexFromFlattenId(VertexIdx);

        float Dx = 0.0f;
        float Dy = 0.0f;
        float Dz = 0.0f;
        FSpatialVector VertexNormal = FSpatialVector(0.0f);

#ifdef IFSHADER_SURFRECON_2D
        float XR = GetVertexDensity(VertexCoord, FSpatialIndex(1, 0), Grid);
        float XL = GetVertexDensity(VertexCoord, FSpatialIndex(-1, 0), Grid);
        float YR = GetVertexDensity(VertexCoord, FSpatialIndex(0, 1), Grid);
        float YL = GetVertexDensity(VertexCoord, FSpatialIndex(0, -1), Grid);
        Dx = (XR - XL) * 0.5f;
        Dy = (YR - YL) * 0.5f;
        VertexNormal = normalize(FSpatialVector(Dx, Dy));
#else
        float XR = GetVertexDensity(VertexCoord, FSpatialIndex(1, 0, 0), Grid);
        float XL = GetVertexDensity(VertexCoord, FSpatialIndex(-1, 0, 0), Grid);
        float YR = GetVertexDensity(VertexCoord, FSpatialIndex(0, 1, 0), Grid);
        float YL = GetVertexDensity(VertexCoord, FSpatialIndex(0, -1, 0), Grid);
        float ZR = GetVertexDensity(VertexCoord, FSpatialIndex(0, 0, 1), Grid);
        float ZL = GetVertexDensity(VertexCoord, FSpatialIndex(0, 0, -1), Grid);
        Dx = (XR - XL) * 0.5f;
        Dy = (YR - YL) * 0.5f;
        Dz = (ZR - ZL) * 0.5f;
        VertexNormal = normalize(FSpatialVector(Dx, Dy, Dz));
#endif
        PushConst.m_CellVertexNormal.Store(VertexIdx, VertexNormal);
    }

}}}
